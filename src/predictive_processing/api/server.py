"""FastAPI Backend Server for Predictive Processing Multimodal Demonstrator.

Provides REST endpoints for:
1. IEMOCAP Dataset exploration (clips, transcripts, audio streaming, rater baselines).
2. Closed-form Bayesian PWP calculation (Level-1 sensory fusion + Level-2 context fusion).
3. Live multimodal conversational turn execution (WebM video decoding -> Whisper STT ->
   GRU prior -> Wav2Vec2 audio -> EmoNet vision -> PWP -> LLM -> pyCub gesture -> Kokoro TTS).
"""

from __future__ import annotations

import base64
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path
from typing import Any
import numpy as np


import torch
from fastapi import APIRouter, FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel

PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from predictive_processing.core.bayesian_engine import (
    hierarchical_posterior_params,
    posterior_params,
    precision,
    sensory_fusion_params,
    sigma_from_precision,
)
from predictive_processing.core.clarification_policy import (
    combine_policy_decisions,
    decide_dimension_policy,
    dialogue_context_is_warming_up,
    dialogue_context_warmup_policy,
)
from predictive_processing.core.config import ExperimentConfig
from predictive_processing.core.demo_models import BEST_CV5_GRU, BEST_CV5_WAV2VEC2
from predictive_processing.core.terminology import (
    ACOUSTIC_LIKELIHOOD,
    DEPLOYABLE_SPEAKER_HISTORY_PRIOR,
    DEPLOYABLE_SPEAKER_HISTORY_PRIOR_MULTIMODAL,
    DEPLOYABLE_SPEAKER_HISTORY_PRIOR_VISUAL,
    DIALOGUE_CONTEXT_PRIOR,
    FLAT_PRIOR,
    MULTIMODAL_SENSORY_LIKELIHOOD,
    RATER_SPECIFIC_BASELINE_PRIOR,
    VISUAL_LIKELIHOOD,
    demo_distribution,
    display_name,
)
from predictive_processing.live.live_expectation import predict_dialogue_context_prior
from predictive_processing.live.live_icub_client import check_icub_server_health, notify_icub_gesture
from predictive_processing.live.live_inference import predict_acoustic_likelihood
from predictive_processing.live.live_llm import (
    build_icub_system_prompt,
    extract_gesture_and_text,
    generate_llm_response,
)
from predictive_processing.live.live_logger import (
    clear_logged_turns,
    delete_last_turn,
    get_logged_turns,
    log_turn,
)
from predictive_processing.live.live_stt import transcribe_audio_bytes
from predictive_processing.live.live_tts import synthesize_speech
from predictive_processing.live.live_visual_inference import predict_visual_likelihood

# ── Paths & Data ─────────────────────────────────────────────────────────────

DATA_PATH_CV5_CASES = PROJECT_ROOT / "outputs" / "demo_cv5_cases_v2.json"
DATA_PATH_V2 = PROJECT_ROOT / "outputs" / "demo_data_v2.json"
CLIPS_PATH = PROJECT_ROOT / "demo_clips.json"
RATER_BASELINES_PATH = PROJECT_ROOT / "outputs" / "rater_baselines.json"
DEMO_AUDIO_DIR = PROJECT_ROOT / "demo_audio"

config = ExperimentConfig()

router = APIRouter(prefix="/api")

app = FastAPI(
    title="Predictive Processing Multimodal API",
    description="Backend API for interactive Bayesian affect perception and live multimodal HRI.",
    version="2.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
        "http://localhost:3000",
        "http://127.0.0.1:3000",
        "http://localhost:8000",
        "http://127.0.0.1:8000",
    ],
    allow_origin_regex=r"https?://(localhost|127\.0\.0\.1)(:\d+)?",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# In-memory datasets
_DEMO_DATA: list[dict] = []
_CURATED_CLIPS: list[dict] = []
_RATER_BASELINES: dict[str, dict] = {}
_LOOKUP_BY_ID: dict[str, dict] = {}


def _init_data():
    global _DEMO_DATA, _CURATED_CLIPS, _RATER_BASELINES, _LOOKUP_BY_ID
    source = DATA_PATH_CV5_CASES if DATA_PATH_CV5_CASES.exists() else DATA_PATH_V2
    if source.exists():
        with open(source, "r", encoding="utf-8") as f:
            _DEMO_DATA = json.load(f)
        _LOOKUP_BY_ID = {e["utterance_id"]: e for e in _DEMO_DATA}

    if CLIPS_PATH.exists():
        with open(CLIPS_PATH, "r", encoding="utf-8") as f:
            _CURATED_CLIPS = json.load(f)

    if RATER_BASELINES_PATH.exists():
        with open(RATER_BASELINES_PATH, "r", encoding="utf-8") as f:
            _RATER_BASELINES = json.load(f)


_init_data()


# ── Pydantic Request Models ──────────────────────────────────────────────────

class BayesianCalcRequest(BaseModel):
    dimension: str  # "Valence" or "Arousal"
    mu_prior: float
    tau_prior: float
    mu_audio: float
    tau_audio: float
    mu_visual: float
    tau_visual: float
    modality_mode: str = "multimodal"  # "multimodal", "audio_only", "visual_only"


# ── REST Endpoints on router ──────────────────────────────────────────────────


@router.get("/llm/models")
def get_llm_models():
    """Return list of available local models (Ollama or OpenAI-compatible)."""
    from predictive_processing.live.live_llm import get_available_ollama_models
    api_base = config.openai_api_base
    installed = []
    if "11434" in str(api_base):
        installed = get_available_ollama_models(api_base)
    return {
        "provider": "ollama" if "11434" in str(api_base) else "openai",
        "api_base": api_base,
        "models": installed,
        "active_model": config.openai_model_name,
    }

@router.get("/health")
def get_health():
    """Health check for all subsystem dependencies."""
    icub_online = False
    try:
        icub_online = check_icub_server_health()
    except Exception:
        pass

    return {
        "status": "online",
        "cuda_available": bool(os.environ.get("CUDA_VISIBLE_DEVICES", True)),
        "pycub_connected": icub_online,
        "dataset_records_loaded": len(_DEMO_DATA),
        "curated_clips_loaded": len(_CURATED_CLIPS),
        "whisper_model_size": config.whisper_model_size,
    }


@router.get("/clips")
def get_clips():
    """Return curated clip list and full utterance search catalog."""
    all_summaries = [
        {
            "utterance_id": e["utterance_id"],
            "dialog_id": e.get("dialog_id", ""),
            "turn_index": e.get("turn_index", 0),
            "speaker": e.get("speaker", ""),
            "emotion": e.get("emotion", "neu"),
        }
        for e in _DEMO_DATA
    ]
    return {
        "curated_clips": _CURATED_CLIPS,
        "all_clips": all_summaries,
    }


@router.get("/clip/{utterance_id}")
def get_clip_details(utterance_id: str):
    """Return full context, transcripts, per-rater scores, and distributions for one turn."""
    clip = _LOOKUP_BY_ID.get(utterance_id)
    if clip is None:
        raise HTTPException(status_code=404, detail=f"Utterance {utterance_id} not found")

    curated_meta = next((c for c in _CURATED_CLIPS if c["utterance_id"] == utterance_id), None)
    return {
        "record": clip,
        "curated_info": curated_meta,
        "audio_url": f"/api/audio/{utterance_id}",
    }


@router.get("/audio/{utterance_id}")
def stream_audio(utterance_id: str):
    """Stream WAV audio file for an utterance."""
    local_path = DEMO_AUDIO_DIR / f"{utterance_id}.wav"
    if local_path.exists():
        return FileResponse(local_path, media_type="audio/wav")

    clip = _LOOKUP_BY_ID.get(utterance_id)
    if clip and "wav_path" in clip:
        orig = Path(clip["wav_path"])
        if orig.exists():
            return FileResponse(orig, media_type="audio/wav")

    raise HTTPException(status_code=404, detail="Audio file not found")


@router.get("/rater-baselines")
def get_rater_baselines():
    """Return annotator counterfactual baseline prior distributions."""
    return _RATER_BASELINES


@router.post("/bayesian/calculate")
def calculate_bayesian(req: BayesianCalcRequest):
    """Compute Level-1 sensory fusion, Level-2 PWP, and policy decisions."""
    sigma_p = sigma_from_precision(req.tau_prior)
    sigma_a = sigma_from_precision(req.tau_audio)
    sigma_v = sigma_from_precision(req.tau_visual)

    if req.modality_mode == "audio_only":
        unimodal = posterior_params(
            mu_prior=req.mu_prior,
            sigma_prior=sigma_p,
            mu_likelihood=req.mu_audio,
            sigma_likelihood=sigma_a,
        )
        pol_dim = decide_dimension_policy(
            dimension=req.dimension,
            mu_prior=req.mu_prior,
            sigma_prior=sigma_p,
            mu_likelihood=req.mu_audio,
            sigma_likelihood=sigma_a,
            posterior_sigma=unimodal["sigma_post"],
        )
        pol = combine_policy_decisions([pol_dim])
        return {
            "mode": "audio_only",
            "mu_post": unimodal["mu_post"],
            "sigma_post": unimodal["sigma_post"],
            "tau_prior": req.tau_prior,
            "tau_audio": req.tau_audio,
            "tau_visual": 0.0,
            "tau_sensory": req.tau_audio,
            "tau_post": unimodal["tau_post"],
            "w_prior": unimodal["w_prior"],
            "w_audio": unimodal["w_likelihood"],
            "w_visual": 0.0,
            "w_sensory": unimodal["w_likelihood"],
            "w_audio_sensory": 1.0,
            "w_visual_sensory": 0.0,
            "mu_sensory": req.mu_audio,
            "sigma_sensory": sigma_a,
            "prior_sensory_discrepancy": unimodal["prediction_error"],
            "prior_sensory_standardized_discrepancy": pol_dim.standardized_discrepancy,
            "cross_sensory_discrepancy": 0.0,
            "cross_sensory_standardized_discrepancy": 0.0,
            "policy": {
                "action": pol.action.value,
                "label": pol.label,
                "badge_color": pol.badge_color,
                "icon": pol.icon,
                "gesture": pol.gesture_override or "neutral",
                "guidance": pol.response_guidance,
                "clarification_prompt": pol.clarification_prompt if pol.action.value == "clarify" else None,
                "triggers": list(pol.triggers),
            },
        }

    # Multimodal mode
    result = hierarchical_posterior_params(
        mu_prior=req.mu_prior,
        sigma_prior=sigma_p,
        mu_audio=req.mu_audio,
        sigma_audio=sigma_a,
        mu_visual=req.mu_visual,
        sigma_visual=sigma_v,
    )

    pol_dim = decide_dimension_policy(
        dimension=req.dimension,
        mu_prior=req.mu_prior,
        sigma_prior=sigma_p,
        mu_likelihood=result["mu_sensory"],
        sigma_likelihood=result["sigma_sensory"],
        posterior_sigma=result["sigma_post"],
    )
    pol = combine_policy_decisions([pol_dim])

    return {
        "mode": "multimodal",
        **result,
        "policy": {
            "action": pol.action.value,
            "label": pol.label,
            "badge_color": pol.badge_color,
            "icon": pol.icon,
            "gesture": pol.gesture_override or "neutral",
            "guidance": pol.response_guidance,
            "clarification_prompt": pol.clarification_prompt if pol.action.value == "clarify" else None,
            "triggers": list(pol.triggers),
        },
    }


def compute_speaker_history_prior(history: list[dict], source_key: str, dim_key: str) -> tuple[float, float]:
    """Extract past estimates from conversational history for speaker history baseline."""
    values = []
    for msg in history:
        if msg.get("role") == "assistant" and msg.get("dimension_results"):
            dr = msg["dimension_results"].get(dim_key, {})
            if source_key == "sensory" and "mu_sensory" in dr:
                values.append(dr["mu_sensory"])
            elif source_key == "audio" and "mu_audio" in dr:
                values.append(dr["mu_audio"])
            elif source_key == "visual" and "mu_visual" in dr:
                values.append(dr["mu_visual"])
        elif msg.get(f"{source_key}_{dim_key}") is not None:
            values.append(float(msg[f"{source_key}_{dim_key}"]))

    if not values:
        return 0.0, 0.50

    mu = float(np.mean(values))
    if len(values) > 1:
        sig = float(max(0.18, np.std(values)))
    else:
        sig = 0.35
    return mu, sig


@router.post("/live/process-turn")
async def process_live_turn(
    video: UploadFile = File(...),
    chat_history: str = Form("[]"),
    prior_type: str = Form("dialogue_context_prior"),
    whisper_size: str = Form("small"),
    tts_voice: str = Form("af_heart"),
    language: str = Form("en"),
    enable_logging: bool = Form(True),
    turn_idx: int = Form(1),
    llm_model: str | None = Form(None),
    llm_api_base: str | None = Form(None),
    llm_api_key: str | None = Form(None),
    fallback_mode: str = Form("error"),
):
    """Execute live conversational turn with native video+audio MediaRecorder input in English."""
    t_turn_start = time.perf_counter()
    video_bytes = await video.read()
    if not video_bytes:
        raise HTTPException(status_code=400, detail="Empty video file received")

    history = []
    try:
        history = json.loads(chat_history)
    except Exception:
        pass

    target_lang = "en"

    try:
        # 1. Transcribe speech using faster-whisper (English)
        user_text = transcribe_audio_bytes(
            video_bytes,
            model_size=whisper_size,
            device="cuda" if torch.cuda.is_available() else "auto",
            language="en",
        )
        if not user_text:
            user_text = "[Spoken turn recorded]"
        t_stt = time.perf_counter()

        # 2. Compute Context Prior (GRU) from past transcripts
        past_transcripts = [m["text"] for m in history if m.get("text")]
        current_speaker_turns = sum(1 for m in history if m.get("role") == "user") + 1

        expectation_prior = predict_dialogue_context_prior(
            past_transcripts,
            context_window=BEST_CV5_GRU.context_window or 10,
            checkpoint_path=str(BEST_CV5_GRU.checkpoint_path),
        )
        if not expectation_prior:
            expectation_prior = {
                "valence_mu": 0.0,
                "valence_sigma": 0.50,
                "arousal_mu": 0.0,
                "arousal_sigma": 0.50,
            }

        # 3. Acoustic Likelihood (Wav2Vec2)
        acoustic = predict_acoustic_likelihood(
            video_bytes,
            checkpoint_path=str(BEST_CV5_WAV2VEC2.checkpoint_path),
            max_seconds=10.0,
            normalize_amplitude=True,
            pad_to_max_seconds=True,
        ) or {
            "valence_mu": 0.0,
            "valence_sigma": 0.40,
            "arousal_mu": 0.0,
            "arousal_sigma": 0.40,
        }

        # 4. Visual Likelihood (EmoNet)
        visual = predict_visual_likelihood(video_bytes) or {
            "valence_mu": 0.0,
            "valence_sigma": 0.40,
            "arousal_mu": 0.0,
            "arousal_sigma": 0.40,
        }

        # 5. Multimodal Bayesian Integration & 2D Policy Decision
        dimension_decisions = []
        dimension_results = {}
        for dim_key, dim_name in (("valence", "Valence"), ("arousal", "Arousal")):
            if prior_type in (DIALOGUE_CONTEXT_PRIOR, "C", "context"):
                p_mu = expectation_prior.get(f"{dim_key}_mu", 0.0)
                p_sig = expectation_prior.get(f"{dim_key}_sigma", 0.50)
                prior_name = "Dialogue Context C (GRU)"
            elif prior_type in (DEPLOYABLE_SPEAKER_HISTORY_PRIOR_MULTIMODAL, "H(AV)", "multimodal_history"):
                p_mu, p_sig = compute_speaker_history_prior(history, "sensory", dim_key)
                prior_name = "Speaker History H(AV)"
            elif prior_type in (DEPLOYABLE_SPEAKER_HISTORY_PRIOR, "H(A)", "audio_history"):
                p_mu, p_sig = compute_speaker_history_prior(history, "audio", dim_key)
                prior_name = "Speaker History H(A)"
            elif prior_type in (DEPLOYABLE_SPEAKER_HISTORY_PRIOR_VISUAL, "H(V)", "visual_history"):
                p_mu, p_sig = compute_speaker_history_prior(history, "visual", dim_key)
                prior_name = "Speaker History H(V)"
            else:
                p_mu = 0.0
                p_sig = 0.50
                prior_name = "Flat Prior"

            a_mu = acoustic.get(f"{dim_key}_mu", 0.0)
            a_sig = acoustic.get(f"{dim_key}_sigma", 0.40)
            v_mu = visual.get(f"{dim_key}_mu", 0.0)
            v_sig = visual.get(f"{dim_key}_sigma", 0.40)

            dim_res = hierarchical_posterior_params(
                mu_prior=p_mu,
                sigma_prior=p_sig,
                mu_audio=a_mu,
                sigma_audio=a_sig,
                mu_visual=v_mu,
                sigma_visual=v_sig,
            )
            dim_res["mu_prior"] = p_mu
            dim_res["sigma_prior"] = p_sig
            dim_res["prior_name"] = prior_name
            dim_res["prior_type"] = prior_type
            dim_res["mu_audio"] = a_mu
            dim_res["sigma_audio"] = a_sig
            dim_res["mu_visual"] = v_mu
            dim_res["sigma_visual"] = v_sig

            dimension_results[dim_key] = dim_res

            decision = decide_dimension_policy(
                dimension=dim_name,
                mu_prior=p_mu,
                sigma_prior=p_sig,
                mu_likelihood=dim_res["mu_sensory"],
                sigma_likelihood=dim_res["sigma_sensory"],
                posterior_sigma=dim_res["sigma_post"],
            )
            dimension_decisions.append(decision)

        # Warmup gating check
        if (
            prior_type == DIALOGUE_CONTEXT_PRIOR
            and dialogue_context_is_warming_up(current_speaker_turns)
        ):
            combined_policy = dialogue_context_warmup_policy(
                dimension_decisions, speaker_turn_count=current_speaker_turns
            )
        else:
            combined_policy = combine_policy_decisions(dimension_decisions)
        t_bayes = time.perf_counter()

        # 6. LLM Response Generation & Gesture Selection
        system_prompt = build_icub_system_prompt(combined_policy.response_guidance, language=target_lang)
        llm_messages = [{"role": "system", "content": system_prompt}]
        for msg in history:
            llm_messages.append({"role": msg["role"], "content": msg["text"]})
        llm_messages.append({"role": "user", "content": user_text})

        selected_gesture = combined_policy.gesture_override or "nod"
        assistant_text = ""
        llm_error = None

        target_api_base = (llm_api_base or config.openai_api_base or "http://127.0.0.1:11434/v1").rstrip("/")
        target_api_key = llm_api_key or config.openai_api_key or "sk-no-key-required"
        target_model = llm_model or config.openai_model_name

        try:
            raw_llm_response = generate_llm_response(
                messages=llm_messages,
                api_base=target_api_base,
                api_key=target_api_key,
                model_name=target_model,
            )
            parsed_gesture, parsed_text = extract_gesture_and_text(raw_llm_response)
            if parsed_text:
                assistant_text = parsed_text
            if parsed_gesture:
                selected_gesture = combined_policy.gesture_override or parsed_gesture
        except Exception as exc:
            llm_error = str(exc)
            print(f"LLM generation notice: {exc}")

        if not assistant_text:
            if combined_policy.action.value == "clarify":
                assistant_text = (
                    combined_policy.clarification_prompt
                    or "I want to make sure I understand how that felt. Could you share a bit more?"
                )
            elif combined_policy.action.value == "cautious":
                assistant_text = "I hear what you're saying, though I'm sensing mixed cues. Please go on."
            else:
                if fallback_mode == "canned":
                    assistant_text = "Thank you for sharing that with me. I understand."
                else:
                    err_msg = llm_error or "LLM returned empty or connection failed"
                    assistant_text = f"[No text generated - {err_msg}]"

        # 7. Dispatch physical gesture to pyCub server
        try:
            notify_icub_gesture(selected_gesture, duration=2.5, start_delay=0.0)
        except Exception as exc:
            print(f"pyCub dispatch notice: {exc}")

        # 8. Synthesize TTS Audio with Kokoro-82M (English)
        assistant_audio_b64 = None
        should_synthesize = bool(assistant_text) and not assistant_text.startswith("[No text generated")
        if should_synthesize:
            try:
                val_val = dimension_results["valence"]["mu_post"]
                aro_val = dimension_results["arousal"]["mu_post"]
                pe_val = max(
                    dimension_results["valence"]["prior_sensory_discrepancy"],
                    dimension_results["arousal"]["prior_sensory_discrepancy"],
                )

                selected_voice = tts_voice or config.kokoro_voice_en
                kokoro_lang = "b" if selected_voice.startswith("b") else "a"

                tts_bytes = synthesize_speech(
                    text=assistant_text,
                    voice=selected_voice,
                    lang_code=kokoro_lang,
                    valence=float(val_val),
                    arousal=float(aro_val),
                    prediction_error=float(pe_val),
                )
                if tts_bytes:
                    assistant_audio_b64 = f"data:audio/wav;base64,{base64.b64encode(tts_bytes).decode('utf-8')}"
            except Exception as exc:
                print(f"Kokoro TTS generation error: {exc}")

        t_tts = time.perf_counter()
        total_latency_ms = (t_tts - t_turn_start) * 1000

        # 9. Auto-log interaction turn if enabled
        turn_log = None
        if enable_logging:
            try:
                turn_log = log_turn(
                    turn_idx=turn_idx,
                    user_text=user_text,
                    assistant_text=assistant_text,
                    selected_gesture=selected_gesture,
                    dimension_results=dimension_results,
                    policy=combined_policy,
                    latency_ms=total_latency_ms,
                    timing_breakdown_ms={
                        "stt": (t_stt - t_turn_start) * 1000,
                        "prior": (t_bayes - t_stt) * 0.35 * 1000,  # approximate stage split
                        "bayes": (t_bayes - t_stt) * 0.65 * 1000,
                        "llm": (t_tts - t_bayes - (t_tts - t_bayes) * 0.3) * 1000,
                        "tts": (t_tts - t_bayes) * 0.3 * 1000,
                    },
                )
            except Exception as log_err:
                print(f"Notice: Failed to log turn: {log_err}")

        return {
            "user_text": user_text,
            "assistant_text": assistant_text,
            "assistant_audio_b64": assistant_audio_b64,
            "selected_gesture": selected_gesture,
            "llm_error": llm_error,
            "policy": {
                "action": combined_policy.action.value,
                "label": combined_policy.label,
                "badge_color": combined_policy.badge_color,
                "icon": combined_policy.icon,
                "guidance": combined_policy.response_guidance,
                "clarification_prompt": combined_policy.clarification_prompt if combined_policy.action.value == "clarify" else None,
                "triggers": list(combined_policy.triggers),
                "context_warmup": combined_policy.context_warmup,
                "context_turn_count": combined_policy.context_turn_count,
            },
            "expectation_prior": expectation_prior,
            "acoustic_likelihood": acoustic,
            "visual_likelihood": visual,
            "dimension_results": dimension_results,
            "latency_ms": round(total_latency_ms, 1),
            "turn_log": turn_log,
        }
    except Exception as exc:
        print(f"ERROR in process_live_turn: {exc}")
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Turn processing error: {exc}")


@router.get("/live/logs")
def get_interaction_logs():
    """Retrieve all logged interaction turns."""
    return {"logs": get_logged_turns()}


@router.post("/live/delete-last-log")
def delete_last_interaction_log():
    """Remove the most recently recorded turn from logs."""
    removed = delete_last_turn()
    if removed is None:
        return {"status": "empty", "message": "No logged turns to remove"}
    return {
        "status": "success",
        "message": f"Turn #{removed.get('turn_idx')} removed",
        "removed_turn": removed,
    }


@router.post("/live/clear-logs")
def clear_interaction_logs():
    """Clear recorded interaction logs."""
    clear_logged_turns()
    return {"status": "success", "message": "Interaction logs cleared"}


# Include the API router into the FastAPI application
app.include_router(router)


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("predictive_processing.api.server:app", host="127.0.0.1", port=8000, reload=True)

