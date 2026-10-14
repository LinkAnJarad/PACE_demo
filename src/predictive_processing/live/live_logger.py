"""Turn-level interaction logger for live demo sessions.

Records full Bayesian state, gating metrics, response text, gesture, and end-to-end
latency (end of user speech to onset of synthesized robot speech) for every turn.
Auto-saves to CSV and JSON in analysis/outputs/ for sample interaction reporting.
"""

from __future__ import annotations

import csv
from datetime import datetime
import json
import os
from pathlib import Path
import threading
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "analysis" / "outputs"
DEFAULT_JSON_PATH = DEFAULT_OUTPUT_DIR / "live_interaction_sample.json"
DEFAULT_CSV_PATH = DEFAULT_OUTPUT_DIR / "live_interaction_sample.csv"

_lock = threading.Lock()

CSV_FIELDNAMES = [
    "turn_idx",
    "timestamp",
    "user_text",
    # Valence Bayesian State
    "val_wC",
    "val_wA",
    "val_wV",
    "val_wAV",
    "val_lik_mu",
    "val_lik_var",
    "val_prior_mu",
    "val_prior_var",
    "val_post_mu",
    "val_post_var",
    # Arousal Bayesian State
    "aro_wC",
    "aro_wA",
    "aro_wV",
    "aro_wAV",
    "aro_lik_mu",
    "aro_lik_var",
    "aro_prior_mu",
    "aro_prior_var",
    "aro_post_mu",
    "aro_post_var",
    # Gating & Policy
    "delta_val",
    "delta_aro",
    "delta_max",
    "sigma_val",
    "sigma_aro",
    "sigma_max",
    "policy_action",
    "policy_label",
    "context_warmup",
    # Response & Gesture
    "assistant_text",
    "gesture",
    # Latency
    "latency_ms",
    "stt_ms",
    "prior_ms",
    "bayes_ms",
    "llm_ms",
    "tts_ms",
]


def log_turn(
    turn_idx: int,
    user_text: str,
    assistant_text: str,
    selected_gesture: str,
    dimension_results: dict[str, Any],
    policy: Any,
    latency_ms: float,
    timing_breakdown_ms: dict[str, float] | None = None,
    output_json_path: Path = DEFAULT_JSON_PATH,
    output_csv_path: Path = DEFAULT_CSV_PATH,
) -> dict[str, Any]:
    """Record an interaction turn and auto-save to JSON and CSV."""
    if timing_breakdown_ms is None:
        timing_breakdown_ms = {}

    v_res = dimension_results.get("valence", {})
    a_res = dimension_results.get("arousal", {})

    delta_v = float(v_res.get("prior_sensory_standardized_discrepancy", 0.0))
    delta_a = float(a_res.get("prior_sensory_standardized_discrepancy", 0.0))
    sigma_v = float(v_res.get("sigma_post", 0.0))
    sigma_a = float(a_res.get("sigma_post", 0.0))

    action_val = getattr(policy, "action", policy.get("action") if isinstance(policy, dict) else "respond")
    if hasattr(action_val, "value"):
        action_str = str(action_val.value)
    else:
        action_str = str(action_val)

    label_str = getattr(policy, "label", policy.get("label") if isinstance(policy, dict) else action_str.capitalize())
    warmup_bool = bool(getattr(policy, "context_warmup", policy.get("context_warmup") if isinstance(policy, dict) else False))
    triggers_list = list(getattr(policy, "triggers", policy.get("triggers") if isinstance(policy, dict) else []))

    turn_record: dict[str, Any] = {
        "turn_idx": int(turn_idx),
        "timestamp": datetime.now().isoformat(),
        "user_text": user_text,
        "source_weights": {
            "valence": {
                "w_prior": round(float(v_res.get("w_prior", 0.0)), 3),
                "w_audio": round(float(v_res.get("w_audio", 0.0)), 3),
                "w_visual": round(float(v_res.get("w_visual", 0.0)), 3),
                "w_sensory": round(float(v_res.get("w_sensory", 0.0)), 3),
            },
            "arousal": {
                "w_prior": round(float(a_res.get("w_prior", 0.0)), 3),
                "w_audio": round(float(a_res.get("w_audio", 0.0)), 3),
                "w_visual": round(float(a_res.get("w_visual", 0.0)), 3),
                "w_sensory": round(float(a_res.get("w_sensory", 0.0)), 3),
            },
        },
        "likelihood": {
            "valence": {
                "mu_audio": round(float(v_res.get("mu_audio", 0.0)), 3),
                "var_audio": round(float(v_res.get("sigma_audio", 0.0)) ** 2, 4),
                "mu_visual": round(float(v_res.get("mu_visual", 0.0)), 3),
                "var_visual": round(float(v_res.get("sigma_visual", 0.0)) ** 2, 4),
                "mu_sensory": round(float(v_res.get("mu_sensory", 0.0)), 3),
                "var_sensory": round(float(v_res.get("sigma_sensory", 0.0)) ** 2, 4),
            },
            "arousal": {
                "mu_audio": round(float(a_res.get("mu_audio", 0.0)), 3),
                "var_audio": round(float(a_res.get("sigma_audio", 0.0)) ** 2, 4),
                "mu_visual": round(float(a_res.get("mu_visual", 0.0)), 3),
                "var_visual": round(float(a_res.get("sigma_visual", 0.0)) ** 2, 4),
                "mu_sensory": round(float(a_res.get("mu_sensory", 0.0)), 3),
                "var_sensory": round(float(a_res.get("sigma_sensory", 0.0)) ** 2, 4),
            },
        },
        "prior": {
            "valence": {
                "name": str(v_res.get("prior_name", "Dialogue Context C")),
                "mu": round(float(v_res.get("mu_prior", 0.0)), 3),
                "var": round(float(v_res.get("sigma_prior", 0.0)) ** 2, 4),
            },
            "arousal": {
                "name": str(a_res.get("prior_name", "Dialogue Context C")),
                "mu": round(float(a_res.get("mu_prior", 0.0)), 3),
                "var": round(float(a_res.get("sigma_prior", 0.0)) ** 2, 4),
            },
        },
        "posterior": {
            "valence": {
                "mu": round(float(v_res.get("mu_post", 0.0)), 3),
                "var": round(float(sigma_v) ** 2, 4),
                "sigma": round(float(sigma_v), 3),
            },
            "arousal": {
                "mu": round(float(a_res.get("mu_post", 0.0)), 3),
                "var": round(float(sigma_a) ** 2, 4),
                "sigma": round(float(sigma_a), 3),
            },
        },
        "gating_and_policy": {
            "delta_valence": round(delta_v, 3),
            "delta_arousal": round(delta_a, 3),
            "delta_max": round(max(delta_v, delta_a), 3),
            "sigma_valence": round(sigma_v, 3),
            "sigma_arousal": round(sigma_a, 3),
            "sigma_max": round(max(sigma_v, sigma_a), 3),
            "action": action_str,
            "label": label_str,
            "triggers": triggers_list,
            "context_warmup": warmup_bool,
        },
        "assistant_text": assistant_text,
        "gesture": selected_gesture,
        "latency_ms": round(float(latency_ms), 1),
        "timing_breakdown_ms": {k: round(float(v), 1) for k, v in timing_breakdown_ms.items()},
    }

    # Flatten row for CSV
    csv_row = _record_to_csv_row(turn_record)

    with _lock:
        output_json_path.parent.mkdir(parents=True, exist_ok=True)
        output_csv_path.parent.mkdir(parents=True, exist_ok=True)

        # 1. Update JSON array
        current_data: list[dict[str, Any]] = []
        if output_json_path.exists():
            try:
                with open(output_json_path, "r", encoding="utf-8") as f:
                    current_data = json.load(f)
            except Exception:
                current_data = []

        # Replace existing turn if same index, else append
        existing_idx = next((i for i, r in enumerate(current_data) if r.get("turn_idx") == turn_record["turn_idx"]), None)
        if existing_idx is not None:
            current_data[existing_idx] = turn_record
        else:
            current_data.append(turn_record)

        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)

        # 2. Append/Write CSV
        write_header = not output_csv_path.exists() or os.path.getsize(output_csv_path) == 0
        with open(output_csv_path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
            if write_header:
                writer.writeheader()
            writer.writerow(csv_row)

    return turn_record


def _record_to_csv_row(turn_record: dict[str, Any]) -> dict[str, Any]:
    """Convert nested turn record to flat CSV row."""
    return {
        "turn_idx": turn_record["turn_idx"],
        "timestamp": turn_record["timestamp"],
        "user_text": turn_record["user_text"],
        "val_wC": turn_record["source_weights"]["valence"]["w_prior"],
        "val_wA": turn_record["source_weights"]["valence"]["w_audio"],
        "val_wV": turn_record["source_weights"]["valence"]["w_visual"],
        "val_wAV": turn_record["source_weights"]["valence"]["w_sensory"],
        "val_lik_mu": turn_record["likelihood"]["valence"]["mu_sensory"],
        "val_lik_var": turn_record["likelihood"]["valence"]["var_sensory"],
        "val_prior_mu": turn_record["prior"]["valence"]["mu"],
        "val_prior_var": turn_record["prior"]["valence"]["var"],
        "val_post_mu": turn_record["posterior"]["valence"]["mu"],
        "val_post_var": turn_record["posterior"]["valence"]["var"],
        "aro_wC": turn_record["source_weights"]["arousal"]["w_prior"],
        "aro_wA": turn_record["source_weights"]["arousal"]["w_audio"],
        "aro_wV": turn_record["source_weights"]["arousal"]["w_visual"],
        "aro_wAV": turn_record["source_weights"]["arousal"]["w_sensory"],
        "aro_lik_mu": turn_record["likelihood"]["arousal"]["mu_sensory"],
        "aro_lik_var": turn_record["likelihood"]["arousal"]["var_sensory"],
        "aro_prior_mu": turn_record["prior"]["arousal"]["mu"],
        "aro_prior_var": turn_record["prior"]["arousal"]["var"],
        "aro_post_mu": turn_record["posterior"]["arousal"]["mu"],
        "aro_post_var": turn_record["posterior"]["arousal"]["var"],
        "delta_val": turn_record["gating_and_policy"]["delta_valence"],
        "delta_aro": turn_record["gating_and_policy"]["delta_arousal"],
        "delta_max": turn_record["gating_and_policy"]["delta_max"],
        "sigma_val": turn_record["gating_and_policy"]["sigma_valence"],
        "sigma_aro": turn_record["gating_and_policy"]["sigma_arousal"],
        "sigma_max": turn_record["gating_and_policy"]["sigma_max"],
        "policy_action": turn_record["gating_and_policy"]["action"],
        "policy_label": turn_record["gating_and_policy"]["label"],
        "context_warmup": turn_record["gating_and_policy"]["context_warmup"],
        "assistant_text": turn_record["assistant_text"],
        "gesture": turn_record["gesture"],
        "latency_ms": turn_record["latency_ms"],
        "stt_ms": turn_record.get("timing_breakdown_ms", {}).get("stt", 0.0),
        "prior_ms": turn_record.get("timing_breakdown_ms", {}).get("prior", 0.0),
        "bayes_ms": turn_record.get("timing_breakdown_ms", {}).get("bayes", 0.0),
        "llm_ms": turn_record.get("timing_breakdown_ms", {}).get("llm", 0.0),
        "tts_ms": turn_record.get("timing_breakdown_ms", {}).get("tts", 0.0),
    }


def delete_last_turn(
    output_json_path: Path = DEFAULT_JSON_PATH,
    output_csv_path: Path = DEFAULT_CSV_PATH,
) -> dict[str, Any] | None:
    """Delete the most recently recorded turn from JSON and CSV logs."""
    with _lock:
        if not output_json_path.exists():
            return None
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                current_data: list[dict[str, Any]] = json.load(f)
        except Exception:
            return None

        if not current_data:
            return None

        removed = current_data.pop()

        # Update JSON file
        with open(output_json_path, "w", encoding="utf-8") as f:
            json.dump(current_data, f, indent=2)

        # Update CSV file
        if not current_data:
            if output_csv_path.exists():
                output_csv_path.unlink()
        else:
            with open(output_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=CSV_FIELDNAMES)
                writer.writeheader()
                for rec in current_data:
                    writer.writerow(_record_to_csv_row(rec))

        return removed


def get_logged_turns(output_json_path: Path = DEFAULT_JSON_PATH) -> list[dict[str, Any]]:
    """Retrieve all logged turns from the JSON file."""
    with _lock:
        if not output_json_path.exists():
            return []
        try:
            with open(output_json_path, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return []


def clear_logged_turns(
    output_json_path: Path = DEFAULT_JSON_PATH,
    output_csv_path: Path = DEFAULT_CSV_PATH,
) -> None:
    """Clear both the JSON and CSV log files."""
    with _lock:
        if output_json_path.exists():
            output_json_path.unlink()
        if output_csv_path.exists():
            output_csv_path.unlink()
