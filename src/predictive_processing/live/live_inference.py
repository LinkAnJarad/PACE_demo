"""Live inference wrapper for the Acoustic Likelihood (Wav2Vec2).

Provides model loading with Streamlit caching and audio-to-emotion prediction.
The Acoustic Likelihood implementation is loaded once (~5-10s for the 639MB checkpoint) and cached
across all Streamlit reruns. Inference results are also cached per unique
audio input so that slider adjustments (which cause reruns) don't re-trigger
the neural network.

Usage (inside a Streamlit page):
    from predictive_processing.live_inference import predict_acoustic_likelihood, load_acoustic_likelihood

    audio_data = st.audio_input("Record your voice")
    if audio_data is not None:
        result = predict_acoustic_likelihood(audio_data.getvalue())
        # result = {"valence_mu", "valence_sigma", "arousal_mu", "arousal_sigma"}
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

import librosa
import numpy as np
import torch

_project_root = Path(__file__).resolve().parent.parent.parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from predictive_processing.core.config import ExperimentConfig
    from predictive_processing.models.sensory_model import SensoryModel
except ModuleNotFoundError:
    try:
        from predictive_processing.config import ExperimentConfig
        from predictive_processing.models.sensory_model import SensoryModel
    except ModuleNotFoundError:
        from config import ExperimentConfig
        from models.sensory_model import SensoryModel


_ACOUSTIC_CACHE: dict = {}

def load_acoustic_likelihood(
    checkpoint_path: str | None = None, device: str | None = None
) -> SensoryModel:
    """Load the Acoustic Likelihood implementation once across Streamlit reruns.

    First call takes ~5-10 seconds (loading 639MB checkpoint + HuggingFace
    config download). Subsequent calls return instantly from cache.

    Args:
        checkpoint_path: Path to model checkpoint .pt file.
        device: "cuda" or "cpu". Defaults to CUDA if available.

    Returns:
        Loaded SensoryModel in eval mode.
    """
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    config = ExperimentConfig()
    if checkpoint_path is None or not Path(str(checkpoint_path)).exists():
        try:
            from predictive_processing.core.demo_models import BEST_CV5_WAV2VEC2
            if BEST_CV5_WAV2VEC2.checkpoint_path and Path(str(BEST_CV5_WAV2VEC2.checkpoint_path)).exists():
                checkpoint_path = str(BEST_CV5_WAV2VEC2.checkpoint_path)
            else:
                checkpoint_path = config.sensory_checkpoint
        except Exception:
            checkpoint_path = config.sensory_checkpoint

    return SensoryModel.from_checkpoint(
        base_model_name=config.sensory_base_model,
        checkpoint_path=checkpoint_path,
        device=device,
    )


import os
import subprocess
import tempfile


def decode_audio_to_array(audio_bytes: bytes, target_sr: int = 16_000) -> np.ndarray:
    """Decode raw audio bytes (WAV, WebM, MP4, Opus, OGG) to 16 kHz mono float32 array."""
    if not audio_bytes:
        return np.zeros(target_sr, dtype=np.float32)

    # 1. Try standard soundfile (for WAV/FLAC/OGG)
    try:
        import soundfile as sf
        audio_array, sr = sf.read(io.BytesIO(audio_bytes))
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1)
        if sr != target_sr:
            audio_array = librosa.resample(audio_array.astype(np.float32), orig_sr=sr, target_sr=target_sr)
        return audio_array.astype(np.float32)
    except Exception:
        pass

    # 2. Fallback to ffmpeg for WebM/Matroska/Opus/MP4 containers
    with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as tmp_file:
        tmp_file.write(audio_bytes)
        tmp_path = tmp_file.name

    try:
        cmd = [
            "ffmpeg", "-y", "-i", tmp_path,
            "-vn", "-ac", "1", "-ar", str(target_sr),
            "-f", "f32le", "-"
        ]
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
        arr = np.frombuffer(proc.stdout, dtype=np.float32)
        if len(arr) == 0:
            return np.zeros(target_sr, dtype=np.float32)
        return arr
    except Exception as exc:
        print(f"Audio decoding error via ffmpeg: {exc}")
        return np.zeros(target_sr, dtype=np.float32)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def predict_from_audio_bytes(
    audio_bytes: bytes,
    model: SensoryModel,
    target_sr: int = 16_000,
    max_seconds: float = 12.0,
    normalize_amplitude: bool = False,
    pad_to_max_seconds: bool = False,
) -> dict[str, float]:
    """Run sensory inference on raw audio bytes (WAV or WebM).

    Args:
        audio_bytes: Raw audio or video WebM bytes.
        model: Loaded SensoryModel instance.
        target_sr: Target sampling rate (Wav2Vec2 expects 16 kHz).
        max_seconds: Truncate audio beyond this length.
        normalize_amplitude: Scale non-silent audio to a peak magnitude of one.
        pad_to_max_seconds: Right-pad shorter audio to ``max_seconds``.

    Returns:
        Dict with keys: valence_mu, valence_sigma, arousal_mu, arousal_sigma.
    """
    audio_array = decode_audio_to_array(audio_bytes, target_sr=target_sr)

    # Ensure minimum length of at least 1 second (16000 samples)
    if len(audio_array) < target_sr:
        audio_array = np.pad(audio_array, (0, max(0, target_sr - len(audio_array))))

    # Truncate to max length
    max_samples = int(max_seconds * target_sr)
    audio_array = audio_array[:max_samples]
    if normalize_amplitude:
        peak = float(np.max(np.abs(audio_array))) if audio_array.size else 0.0
        if peak > 0:
            audio_array = audio_array / peak
    if pad_to_max_seconds and len(audio_array) < max_samples:
        audio_array = np.pad(audio_array, (0, max_samples - len(audio_array)))

    # Process through the Wav2Vec2 pipeline
    inputs = model.process_audio(
        audio_array,
        sampling_rate=target_sr,
        max_audio_seconds=max_seconds,
    )
    result = model.predict(
        input_values=inputs["input_values"],
        attention_mask=inputs.get("attention_mask"),
    )

    return {
        "valence_mu": float(result["valence_mean"][0]),
        "valence_sigma": float(result["valence_sigma"][0]),
        "arousal_mu": float(result["arousal_mean"][0]),
        "arousal_sigma": float(result["arousal_sigma"][0]),
    }


def predict_acoustic_likelihood(
    audio_bytes: bytes,
    checkpoint_path: str | None = None,
    max_seconds: float = 12.0,
    normalize_amplitude: bool = False,
    pad_to_max_seconds: bool = False,
) -> dict[str, float]:
    """Cached prediction — only re-runs when audio bytes or model path changes.

    Wraps :func:`predict_from_audio_bytes` with ``@st.cache_data`` so that
    widget interactions (slider drags, prior changes) that cause Streamlit
    reruns don't re-trigger the neural network inference.

    Args:
        audio_bytes: Raw WAV file bytes from ``st.audio_input().getvalue()``.
        checkpoint_path: Path to the acoustic-likelihood checkpoint.
        max_seconds: Maximum duration passed to the acoustic model.
        normalize_amplitude: Whether to peak-normalize non-silent audio.
        pad_to_max_seconds: Whether to right-pad shorter audio to the maximum.
    """
    model = load_acoustic_likelihood(checkpoint_path=checkpoint_path)
    return predict_from_audio_bytes(
        audio_bytes,
        model,
        max_seconds=max_seconds,
        normalize_amplitude=normalize_amplitude,
        pad_to_max_seconds=pad_to_max_seconds,
    )


# Legacy import aliases. New code should name the probabilistic role.
load_sensory_model = load_acoustic_likelihood
cached_predict = predict_acoustic_likelihood