"""Live Speech-to-Text (STT) module using faster-whisper.

Provides cached Whisper model loading and speech transcription for live audio input.
"""

import os
import sys
import tempfile
from pathlib import Path


# Automatically register Windows NVIDIA CUDA DLL paths for ctranslate2 / faster-whisper
if sys.platform == "win32":
    import site
    extra_dll_dirs = []
    for site_p in site.getsitepackages():
        for base in [Path(site_p) / "nvidia", Path(site_p) / "torch"]:
            if base.exists():
                extra_dll_dirs.extend([p for p in list(base.glob("*/bin")) + list(base.glob("*/lib")) + [base / "lib"] if p.is_dir()])
    
    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_bin = Path(cuda_path) / "bin"
        if cuda_bin.is_dir():
            extra_dll_dirs.append(cuda_bin)

    for p in extra_dll_dirs:
        try:
            os.add_dll_directory(str(p))
        except Exception:
            pass
        os.environ["PATH"] = str(p) + os.pathsep + os.environ.get("PATH", "")

# Compatibility patch for faster-whisper with PyAV >= 19.0.0 (where metadata_errors was removed)
try:
    import av
    _orig_av_open = av.open

    def _safe_av_open(*args, **kwargs):
        try:
            return _orig_av_open(*args, **kwargs)
        except TypeError as err:
            if "metadata_errors" in str(err) or "metadata_errors" in kwargs:
                kwargs.pop("metadata_errors", None)
                return _orig_av_open(*args, **kwargs)
            raise

    av.open = _safe_av_open
except Exception:
    pass

from faster_whisper import WhisperModel


_STT_CACHE: dict = {}

def load_whisper_model(model_size: str = "small", device: str = "cpu") -> WhisperModel:
    """Load and cache the faster-whisper model.

    Args:
        model_size: Model size name ("tiny", "base", "small", "medium", "large-v3").
        device: Device to use ("cuda", "cpu", or "auto").

    Returns:
        Loaded WhisperModel instance.
    """
    compute_type = "float16" if device == "cuda" else "int8"
    try:
        return WhisperModel(model_size, device=device, compute_type=compute_type)
    except Exception:
        return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_audio_bytes(
    audio_bytes: bytes,
    model_size: str = "small",
    device: str = "auto",
    language: str | None = None,
) -> str:
    """Transcribe raw audio bytes (WAV or WebM) using faster-whisper.

    Args:
        audio_bytes: Raw audio or video WebM bytes.
        model_size: Whisper model size.
        device: "auto", "cuda", or "cpu".
        language: Language code ("en", "ja", or None for auto-detection).

    Returns:
        Transcribed text string.
    """
    if not audio_bytes:
        return ""

    model = load_whisper_model(model_size=model_size, device=device)

    # Convert audio bytes to standard 16kHz WAV file using ffmpeg for reliable decoding
    with tempfile.NamedTemporaryFile(suffix=".input", delete=False) as raw_file:
        raw_file.write(audio_bytes)
        raw_path = raw_file.name

    wav_path = raw_path + ".wav"
    try:
        import subprocess
        # Extract and convert audio stream to 16kHz mono WAV
        cmd = [
            "ffmpeg", "-y", "-i", raw_path,
            "-vn", "-ac", "1", "-ar", "16000",
            wav_path
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False)
        target_audio_file = wav_path if os.path.exists(wav_path) and os.path.getsize(wav_path) > 0 else raw_path

        try:
            segments, _info = model.transcribe(
                target_audio_file,
                language=language,
                beam_size=5,
                vad_filter=True,
            )
            text = " ".join([segment.text.strip() for segment in segments]).strip()
            return text
        except RuntimeError as cuda_err:
            # If CUDA / cuBLAS DLL missing (e.g. cublas64_12.dll), fallback to CPU
            if "cublas" in str(cuda_err).lower() or "cuda" in str(cuda_err).lower() or device != "cpu":
                print(f"Whisper CUDA error ({cuda_err}), falling back to CPU...")
                cpu_model = load_whisper_model(model_size=model_size, device="cpu")
                segments, _info = cpu_model.transcribe(
                    target_audio_file,
                    language=language,
                    beam_size=5,
                    vad_filter=True,
                )
                text = " ".join([segment.text.strip() for segment in segments]).strip()
                return text
            raise cuda_err
    finally:
        if os.path.exists(raw_path):
            os.remove(raw_path)
        if os.path.exists(wav_path):
            os.remove(wav_path)
