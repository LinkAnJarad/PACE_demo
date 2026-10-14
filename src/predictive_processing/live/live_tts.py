"""Live Text-to-Speech (TTS) module using Kokoro-82M.

Synthesizes LLM textual responses into playable WAV audio bytes using Kokoro.
"""

from __future__ import annotations

import io
import numpy as np
import soundfile as sf


_KOKORO_CACHE: dict = {}

def load_kokoro_pipeline(lang_code: str = "a"):
    """Load and cache the Kokoro TTS pipeline.

    Args:
        lang_code: Language code ("a" for American English, "b" for British, "j" for Japanese).

    Returns:
        Kokoro KPipeline instance.
    """
    try:
        from kokoro import KPipeline
        return KPipeline(lang_code=lang_code)
    except Exception as e:
        print(f"Could not initialize Kokoro TTS pipeline: {e}")
        return None


def synthesize_speech(
    text: str,
    voice: str = "af_heart",
    lang_code: str = "a",
    speed: float = 1.0,
    valence: float = 0.0,
    arousal: float = 0.0,
    prediction_error: float = 0.0,
    start_delay: float = 0.5,
) -> bytes | None:
    """Synthesize text into WAV audio bytes using Kokoro-82M.

    Args:
        text: Input text string to synthesize.
        voice: Voice name (e.g. "af_heart", "jf_alpha").
        lang_code: Language code ("a" for EN, "j" for JP).
        speed: Speech speed multiplier.
        valence: Perceived valence [-1.0 to 1.0].
        arousal: Perceived arousal [-1.0 to 1.0].
        prediction_error: Prediction Error magnitude [0.0 to 2.0].
        start_delay: Delay in seconds before starting movement to sync with browser audio.

    Returns:
        Raw WAV bytes or None if synthesis failed.
    """
    if not text.strip():
        return None

    pipeline = load_kokoro_pipeline(lang_code=lang_code)
    if pipeline is None:
        return None

    try:
        audio_chunks = []
        generator = pipeline(text, voice=voice, speed=speed, split_pattern=r"\n+")
        for _gs, _ps, audio in generator:
            if audio is not None and len(audio) > 0:
                audio_chunks.append(audio)

        if not audio_chunks:
            return None

        full_audio = np.concatenate(audio_chunks)
        duration = len(full_audio) / 24000.0

        # Speech micro-movements temporarily disabled for gesture testing
        # try:
        #     from predictive_processing.live_icub_client import notify_icub_talk
        #     notify_icub_talk(duration, valence=valence, arousal=arousal, prediction_error=prediction_error, start_delay=start_delay)
        # except Exception:
        #     pass

        # Kokoro default output sample rate is 24000 Hz
        buffer = io.BytesIO()
        sf.write(buffer, full_audio, samplerate=24000, format="WAV")
        buffer.seek(0)
        return buffer.getvalue()
    except Exception as e:
        print(f"TTS synthesis error: {e}")
        return None
