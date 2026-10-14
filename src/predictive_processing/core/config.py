"""Central configuration for the Predictive Processing experiment."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()


@dataclass
class ExperimentConfig:
    """All hyperparameters and paths for the experiment."""

    # ── Paths ──────────────────────────────────────────────────────────────
    iemocap_root: str = "/home/lana/ICMI/IEMOCAP"
    sensory_checkpoint: str = "/home/lana/ICMI/predictive_processing/outputs/sensory_model.pt"
    sensory_base_model: str = "audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim"
    text_encoder_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    output_dir: str = "/home/lana/ICMI/predictive_processing/outputs"

    # ── Data ───────────────────────────────────────────────────────────────
    sampling_rate: int = 16_000
    max_audio_seconds: float = 12.0
    context_window: int = 3  # Default; configurable for ablation
    max_context_window: int = 15  # Maximum sliding window size
    only_improvised: bool = True  # Filter to improvised dialogs
    min_raters_test3: int = 3  # Minimum raters for Test 3 (disagreement)
    seed: int = 42

    # ── Splits ─────────────────────────────────────────────────────────────
    train_sessions: list[str] = field(
        default_factory=lambda: ["Session1", "Session2", "Session3", "Session4"]
    )
    val_sessions: list[str] = field(default_factory=lambda: ["Session5"])

    # ── Expectation Model ──────────────────────────────────────────────────
    text_embed_dim: int = 384  # all-MiniLM-L6-v2 output dimension
    expectation_hidden_dim: int = 128
    expectation_num_layers: int = 2
    expectation_dropout: float = 0.1
    expectation_lr: float = 1e-3
    expectation_weight_decay: float = 1e-4
    expectation_epochs: int = 50
    expectation_batch_size: int = 32
    expectation_patience: int = 10  # Early stopping patience

    # ── Acoustic Likelihood (Wav2Vec2) ─────────────────────────────────────
    sensory_batch_size: int = 32

    # ── Live Interactive Pipeline (STT / LLM / TTS) ────────────────────────
    openai_api_base: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_BASE", "http://localhost:11434/v1")
    )
    openai_api_key: str = field(
        default_factory=lambda: os.getenv("OPENAI_API_KEY", "sk-no-key-required")
    )
    openai_model_name: str = field(
        default_factory=lambda: os.getenv("OPENAI_MODEL_NAME", "gpt-4o-mini")
    )
    whisper_model_size: str = field(
        default_factory=lambda: os.getenv("WHISPER_MODEL_SIZE", "small")
    )
    kokoro_voice_en: str = field(
        default_factory=lambda: os.getenv("KOKORO_VOICE_EN", "af_heart")
    )
    kokoro_voice_jp: str = field(
        default_factory=lambda: os.getenv("KOKORO_VOICE_JP", "jf_alpha")
    )

    @property
    def iemocap_path(self) -> Path:
        return Path(self.iemocap_root)

    @property
    def output_path(self) -> Path:
        p = Path(self.output_dir)
        if not p.exists() or not (p / "expectation_model_cw3_best.pt").exists():
            local_p = Path(__file__).resolve().parent.parent.parent.parent / "outputs"
            if local_p.exists():
                return local_p
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def cached_embeddings_dir(self) -> Path:
        p = self.output_path / "cached_embeddings"
        p.mkdir(parents=True, exist_ok=True)
        return p

    @property
    def figures_dir(self) -> Path:
        p = self.output_path / "figures"
        p.mkdir(parents=True, exist_ok=True)
        return p
