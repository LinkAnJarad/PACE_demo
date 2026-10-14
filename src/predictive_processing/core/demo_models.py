"""Temporary CV5 checkpoint selection for the interactive English demo.

The checkpoints below were chosen by the highest mean held-out CCC across
valence and arousal among the five LOSO folds. They are deliberately kept in
one place so that replacing them with final all-session deployment models is a
small, explicit change.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
CV5_CHECKPOINT_DIR = PROJECT_ROOT / "checkpoints"


@dataclass(frozen=True)
class DemoModelSelection:
    """Metadata for a checkpoint intentionally used by the live demo."""

    component: str
    checkpoint_path: Path
    fold: int
    valence_ccc: float
    arousal_ccc: float
    context_window: int | None = None

    @property
    def mean_ccc(self) -> float:
        """Mean of the per-dimension held-out CCC values."""
        return (self.valence_ccc + self.arousal_ccc) / 2

    @property
    def label(self) -> str:
        """Short, user-facing description of the temporary selection."""
        return f"CV5 fold {self.fold} (held-out mean CCC {self.mean_ccc:.3f})"


# Per-fold CCC was calculated from the saved held-out prediction exports.
# These are temporary demo choices, not final deployment checkpoints.
BEST_CV5_GRU = DemoModelSelection(
    component="Dialogue-context GRU",
    checkpoint_path=CV5_CHECKPOINT_DIR / "gru_fold_2_best.pt",
    fold=2,
    valence_ccc=0.6788,
    arousal_ccc=0.1612,
    context_window=10,
)

BEST_CV5_WAV2VEC2 = DemoModelSelection(
    component="Acoustic likelihood Wav2Vec2",
    checkpoint_path=CV5_CHECKPOINT_DIR / "wav2vec2_fold_1_best.pt",
    fold=1,
    valence_ccc=0.6645,
    arousal_ccc=0.7079,
)
