"""Live Dialogue-Context Prior (GRU) wrapper ("The Deaf Mind-Reader").

Encodes preceding dialogue text transcript turns using sentence-transformers/all-MiniLM-L6-v2
and evaluates the trained GRU implementation to predict the Dialogue-Context Prior.
"""

from __future__ import annotations

from dataclasses import dataclass
import sys
from pathlib import Path
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

# Ensure project root and src directory are in sys.path
_project_root = Path(__file__).resolve().parent.parent.parent.parent
_src_dir = _project_root / "src"
if str(_src_dir) not in sys.path:
    sys.path.insert(0, str(_src_dir))
if str(_project_root) not in sys.path:
    sys.path.insert(0, str(_project_root))

try:
    from predictive_processing.core.demo_models import BEST_CV5_GRU
    from predictive_processing.models.expectation_model import ExpectationModel
except ModuleNotFoundError:
    try:
        from predictive_processing.demo_models import BEST_CV5_GRU
        from predictive_processing.models.expectation_model import ExpectationModel
    except ModuleNotFoundError:
        from demo_models import BEST_CV5_GRU
        from models.expectation_model import ExpectationModel


_TEXT_ENCODER_CACHE: dict = {}
_DIALOGUE_PRIOR_CACHE: dict = {}

def load_text_encoder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Load and cache HuggingFace text encoder tokenizer and model."""
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    return tokenizer, model


def encode_text(text: str, tokenizer, model) -> np.ndarray:
    """Encode a single sentence into a 384-d vector via mean pooling."""
    if not text.strip():
        return np.zeros(384, dtype=np.float32)
    inputs = tokenizer(
        text, padding=True, truncation=True, max_length=128, return_tensors="pt"
    )
    with torch.no_grad():
        outputs = model(**inputs)
        mask = inputs["attention_mask"].unsqueeze(-1).expand(outputs.last_hidden_state.size()).float()
        sum_embeddings = torch.sum(outputs.last_hidden_state * mask, 1)
        sum_mask = torch.clamp(mask.sum(1), min=1e-9)
        mean_pooled = (sum_embeddings / sum_mask).squeeze(0).cpu().numpy()
    return mean_pooled


@dataclass(frozen=True)
class LoadedDialogueContextPrior:
    """A GRU plus the sequence semantics used during its training."""

    model: ExpectationModel
    uses_packed_context: bool


def _load_dialogue_context_prior_runtime(
    checkpoint_path: str | None = None,
) -> LoadedDialogueContextPrior | None:
    """Load a legacy or CV5 dialogue-context checkpoint with its input semantics."""
    path = Path(checkpoint_path) if checkpoint_path else BEST_CV5_GRU.checkpoint_path
    if not path.exists():
        return None

    try:
        ckpt = torch.load(path, map_location="cpu")
        uses_packed_context = "model_state_dict" in ckpt
        ckpt_config = ckpt.get("config", {}) if uses_packed_context else {}
        state_dict = ckpt["model_state_dict"] if uses_packed_context else ckpt
        model = ExpectationModel(
            embed_dim=ckpt_config.get("embed_dim", 384),
            hidden_dim=ckpt_config.get("hidden_dim", 128),
            num_layers=ckpt_config.get("num_layers", 2),
            dropout=ckpt_config.get("dropout", 0.1),
        )
        model.load_state_dict(state_dict)
        model.eval()
        return LoadedDialogueContextPrior(
            model=model,
            uses_packed_context=uses_packed_context,
        )
    except Exception as e:
        print(f"Error loading expectation model checkpoint: {e}")
        return None


def load_dialogue_context_prior_model(
    checkpoint_path: str | None = None,
) -> ExpectationModel | None:
    """Load the GRU implementation of the Dialogue-Context Prior."""
    runtime = _load_dialogue_context_prior_runtime(checkpoint_path)
    return runtime.model if runtime else None


def predict_dialogue_context_prior(
    text_history: list[str],
    context_window: int = BEST_CV5_GRU.context_window or 10,
    checkpoint_path: str | None = None,
) -> dict[str, float] | None:
    """Predict Dialogue-Context Prior parameters from preceding transcripts.

    Args:
        text_history: List of preceding transcript turn text strings.
        context_window: Context sliding window size.
        checkpoint_path: Optional GRU checkpoint. Defaults to the selected
            CV5 demo checkpoint.

    Returns:
        Dict with keys valence_mu, valence_sigma, arousal_mu, arousal_sigma,
        or None if no history/model unavailable.
    """
    if not text_history:
        return None

    runtime = _load_dialogue_context_prior_runtime(checkpoint_path)
    if runtime is None:
        return None

    tokenizer, encoder = load_text_encoder()

    # Slice preceding turns up to context_window
    recent_texts = text_history[-context_window:]
    context_embs = [encode_text(t, tokenizer, encoder) for t in recent_texts]
    actual_len = len(context_embs)

    # Left pad to context_window
    while len(context_embs) < context_window:
        context_embs.insert(0, np.zeros(384, dtype=np.float32))

    emb_tensor = torch.tensor(np.stack(context_embs), dtype=torch.float32).unsqueeze(0)
    len_tensor = torch.tensor([actual_len], dtype=torch.long)

    with torch.no_grad():
        # CV5 was trained on fixed, left-padded windows; its full padded
        # sequence must be preserved. Legacy checkpoints used packed inputs.
        out = runtime.model(
            emb_tensor,
            len_tensor if runtime.uses_packed_context else None,
        )

    return {
        "valence_mu": float(out["valence_mean"][0].item()),
        "valence_sigma": float(out["valence_sigma"][0].item()),
        "arousal_mu": float(out["arousal_mean"][0].item()),
        "arousal_sigma": float(out["arousal_sigma"][0].item()),
    }


# Backward-compatible import aliases. New code should use the semantic names.
load_expectation_model = load_dialogue_context_prior_model
predict_expectation_prior = predict_dialogue_context_prior