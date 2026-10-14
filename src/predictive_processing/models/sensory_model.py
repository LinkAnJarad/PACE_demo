"""Acoustic Likelihood (Wav2Vec2): the "Amnesic" Listener.

Wraps the Wav2Vec2 + Stirn faithful heteroscedastic model from finetuning_stirn/
for clean inference on IEMOCAP. This model listens only to the raw audio of the
current turn — it has no memory of the conversation.

Probabilistic role: Acoustic Likelihood for the current utterance.
Output: Predicted emotion (V/A) + uncertainty (sigma).
"""

from __future__ import annotations

import json
from pathlib import Path

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from transformers import Wav2Vec2Config, Wav2Vec2FeatureExtractor, Wav2Vec2Model


class EmotionModelWithUncertainty(nn.Module):
    """Wav2Vec2 + dual-head (mean + scale) for V/A with uncertainty.

    Architecture matches finetuning_stirn/train.py:EmotionModel.
    """

    def __init__(self, config: Wav2Vec2Config):
        super().__init__()
        self.wav2vec2 = Wav2Vec2Model(config)
        hidden_state_size = config.hidden_size
        hidden_size = 368

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
        )
        self.scale_head = nn.Sequential(
            nn.Linear(hidden_state_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.SiLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_size, 2),
            nn.Softplus(),
        )
        self.tanh = nn.Tanh()

    def forward(
        self, input_values: torch.Tensor, attention_mask: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        outputs = self.wav2vec2(input_values, attention_mask=attention_mask)
        hidden_states = outputs[0]
        pooled = torch.mean(hidden_states, dim=1)

        va_means = self.mean_head(pooled)
        va_scales = self.scale_head(pooled)

        return {
            "valence_mean": self.tanh(va_means[:, 0]),
            "arousal_mean": self.tanh(va_means[:, 1]),
            "valence_sigma": va_scales[:, 0] + 1e-6,
            "arousal_sigma": va_scales[:, 1] + 1e-6,
            "embedding": pooled,  # Raw Wav2Vec2 pooled hidden state
        }


# ── Sensory Model wrapper ─────────────────────────────────────────────────


# Semantic alias for new code. The original class name remains checkpoint and
# import compatible.
AcousticLikelihoodModel = EmotionModelWithUncertainty


class SensoryModel:
    """High-level wrapper for loading and running the Acoustic Likelihood.

    Usage:
        sensory = SensoryModel.from_checkpoint(
            base_model_name="audeering/wav2vec2-large-robust-12-ft-emotion-msp-dim",
            checkpoint_path="finetuning_stirn/model_best_ccc.pt",
            device="cuda",
        )
        result = sensory.predict(waveform, attention_mask)
    """

    def __init__(
        self,
        model: EmotionModelWithUncertainty,
        processor: Wav2Vec2FeatureExtractor,
        device: str = "cpu",
    ):
        self.model = model.to(device)
        self.model.eval()
        self.processor = processor
        self.device = device

    @classmethod
    def from_checkpoint(
        cls,
        base_model_name: str,
        checkpoint_path: str | Path,
        device: str = "cpu",
    ) -> "SensoryModel":
        """Load the Acoustic Likelihood implementation from a finetuning_stirn checkpoint."""
        # Prefer already-cached Hugging Face assets. This prevents an
        # unnecessary network request on each local demo launch while still
        # falling back to a first-time download when needed.
        try:
            processor = Wav2Vec2FeatureExtractor.from_pretrained(
                base_model_name,
                local_files_only=True,
            )
            config_file = hf_hub_download(
                repo_id=base_model_name,
                filename="config.json",
                local_files_only=True,
            )
        except OSError:
            processor = Wav2Vec2FeatureExtractor.from_pretrained(base_model_name)
            config_file = hf_hub_download(repo_id=base_model_name, filename="config.json")

        # Patch Wav2Vec2Config to bypass vocab_size=None validation errors.
        with open(config_file, "r") as f:
            config_dict = json.load(f)
        config_dict["vocab_size"] = 32  # Add dummy integer to bypass Hugging Face validation check
        config = Wav2Vec2Config.from_dict(config_dict)

        # Instantiate structure and load checkpoint state dict
        model = EmotionModelWithUncertainty(config)
        checkpoint = torch.load(str(checkpoint_path), map_location=device, weights_only=False)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        model.load_state_dict(state_dict, strict=True)

        return cls(model=model, processor=processor, device=device)

    @torch.no_grad()
    def predict(
        self,
        input_values: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Run inference on a batch of audio.

        Args:
            input_values: [B, T] raw audio waveform at 16kHz
            attention_mask: [B, T] mask (1 = real audio, 0 = padding)

        Returns:
            dict with keys: valence_mean, arousal_mean, valence_sigma,
            arousal_sigma, embedding (all as tensors on CPU).
        """
        input_values = input_values.to(self.device)
        if attention_mask is not None:
            attention_mask = attention_mask.to(self.device)

        outputs = self.model(input_values, attention_mask=attention_mask)
        return {k: v.cpu() for k, v in outputs.items()}

    def process_audio(
        self,
        audio_array: list | "np.ndarray",
        sampling_rate: int = 16_000,
        max_audio_seconds: float | None = 12.0,
    ) -> dict[str, torch.Tensor]:
        """Process raw numpy audio through the full pipeline.

        Args:
            audio_array: Raw audio as numpy array or list.
            sampling_rate: Sample rate of the audio.
            max_audio_seconds: Truncate audio to this length.

        Returns:
            dict with processor outputs (input_values, attention_mask).
        """
        import numpy as np

        audio = np.asarray(audio_array, dtype=np.float32)
        if max_audio_seconds is not None:
            max_samples = int(max_audio_seconds * sampling_rate)
            audio = audio[:max_samples]

        inputs = self.processor(
            audio,
            sampling_rate=sampling_rate,
            return_tensors="pt",
            padding=False,
        )
        return inputs
