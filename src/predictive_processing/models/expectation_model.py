"""Dialogue-Context Prior (GRU): the "Deaf Mind-Reader".

A GRU-based sequence model that predicts the emotion of turn t from text
embeddings of the preceding turns (t-N to t-1). It is completely deaf to the
current audio — it only reads what was *said* before.

Probabilistic role: Dialogue-Context Prior. It predicts what emotion the
speaker should exhibit next, before they speak.

Output: Predicted emotion (V/A) + uncertainty (sigma).
"""

from __future__ import annotations

import torch
import torch.nn as nn


class ExpectationModel(nn.Module):
    """GRU implementation of the Dialogue-Context Prior.

    Architecture:
        text_embeddings [B, seq_len, embed_dim]
            → GRU(hidden_dim, num_layers)
            → final hidden state [B, hidden_dim]
            → Mean Head → [V, A]  (tanh-bounded to [-1, 1])
            → Scale Head → [σ_V, σ_A]  (softplus for positivity)

    Args:
        embed_dim: Dimensionality of input text embeddings (384 for MiniLM).
        hidden_dim: GRU hidden state size.
        num_layers: Number of stacked GRU layers.
        dropout: Dropout rate between GRU layers and in heads.
    """

    def __init__(
        self,
        embed_dim: int = 384,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.gru = nn.GRU(
            input_size=embed_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0.0,
            batch_first=True,
        )

        self.mean_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [Valence, Arousal]
        )

        self.scale_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),  # [σ_V, σ_A]
            nn.Softplus(),
        )

        self.tanh = nn.Tanh()

    def forward(
        self,
        context_embeddings: torch.Tensor,
        context_lengths: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            context_embeddings: [B, max_seq_len, embed_dim] padded text embeddings
                of the preceding turns.
            context_lengths: [B] actual lengths of each sequence (for packing).
                If None, assumes all sequences are full length.

        Returns:
            dict with keys:
                valence_mean, arousal_mean: [B] predictions in [-1, 1]
                valence_sigma, arousal_sigma: [B] uncertainty (std dev)
        """
        if context_lengths is not None:
            # Pack padded sequences for efficient GRU processing
            packed = nn.utils.rnn.pack_padded_sequence(
                context_embeddings,
                context_lengths.cpu().clamp(min=1),
                batch_first=True,
                enforce_sorted=False,
            )
            _, h_n = self.gru(packed)
        else:
            _, h_n = self.gru(context_embeddings)

        # h_n: [num_layers, B, hidden_dim] → take last layer
        final_hidden = h_n[-1]  # [B, hidden_dim]

        va_means = self.mean_head(final_hidden)
        va_scales = self.scale_head(final_hidden)

        return {
            "valence_mean": self.tanh(va_means[:, 0]),
            "arousal_mean": self.tanh(va_means[:, 1]),
            "valence_sigma": va_scales[:, 0] + 1e-6,
            "arousal_sigma": va_scales[:, 1] + 1e-6,
        }


# Semantic alias for new code. ``ExpectationModel`` remains the checkpoint-
# compatible implementation name.
DialogueContextPriorModel = ExpectationModel


if __name__ == "__main__":
    # Quick shape test
    model = ExpectationModel(embed_dim=384, hidden_dim=128, num_layers=2)
    x = torch.randn(8, 5, 384)  # batch=8, context=5 turns, MiniLM dim
    lengths = torch.tensor([5, 4, 3, 2, 1, 5, 3, 2])
    out = model(x, lengths)
    print("Valence mean shape:", out["valence_mean"].shape)
    print("Arousal sigma shape:", out["arousal_sigma"].shape)
    print("Sample output:", {k: v[:3].tolist() for k, v in out.items()})
