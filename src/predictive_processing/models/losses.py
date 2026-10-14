"""Shared loss functions and metrics for the Predictive Processing experiment.

Adapted from finetuning_stirn/losses.py (Stirn et al. 2023 faithful loss).
"""

from __future__ import annotations

import numpy as np
import torch


def stirn_faithful_loss(
    mu: torch.Tensor,
    y: torch.Tensor,
    sigma: torch.Tensor,
    eps: float = 1e-6,
) -> torch.Tensor:
    """Stirn et al. (2023) faithful heteroscedastic loss.

    Decomposes the Gaussian NLL into two terms with separate
    gradient responsibilities:

      Term 1 | unit-variance NLL → trains mu (mean head + trunk)
      Term 2 | heteroscedastic NLL → trains sigma head only
              | mu is detached so sigma must explain residual variance

    Args:
        mu:    (batch, dim_out) predicted mean
        y:     (batch, dim_out) ground truth
        sigma: (batch, dim_out) predicted std (must be > 0)
        eps:   numerical floor for scale
    """
    sigma = sigma.clamp(min=eps)

    # Term 1: unit-variance NLL — pure MSE signal for the mean
    nll_mu = 0.5 * (y - mu) ** 2

    # Term 2: heteroscedastic NLL — sigma trains against a frozen mean
    nll_sigma = torch.log(sigma) + 0.5 * (y - mu.detach()) ** 2 / sigma**2

    return (nll_mu + nll_sigma).mean()


def calculate_ccc(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """Concordance Correlation Coefficient (CCC)."""
    if len(y_true) < 2:
        return 0.0
    cor = np.corrcoef(y_true, y_pred)[0, 1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    sd_true = np.std(y_true)
    sd_pred = np.std(y_pred)

    numerator = 2 * cor * sd_true * sd_pred
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return float(numerator / (denominator + 1e-8))
