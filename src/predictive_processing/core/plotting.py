"""Shared plotting utilities for the Predictive Processing demo.

Provides the Gaussian overlay plot used by both the IEMOCAP browser
and the live recording mode. Extracted from app.py for reuse.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

try:
    from predictive_processing.core.bayesian_engine import gaussian_pdf
except ModuleNotFoundError:
    from predictive_processing.bayesian_engine import gaussian_pdf


# ── Color palette (muted academic/scientific publication colors) ───────────
COLOR_PRIOR = "#2B5C8F"       # Muted Navy Blue — expectation/prior
COLOR_LIKELIHOOD = "#B83A2B"  # Muted Crimson Red — sensory/likelihood
COLOR_POSTERIOR = "#683D8C"   # Muted Deep Violet — posterior
COLOR_HUMAN = "#2A7B4C"       # Muted Forest Green — human consensus

# ── Emotion value axis range (model outputs are in [-1, 1]) ───────────────
X_MIN = -1.5
X_MAX = 1.5
X_POINTS = 500


def plot_gaussian_update(
    mu_prior: float,
    sigma_prior: float,
    mu_likelihood: float,
    sigma_likelihood: float,
    mu_post: float,
    sigma_post: float,
    human_consensus: float | None,
    dimension_label: str,
    prior_label: str,
) -> plt.Figure:
    """Draw the three Gaussian bell curves + optional human consensus marker.

    Args:
        mu_prior: Prior mean.
        sigma_prior: Prior standard deviation (after user slider adjustment).
        mu_likelihood: Likelihood mean.
        sigma_likelihood: Likelihood standard deviation.
        mu_post: Posterior mean.
        sigma_post: Posterior standard deviation.
        human_consensus: Ground-truth human average, or None to omit marker.
        dimension_label: "Valence" or "Arousal".
        prior_label: Canonical label for the selected prior source.

    Returns:
        Matplotlib Figure object.
    """
    x = np.linspace(X_MIN, X_MAX, X_POINTS)

    pdf_prior = gaussian_pdf(x, mu_prior, sigma_prior)
    pdf_likelihood = gaussian_pdf(x, mu_likelihood, sigma_likelihood)
    pdf_post = gaussian_pdf(x, mu_post, sigma_post)

    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=120)
    fig.patch.set_facecolor("#FFFFFF")
    ax.set_facecolor("#FFFFFF")

    # Fill under curves with minimal transparency
    ax.fill_between(x, pdf_prior, alpha=0.08, color=COLOR_PRIOR)
    ax.fill_between(x, pdf_likelihood, alpha=0.08, color=COLOR_LIKELIHOOD)
    ax.fill_between(x, pdf_post, alpha=0.12, color=COLOR_POSTERIOR)

    # Draw curves
    ax.plot(x, pdf_prior, color=COLOR_PRIOR, linewidth=2.0,
            label=f"Prior ({prior_label}): μ={mu_prior:+.3f}, σ={sigma_prior:.3f}")
    ax.plot(x, pdf_likelihood, color=COLOR_LIKELIHOOD, linewidth=2.0,
            label=f"Acoustic Likelihood (Wav2Vec2): μ={mu_likelihood:+.3f}, σ={sigma_likelihood:.3f}")
    ax.plot(x, pdf_post, color=COLOR_POSTERIOR, linewidth=2.5, linestyle="--",
            label=f"Precision-Weighted Posterior: μ={mu_post:+.3f}, σ={sigma_post:.3f}")

    # Human consensus marker (only when ground truth is available)
    y_max = max(pdf_prior.max(), pdf_likelihood.max(), pdf_post.max())
    if human_consensus is not None:
        ax.axvline(
            human_consensus, color=COLOR_HUMAN, linewidth=1.5, linestyle=":",
            label=f"Human Consensus: {human_consensus:+.3f}",
        )
        ax.plot(human_consensus, 0, marker="^", color=COLOR_HUMAN, markersize=8,
                zorder=5, clip_on=False)

    # Academic styling
    ax.set_xlabel(f"{dimension_label} coordinate", fontsize=11, color="#333333", labelpad=8)
    ax.set_ylabel("Probability Density", fontsize=11, color="#333333", labelpad=8)
    ax.set_title(
        f"Bayesian Integration Analysis — {dimension_label}",
        fontsize=12, fontweight="bold", color="#111111", pad=12,
    )

    ax.legend(
        loc="upper right", fontsize=8.5, framealpha=0.95,
        facecolor="#F8F9FA", edgecolor="#DDDDDD", labelcolor="#333333",
    )

    ax.set_xlim(X_MIN, X_MAX)
    ax.set_ylim(bottom=0, top=y_max * 1.1)
    ax.tick_params(colors="#333333", labelsize=9)
    ax.xaxis.set_major_locator(ticker.MultipleLocator(0.5))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(0.1))

    # Hide top/right spines
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#CCCCCC")
    ax.spines["bottom"].set_color("#CCCCCC")

    ax.grid(axis="x", alpha=0.3, color="#DDDDDD", linestyle="--")

    fig.tight_layout()
    return fig
