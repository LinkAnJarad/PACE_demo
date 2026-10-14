"""Closed-form Gaussian–Gaussian Bayesian update engine.

Implements the precision-weighted conjugate update from PLAN2.md:

    Prior:      P(θ)   = N(μ_p, σ_p²)
    Likelihood: P(x|θ) = N(μ_l, σ_l²)
    Posterior:  P(θ|x) = N(μ_post, σ_post²)

Where:
    τ = 1/σ²              (precision)
    τ_post = τ_p + τ_l
    μ_post = (τ_p·μ_p + τ_l·μ_l) / τ_post
    W_p = τ_p / τ_post    (prior weight)
    W_l = τ_l / τ_post    (likelihood weight)

No ML framework dependencies — pure numpy/math.
"""

from __future__ import annotations

import math

import numpy as np


# ── Core math ──────────────────────────────────────────────────────────────


def precision(sigma: float) -> float:
    """Convert standard deviation to precision (τ = 1/σ²).

    Args:
        sigma: Standard deviation (must be > 0).

    Returns:
        Precision value.
    """
    if sigma <= 0:
        raise ValueError(f"sigma must be positive, got {sigma}")
    return 1.0 / (sigma * sigma)


def sigma_from_precision(tau: float) -> float:
    """Convert precision back to standard deviation (σ = 1/√τ).

    Args:
        tau: Precision (must be > 0).

    Returns:
        Standard deviation.
    """
    if tau <= 0:
        raise ValueError(f"tau must be positive, got {tau}")
    return 1.0 / math.sqrt(tau)


def posterior_params(
    mu_prior: float,
    sigma_prior: float,
    mu_likelihood: float,
    sigma_likelihood: float,
) -> dict[str, float]:
    """Compute the closed-form Gaussian–Gaussian Bayesian update.

    Args:
        mu_prior: Prior mean.
        sigma_prior: Prior standard deviation.
        mu_likelihood: Likelihood mean.
        sigma_likelihood: Likelihood standard deviation.

    Returns:
        Dict with keys:
            mu_post: Posterior mean.
            sigma_post: Posterior standard deviation.
            tau_prior: Prior precision.
            tau_likelihood: Likelihood precision.
            tau_post: Posterior precision.
            w_prior: Weight of prior (τ_p / τ_post).
            w_likelihood: Weight of likelihood (τ_l / τ_post).
            prediction_error: |μ_l - μ_p|.
    """
    tau_p = precision(sigma_prior)
    tau_l = precision(sigma_likelihood)
    tau_post = tau_p + tau_l

    mu_post = (tau_p * mu_prior + tau_l * mu_likelihood) / tau_post
    sigma_post = sigma_from_precision(tau_post)

    w_prior = tau_p / tau_post
    w_likelihood = tau_l / tau_post
    prediction_error = abs(mu_likelihood - mu_prior)

    return {
        "mu_post": mu_post,
        "sigma_post": sigma_post,
        "tau_prior": tau_p,
        "tau_likelihood": tau_l,
        "tau_post": tau_post,
        "w_prior": w_prior,
        "w_likelihood": w_likelihood,
        "prediction_error": prediction_error,
    }


def sensory_fusion_params(
    mu_audio: float,
    sigma_audio: float,
    mu_visual: float,
    sigma_visual: float,
) -> dict[str, float]:
    """Compute Level-1 multimodal sensory integration (AV = Audio + Visual).

    Args:
        mu_audio: Acoustic likelihood mean.
        sigma_audio: Acoustic likelihood standard deviation.
        mu_visual: Visual likelihood mean.
        sigma_visual: Visual likelihood standard deviation.

    Returns:
        Dict with keys:
            mu_sensory: Combined sensory mean.
            sigma_sensory: Combined sensory standard deviation.
            tau_audio: Acoustic precision.
            tau_visual: Visual precision.
            tau_sensory: Combined sensory precision (tau_A + tau_V).
            w_audio_sensory: Acoustic weight within sensory fusion (tau_A / tau_AV).
            w_visual_sensory: Visual weight within sensory fusion (tau_V / tau_AV).
            cross_sensory_discrepancy: |mu_A - mu_V|.
            cross_sensory_standardized_discrepancy: |mu_A - mu_V| / sqrt(sigma_A^2 + sigma_V^2).
    """
    tau_a = precision(sigma_audio)
    tau_v = precision(sigma_visual)
    tau_av = tau_a + tau_v

    mu_av = (tau_a * mu_audio + tau_v * mu_visual) / tau_av
    sigma_av = sigma_from_precision(tau_av)

    w_a_sens = tau_a / tau_av
    w_v_sens = tau_v / tau_av

    raw_disc = abs(mu_audio - mu_visual)
    joint_std = math.sqrt(sigma_audio * sigma_audio + sigma_visual * sigma_visual)
    stand_disc = raw_disc / joint_std if joint_std > 0 else 0.0

    return {
        "mu_sensory": mu_av,
        "sigma_sensory": sigma_av,
        "tau_audio": tau_a,
        "tau_visual": tau_v,
        "tau_sensory": tau_av,
        "w_audio_sensory": w_a_sens,
        "w_visual_sensory": w_v_sens,
        "cross_sensory_discrepancy": raw_disc,
        "cross_sensory_standardized_discrepancy": stand_disc,
    }


def hierarchical_posterior_params(
    mu_prior: float,
    sigma_prior: float,
    mu_audio: float,
    sigma_audio: float,
    mu_visual: float,
    sigma_visual: float,
) -> dict[str, float]:
    """Compute the two-level hierarchical Bayesian posterior PWP(C, AV).

    Level 1: tau_AV = tau_A + tau_V, mu_AV = (tau_A mu_A + tau_V mu_V) / tau_AV
    Level 2: tau_post = tau_p + tau_AV = tau_p + tau_A + tau_V
             mu_post = (tau_p mu_p + tau_A mu_A + tau_V mu_V) / tau_post

    Args:
        mu_prior: Prior mean (Context C or Speaker History H).
        sigma_prior: Prior standard deviation.
        mu_audio: Acoustic likelihood mean (A).
        sigma_audio: Acoustic likelihood standard deviation.
        mu_visual: Visual likelihood mean (V).
        sigma_visual: Visual likelihood standard deviation.

    Returns:
        Dict with keys:
            mu_post: Posterior mean.
            sigma_post: Posterior standard deviation.
            tau_prior: Prior precision.
            tau_audio: Audio precision.
            tau_visual: Visual precision.
            tau_sensory: Combined sensory precision.
            tau_post: Posterior precision.
            w_prior: Prior source weight (tau_p / tau_post).
            w_audio: Acoustic source weight (tau_A / tau_post).
            w_visual: Visual source weight (tau_V / tau_post).
            w_sensory: Total sensory weight (w_audio + w_visual).
            w_audio_sensory: Audio share of sensory (tau_A / tau_AV).
            w_visual_sensory: Visual share of sensory (tau_V / tau_AV).
            mu_sensory: Level-1 sensory mean.
            sigma_sensory: Level-1 sensory standard deviation.
            prior_sensory_discrepancy: |mu_AV - mu_p|.
            prior_sensory_standardized_discrepancy: |mu_AV - mu_p| / sqrt(sigma_p^2 + sigma_AV^2).
            cross_sensory_discrepancy: |mu_A - mu_V|.
            cross_sensory_standardized_discrepancy: |mu_A - mu_V| / sqrt(sigma_A^2 + sigma_V^2).
    """
    sensory = sensory_fusion_params(
        mu_audio=mu_audio,
        sigma_audio=sigma_audio,
        mu_visual=mu_visual,
        sigma_visual=sigma_visual,
    )

    tau_p = precision(sigma_prior)
    tau_a = sensory["tau_audio"]
    tau_v = sensory["tau_visual"]
    tau_av = sensory["tau_sensory"]
    tau_post = tau_p + tau_av

    mu_post = (tau_p * mu_prior + tau_a * mu_audio + tau_v * mu_visual) / tau_post
    sigma_post = sigma_from_precision(tau_post)

    w_p = tau_p / tau_post
    w_a = tau_a / tau_post
    w_v = tau_v / tau_post
    w_av = tau_av / tau_post

    mu_av = sensory["mu_sensory"]
    sigma_av = sensory["sigma_sensory"]

    raw_disc_prior_sensory = abs(mu_av - mu_prior)
    joint_std_prior_sensory = math.sqrt(sigma_prior * sigma_prior + sigma_av * sigma_av)
    stand_disc_prior_sensory = raw_disc_prior_sensory / joint_std_prior_sensory if joint_std_prior_sensory > 0 else 0.0

    return {
        "mu_post": mu_post,
        "sigma_post": sigma_post,
        "tau_prior": tau_p,
        "tau_audio": tau_a,
        "tau_visual": tau_v,
        "tau_sensory": tau_av,
        "tau_post": tau_post,
        "w_prior": w_p,
        "w_audio": w_a,
        "w_visual": w_v,
        "w_sensory": w_av,
        "w_audio_sensory": sensory["w_audio_sensory"],
        "w_visual_sensory": sensory["w_visual_sensory"],
        "mu_sensory": mu_av,
        "sigma_sensory": sigma_av,
        "prior_sensory_discrepancy": raw_disc_prior_sensory,
        "prior_sensory_standardized_discrepancy": stand_disc_prior_sensory,
        "cross_sensory_discrepancy": sensory["cross_sensory_discrepancy"],
        "cross_sensory_standardized_discrepancy": sensory["cross_sensory_standardized_discrepancy"],
    }


def gaussian_pdf(x: np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Evaluate the Gaussian PDF at points x.

    Args:
        x: Array of points to evaluate.
        mu: Mean of the Gaussian.
        sigma: Standard deviation of the Gaussian.

    Returns:
        PDF values at each point in x.
    """
    coeff = 1.0 / (sigma * math.sqrt(2.0 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * np.exp(exponent)


# ── Smoke test ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Test 1: equal precision unimodal
    result = posterior_params(
        mu_prior=-0.5, sigma_prior=0.3,
        mu_likelihood=0.5, sigma_likelihood=0.3,
    )
    print("Equal precision test:")
    for k, v in result.items():
        print(f"  {k}: {v:.6f}")
    assert abs(result["mu_post"] - 0.0) < 1e-10, "Midpoint test failed"
    assert abs(result["w_prior"] - 0.5) < 1e-10, "Weight symmetry test failed"
    assert abs(result["w_prior"] + result["w_likelihood"] - 1.0) < 1e-10

    # Test 2: hierarchical multimodal fusion
    h_res = hierarchical_posterior_params(
        mu_prior=0.2, sigma_prior=0.4,
        mu_audio=-0.3, sigma_audio=0.3,
        mu_visual=0.1, sigma_visual=0.5,
    )
    print("\nHierarchical Multimodal Test:")
    for k, v in h_res.items():
        print(f"  {k}: {v:.6f}")
    assert abs(h_res["w_prior"] + h_res["w_audio"] + h_res["w_visual"] - 1.0) < 1e-10
    assert abs(h_res["w_sensory"] - (h_res["w_audio"] + h_res["w_visual"])) < 1e-10

    print("\nAll tests passed!")
