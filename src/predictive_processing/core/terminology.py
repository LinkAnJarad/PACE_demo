"""Canonical terminology and schema compatibility for predictive processing.

This module is the single source of truth for names that appear at the
boundaries of the project: interfaces, figures, evaluation reports, and new
analysis exports.  The existing ``gru_*``, ``w2v_*``, and related field names
remain supported as the version-1 research schema so historic experiments and
checkpoints remain reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "2.0"

POPULATION_PRIOR = "population_prior"
DEPLOYABLE_SPEAKER_HISTORY_PRIOR = "speaker_history_prior_deployable"
DEPLOYABLE_SPEAKER_HISTORY_PRIOR_VISUAL = "speaker_history_prior_deployable_visual"
DEPLOYABLE_SPEAKER_HISTORY_PRIOR_MULTIMODAL = "speaker_history_prior_deployable_multimodal"
ORACLE_SPEAKER_HISTORY_PRIOR = "speaker_history_prior_oracle"
DIALOGUE_CONTEXT_PRIOR = "dialogue_context_prior"
ACOUSTIC_LIKELIHOOD = "acoustic_likelihood"
VISUAL_LIKELIHOOD = "visual_likelihood"
MULTIMODAL_SENSORY_LIKELIHOOD = "multimodal_sensory_likelihood"
EQUAL_WEIGHT_FUSION_CONTEXT_ACOUSTIC = "equal_weight_fusion_context_acoustic"
EQUAL_WEIGHT_FUSION_CONTEXT_VISUAL = "equal_weight_fusion_context_visual"
EQUAL_WEIGHT_FUSION_SENSORY = "equal_weight_fusion_sensory"
EQUAL_WEIGHT_FUSION_CONTEXT_MULTIMODAL = "equal_weight_fusion_context_multimodal"
EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC = "equal_weight_fusion_speaker_history_deployable_acoustic"
EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_VISUAL = "equal_weight_fusion_speaker_history_deployable_visual"
EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL = "equal_weight_fusion_speaker_history_deployable_multimodal"
PWP_CONTEXT_ACOUSTIC = "pwp_context_acoustic"
PWP_CONTEXT_VISUAL = "pwp_context_visual"
PWP_CONTEXT_MULTIMODAL = "pwp_context_multimodal"
PWP_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC = "pwp_speaker_history_deployable_acoustic"
PWP_SPEAKER_HISTORY_DEPLOYABLE_VISUAL = "pwp_speaker_history_deployable_visual"
PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL = "pwp_speaker_history_deployable_multimodal"
PWP_SPEAKER_HISTORY_ORACLE_ACOUSTIC = "pwp_speaker_history_oracle_acoustic"
RATER_SPECIFIC_BASELINE_PRIOR = "rater_specific_baseline_prior"
FLAT_PRIOR = "flat_prior"


@dataclass(frozen=True)
class ModelTerm:
    """Terminology for one probabilistic component or comparison model."""

    model_id: str
    display_name: str
    compact_label: str
    plot_label: str
    role: str
    is_deployable: bool
    legacy_display_names: tuple[str, ...] = ()
    legacy_column_prefixes: tuple[str, ...] = ()
    paper_label: str = ""


MODEL_TERMS: tuple[ModelTerm, ...] = (
    ModelTerm(
        POPULATION_PRIOR,
        "Population Prior",
        "Population Prior",
        "Population Prior",
        "prior",
        True,
        ("Global Prior",),
        ("global",),
        paper_label="Pop",
    ),
    ModelTerm(
        DEPLOYABLE_SPEAKER_HISTORY_PRIOR,
        "Deployable Speaker-History Prior",
        "Deployable Speaker-History Prior",
        "Deployable Speaker-\nHistory Prior",
        "prior",
        True,
        ("Deployable Speaker-History Prior",),
        ("spk_dep",),
        paper_label="H(\\ell_A)",
    ),
    ModelTerm(
        DEPLOYABLE_SPEAKER_HISTORY_PRIOR_VISUAL,
        "Deployable Speaker-History Prior (Visual)",
        "Deployable Visual Speaker-History Prior",
        "Deployable Visual\nSpeaker-History Prior",
        "prior",
        True,
        ("Deployable Visual Speaker-History Prior",),
        ("spk_dep_vis",),
        paper_label="H(\\ell_V)",
    ),
    ModelTerm(
        DEPLOYABLE_SPEAKER_HISTORY_PRIOR_MULTIMODAL,
        "Deployable Speaker-History Prior (Multimodal)",
        "Deployable Multimodal Speaker-History Prior",
        "Deployable Multimodal\nSpeaker-History Prior",
        "prior",
        True,
        ("Deployable Multimodal Speaker-History Prior",),
        ("spk_dep_multi",),
        paper_label="H(\\ell_{AV})",
    ),
    ModelTerm(
        ORACLE_SPEAKER_HISTORY_PRIOR,
        "Oracle Speaker-History Prior*",
        "Oracle Speaker-History Prior*",
        "Oracle Speaker-\nHistory Prior*",
        "prior",
        False,
        ("Oracle Speaker-History Prior",),
        ("spk_orc",),
        paper_label="H*",
    ),
    ModelTerm(
        DIALOGUE_CONTEXT_PRIOR,
        "Dialogue-Context Prior (GRU)",
        "Dialogue-Context Prior",
        "Dialogue-Context\nPrior (GRU)",
        "prior",
        True,
        ("GRU Context Prior", "GRU Expectation Prior", "Context History (GRU)"),
        ("gru",),
        paper_label="C",
    ),
    ModelTerm(
        ACOUSTIC_LIKELIHOOD,
        "Acoustic Likelihood (Wav2Vec2)",
        "Acoustic Likelihood",
        "Acoustic Likelihood\n(Wav2Vec2)",
        "likelihood",
        True,
        ("Wav2Vec2 Likelihood", "Sensory (Wav2Vec2)"),
        ("wav2vec2", "w2v"),
        paper_label="\\ell_A",
    ),
    ModelTerm(
        VISUAL_LIKELIHOOD,
        "Visual Likelihood (EmoNet)",
        "Visual Likelihood",
        "Visual Likelihood\n(EmoNet)",
        "likelihood",
        True,
        ("EmoNet Likelihood", "Visual Likelihood"),
        ("visual", "emonet"),
        paper_label="\\ell_V",
    ),
    ModelTerm(
        MULTIMODAL_SENSORY_LIKELIHOOD,
        "Multimodal Sensory Likelihood (Wav2Vec2 + EmoNet)",
        "Multimodal Sensory Likelihood",
        "Multimodal Sensory\nLikelihood",
        "likelihood",
        True,
        ("Sensory Likelihood (Audio + Visual)", "Multimodal Sensory Likelihood"),
        ("post_sensory", "sensory_multi"),
        paper_label="\\ell_{AV}",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_CONTEXT_ACOUSTIC,
        "Equal-Weight Fusion (Dialogue Context + Acoustic)",
        "Equal-Weight Fusion",
        "Equal-Weight Fusion\n(Dialogue Context + Acoustic)",
        "fusion",
        True,
        ("Equal-Weight Fusion (GRU + Wav2Vec2)",),
        ("equal",),
        paper_label="EWF(C, \\ell_A)",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_CONTEXT_VISUAL,
        "Equal-Weight Fusion (Dialogue Context + Visual)",
        "Context-Visual Equal-Weight",
        "Equal-Weight Fusion\n(Dialogue Context + Visual)",
        "fusion",
        True,
        ("Equal-Weight Fusion (GRU + EmoNet)",),
        ("equal_gru_visual",),
        paper_label="EWF(C, \\ell_V)",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_SENSORY,
        "Equal-Weight Sensory Fusion (Acoustic + Visual)",
        "Equal-Weight Sensory Fusion",
        "Equal-Weight Sensory\nFusion (Audio + Video)",
        "fusion",
        True,
        ("Equal-Weight Sensory Fusion (Wav2Vec2 + EmoNet)",),
        ("equal_sensory",),
        paper_label="EWF(\\ell_A, \\ell_V)",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_CONTEXT_MULTIMODAL,
        "Equal-Weight Fusion (Dialogue Context + Multimodal Sensory)",
        "Multimodal Equal-Weight Fusion",
        "Equal-Weight Fusion\n(Dialogue Context + Multimodal)",
        "fusion",
        True,
        ("Equal-Weight Fusion (GRU + Audio + Visual)",),
        ("equal_gru_multi",),
        paper_label="EWF(C, \\ell_{AV})",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC,
        "Equal-Weight Fusion (Deployable Speaker History + Acoustic)",
        "Deployable Speaker-History Equal-Weight Fusion",
        "Equal-Weight Fusion\n(Deployable Speaker History + Acoustic)",
        "fusion",
        True,
        ("Equal-Weight Fusion (Deployable Spk. Hist. + Wav2Vec2)",),
        ("equal_spk_dep",),
        paper_label="EWF(H(\\ell_A), \\ell_A)",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_VISUAL,
        "Equal-Weight Fusion (Deployable Speaker History + Visual)",
        "Deployable Visual Speaker-History Equal-Weight Fusion",
        "Equal-Weight Fusion\n(Deployable Speaker History + Visual)",
        "fusion",
        True,
        ("Equal-Weight Fusion (Deployable Visual Spk. Hist. + EmoNet)",),
        ("equal_spk_dep_vis",),
        paper_label="EWF(H(\\ell_V), \\ell_V)",
    ),
    ModelTerm(
        EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL,
        "Equal-Weight Fusion (Deployable Speaker History + Multimodal Sensory)",
        "Deployable Multimodal Speaker-History Equal-Weight Fusion",
        "Equal-Weight Fusion\n(Deployable Speaker History + Multimodal)",
        "fusion",
        True,
        ("Equal-Weight Fusion (Deployable Multimodal Spk. Hist. + Multimodal Sensory)",),
        ("equal_spk_dep_multi",),
        paper_label="EWF(H(\\ell_{AV}), \\ell_{AV})",
    ),
    ModelTerm(
        PWP_CONTEXT_ACOUSTIC,
        "Precision-Weighted Posterior (Dialogue Context + Acoustic)",
        "Context-Acoustic PWP",
        "Precision-Weighted Posterior\n(Dialogue Context + Acoustic)",
        "posterior",
        True,
        ("Precision-Weighted Posterior (GRU + Wav2Vec2)",),
        ("post_gru_w2v", "post"),
        paper_label="PWP(C, \\ell_A)",
    ),
    ModelTerm(
        PWP_CONTEXT_VISUAL,
        "Precision-Weighted Posterior (Dialogue Context + Visual)",
        "Context-Visual PWP",
        "Precision-Weighted Posterior\n(Dialogue Context + Visual)",
        "posterior",
        True,
        ("Precision-Weighted Posterior (GRU + EmoNet)",),
        ("post_gru_visual",),
        paper_label="PWP(C, \\ell_V)",
    ),
    ModelTerm(
        PWP_CONTEXT_MULTIMODAL,
        "Precision-Weighted Posterior (Dialogue Context + Multimodal Sensory)",
        "Context-Multimodal PWP",
        "Precision-Weighted Posterior\n(Dialogue Context + Multimodal)",
        "posterior",
        True,
        ("Precision-Weighted Posterior (GRU + Audio + Visual)",),
        ("post_gru_multi",),
        paper_label="PWP(C, \\ell_{AV})",
    ),
    ModelTerm(
        PWP_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC,
        "Precision-Weighted Posterior (Deployable Speaker History + Acoustic)",
        "Deployable Speaker-History PWP",
        "Precision-Weighted Posterior\n(Deployable Speaker History + Acoustic)",
        "posterior",
        True,
        ("Precision-Weighted Posterior (Deployable Spk. Hist. + Wav2Vec2)",),
        ("post_spk_dep_w2v",),
        paper_label="PWP(H(\\ell_A), \\ell_A)",
    ),
    ModelTerm(
        PWP_SPEAKER_HISTORY_DEPLOYABLE_VISUAL,
        "Precision-Weighted Posterior (Deployable Speaker History + Visual)",
        "Deployable Visual Speaker-History PWP",
        "Precision-Weighted Posterior\n(Deployable Speaker History + Visual)",
        "posterior",
        True,
        ("Precision-Weighted Posterior (Deployable Visual Spk. Hist. + EmoNet)",),
        ("post_spk_dep_visual",),
        paper_label="PWP(H(\\ell_V), \\ell_V)",
    ),
    ModelTerm(
        PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL,
        "Precision-Weighted Posterior (Deployable Speaker History + Multimodal Sensory)",
        "Deployable Multimodal PWP",
        "Precision-Weighted Posterior\n(Deployable Speaker History + Multimodal)",
        "posterior",
        True,
        ("Precision-Weighted Posterior (Deployable Multimodal Spk. Hist. + Multimodal Sensory)",),
        ("post_spk_dep_multi",),
        paper_label="PWP(H(\\ell_{AV}), \\ell_{AV})",
    ),
    ModelTerm(
        PWP_SPEAKER_HISTORY_ORACLE_ACOUSTIC,
        "Precision-Weighted Posterior (Oracle Speaker History + Acoustic)*",
        "Oracle Speaker-History PWP*",
        "Precision-Weighted Posterior\n(Oracle Speaker History + Acoustic)*",
        "posterior",
        False,
        ("Precision-Weighted Posterior (Oracle Spk. Hist. + Wav2Vec2)",),
        ("post_spk_orc_w2v",),
        paper_label="PWP(H*, \\ell_A)",
    ),
    ModelTerm(
        RATER_SPECIFIC_BASELINE_PRIOR,
        "Rater-Specific Baseline Prior",
        "Rater-Specific Prior",
        "Rater-Specific\nPrior",
        "prior",
        False,
        ("Rater Prior",),
        paper_label="R",
    ),
    ModelTerm(
        FLAT_PRIOR,
        "Flat Prior (no context)",
        "Flat Prior",
        "Flat Prior\n(no context)",
        "prior",
        True,
        ("Flat (No Context)",),
        paper_label="F",
    ),
)

MODEL_BY_ID = {term.model_id: term for term in MODEL_TERMS}
_NUMERIC_PREFIX = re.compile(r"^\s*\d+[a-z]?\.\s*")


def get_term(model_id: str) -> ModelTerm:
    """Return canonical terminology for a stable model identifier."""
    try:
        return MODEL_BY_ID[model_id]
    except KeyError as exc:
        raise KeyError(f"Unknown predictive-processing model id: {model_id!r}") from exc


def display_name(model_id: str) -> str:
    """Return the canonical public label for ``model_id``."""
    return get_term(model_id).display_name


def compact_label(model_id: str) -> str:
    """Return the compact, PWP-aware label for narrow figures and tables."""
    return get_term(model_id).compact_label


def plot_label(model_id: str) -> str:
    """Return a line-broken canonical label suitable for figure axes."""
    return get_term(model_id).plot_label


def paper_label(model_id: str) -> str:
    """Return the paper-facing functional notation for ``model_id``.

    Example: ``paper_label(PWP_CONTEXT_ACOUSTIC)`` returns ``"PWP(C, A)"``.
    The notation is an alias layer for manuscript prose and figures only;
    it never replaces ``model_id`` in code, exports, or analysis artifacts.
    """
    return get_term(model_id).paper_label


def canonical_model_id(label_or_id: str) -> str:
    """Resolve a canonical model ID from a v2 ID or a legacy display label.

    Numbered table rows, terminal oracle asterisks, and calibrated/uncalibrated
    suffixes are accepted so older summary files can be displayed safely.
    """
    candidate = str(label_or_id).strip()
    if candidate in MODEL_BY_ID:
        return candidate

    candidate = _NUMERIC_PREFIX.sub("", candidate).strip()
    candidate = re.sub(r"\s*\((?:Uncalibrated|Calibrated)\)\s*$", "", candidate)
    candidate_without_oracle_marker = candidate.rstrip("*").strip()

    for term in MODEL_TERMS:
        aliases = (term.display_name, *term.legacy_display_names)
        if candidate in aliases or candidate_without_oracle_marker in aliases:
            return term.model_id
    raise KeyError(f"No canonical model ID is registered for {label_or_id!r}")


def canonical_display_name(label_or_id: str) -> str:
    """Resolve a legacy table label or ID to its canonical display label."""
    return display_name(canonical_model_id(label_or_id))


def is_oracle(model_id: str) -> bool:
    """Whether a model is a non-deployable oracle upper-bound reference."""
    return not get_term(model_id).is_deployable


# Longest prefixes must be checked first: ``post_spk_dep_w2v`` must never be
# captured by the generic ``post`` mapping.
_COLUMN_PREFIXES: tuple[tuple[str, str], ...] = tuple(
    sorted(
        (
            ("w_p_spk_dep_multi", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL}_{DEPLOYABLE_SPEAKER_HISTORY_PRIOR_MULTIMODAL}_weight"),
            ("w_sens_spk_dep_multi", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL}_{MULTIMODAL_SENSORY_LIKELIHOOD}_weight"),
            ("w_aud_spk_dep_multi", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL}_{ACOUSTIC_LIKELIHOOD}_weight"),
            ("w_vis_spk_dep_multi", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL}_{VISUAL_LIKELIHOOD}_weight"),
            ("w_p_spk_dep_visual", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_VISUAL}_{DEPLOYABLE_SPEAKER_HISTORY_PRIOR_VISUAL}_weight"),
            ("w_l_spk_dep_visual", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_VISUAL}_{VISUAL_LIKELIHOOD}_weight"),
            ("w_p_spk_dep_w2v", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC}_{DEPLOYABLE_SPEAKER_HISTORY_PRIOR}_weight"),
            ("w_l_spk_dep_w2v", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC}_{ACOUSTIC_LIKELIHOOD}_weight"),
            ("w_p_spk_orc_w2v", f"{PWP_SPEAKER_HISTORY_ORACLE_ACOUSTIC}_{ORACLE_SPEAKER_HISTORY_PRIOR}_weight"),
            ("w_l_spk_orc_w2v", f"{PWP_SPEAKER_HISTORY_ORACLE_ACOUSTIC}_{ACOUSTIC_LIKELIHOOD}_weight"),
            ("w_p_gru_multi", f"{PWP_CONTEXT_MULTIMODAL}_{DIALOGUE_CONTEXT_PRIOR}_weight"),
            ("w_sens_gru_multi", f"{PWP_CONTEXT_MULTIMODAL}_{MULTIMODAL_SENSORY_LIKELIHOOD}_weight"),
            ("w_aud_gru_multi", f"{PWP_CONTEXT_MULTIMODAL}_{ACOUSTIC_LIKELIHOOD}_weight"),
            ("w_vis_gru_multi", f"{PWP_CONTEXT_MULTIMODAL}_{VISUAL_LIKELIHOOD}_weight"),
            ("w_p_gru_visual", f"{PWP_CONTEXT_VISUAL}_{DIALOGUE_CONTEXT_PRIOR}_weight"),
            ("w_l_gru_visual", f"{PWP_CONTEXT_VISUAL}_{VISUAL_LIKELIHOOD}_weight"),
            ("w_p_gru_w2v", f"{PWP_CONTEXT_ACOUSTIC}_{DIALOGUE_CONTEXT_PRIOR}_weight"),
            ("w_l_gru_w2v", f"{PWP_CONTEXT_ACOUSTIC}_{ACOUSTIC_LIKELIHOOD}_weight"),
            ("w_aud_sensory", f"{MULTIMODAL_SENSORY_LIKELIHOOD}_{ACOUSTIC_LIKELIHOOD}_weight"),
            ("w_vis_sensory", f"{MULTIMODAL_SENSORY_LIKELIHOOD}_{VISUAL_LIKELIHOOD}_weight"),
            ("delta_unstand_spk_dep_multi", "speaker_history_deployable_multimodal_discrepancy"),
            ("delta_stand_spk_dep_multi", "speaker_history_deployable_multimodal_standardized_discrepancy"),
            ("delta_unstand_spk_dep_visual", "speaker_history_deployable_visual_discrepancy"),
            ("delta_stand_spk_dep_visual", "speaker_history_deployable_visual_standardized_discrepancy"),
            ("delta_unstand_spk_dep_w2v", "speaker_history_deployable_acoustic_discrepancy"),
            ("delta_stand_spk_dep_w2v", "speaker_history_deployable_acoustic_standardized_discrepancy"),
            ("delta_unstand_spk_orc_w2v", "speaker_history_oracle_acoustic_discrepancy"),
            ("delta_stand_spk_orc_w2v", "speaker_history_oracle_acoustic_standardized_discrepancy"),
            ("delta_unstand_gru_sensory", "dialogue_context_multimodal_sensory_discrepancy"),
            ("delta_stand_gru_sensory", "dialogue_context_multimodal_sensory_standardized_discrepancy"),
            ("delta_unstand_gru_multi", "dialogue_context_multimodal_sensory_discrepancy"),
            ("delta_stand_gru_multi", "dialogue_context_multimodal_sensory_standardized_discrepancy"),
            ("delta_unstand_gru_visual", "dialogue_context_visual_discrepancy"),
            ("delta_stand_gru_visual", "dialogue_context_visual_standardized_discrepancy"),
            ("delta_unstand_aud_vis", "acoustic_visual_discrepancy"),
            ("delta_stand_aud_vis", "acoustic_visual_standardized_discrepancy"),
            ("delta_unstand_gru_w2v", "dialogue_context_acoustic_discrepancy"),
            ("delta_stand_gru_w2v", "dialogue_context_acoustic_standardized_discrepancy"),
            ("abs_err_post_spk_dep_multi", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL}_absolute_error"),
            ("abs_err_post_spk_dep_visual", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_VISUAL}_absolute_error"),
            ("abs_err_post_spk_dep_w2v", f"{PWP_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC}_absolute_error"),
            ("abs_err_post_spk_orc_w2v", f"{PWP_SPEAKER_HISTORY_ORACLE_ACOUSTIC}_absolute_error"),
            ("abs_err_post_gru_multi", f"{PWP_CONTEXT_MULTIMODAL}_absolute_error"),
            ("abs_err_post_gru_visual", f"{PWP_CONTEXT_VISUAL}_absolute_error"),
            ("abs_err_post_gru_w2v", f"{PWP_CONTEXT_ACOUSTIC}_absolute_error"),
            ("abs_err_sensory", f"{MULTIMODAL_SENSORY_LIKELIHOOD}_absolute_error"),
            ("abs_err_visual", f"{VISUAL_LIKELIHOOD}_absolute_error"),
            ("abs_err_gru", f"{DIALOGUE_CONTEXT_PRIOR}_absolute_error"),
            ("abs_err_w2v", f"{ACOUSTIC_LIKELIHOOD}_absolute_error"),
            ("post_spk_dep_multi", PWP_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL),
            ("post_spk_dep_visual", PWP_SPEAKER_HISTORY_DEPLOYABLE_VISUAL),
            ("post_spk_dep_w2v", PWP_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC),
            ("post_spk_orc_w2v", PWP_SPEAKER_HISTORY_ORACLE_ACOUSTIC),
            ("post_gru_multi", PWP_CONTEXT_MULTIMODAL),
            ("post_gru_visual", PWP_CONTEXT_VISUAL),
            ("post_gru_w2v", PWP_CONTEXT_ACOUSTIC),
            ("post_sensory", MULTIMODAL_SENSORY_LIKELIHOOD),
            ("spk_dep_multi", DEPLOYABLE_SPEAKER_HISTORY_PRIOR_MULTIMODAL),
            ("spk_dep_vis", DEPLOYABLE_SPEAKER_HISTORY_PRIOR_VISUAL),
            ("spk_dep", DEPLOYABLE_SPEAKER_HISTORY_PRIOR),
            ("spk_orc", ORACLE_SPEAKER_HISTORY_PRIOR),
            ("wav2vec2", ACOUSTIC_LIKELIHOOD),
            ("visual", VISUAL_LIKELIHOOD),
            ("emonet", VISUAL_LIKELIHOOD),
            ("global", POPULATION_PRIOR),
            ("gru", DIALOGUE_CONTEXT_PRIOR),
            ("w2v", ACOUSTIC_LIKELIHOOD),
            ("equal_spk_dep", EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_ACOUSTIC),
            ("equal_spk_dep_vis", EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_VISUAL),
            ("equal_spk_dep_multi", EQUAL_WEIGHT_FUSION_SPEAKER_HISTORY_DEPLOYABLE_MULTIMODAL),
            ("equal_gru_multi", EQUAL_WEIGHT_FUSION_CONTEXT_MULTIMODAL),
            ("equal_gru_visual", EQUAL_WEIGHT_FUSION_CONTEXT_VISUAL),
            ("equal_sensory", EQUAL_WEIGHT_FUSION_SENSORY),
            ("equal", EQUAL_WEIGHT_FUSION_CONTEXT_ACOUSTIC),
            ("post", PWP_CONTEXT_ACOUSTIC),
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
)

_TAIL_REPLACEMENTS = {
    "mu": "mean",
    "mu_raw": "mean_raw",
    "mu_cal": "mean_calibrated",
    "var": "variance",
    "var_raw": "variance_raw",
    "var_cal": "variance_calibrated",
    "var_input_cal": "input_variance_calibrated",
    "raw": "raw",
    "cal": "calibrated",
    "val_mu": "valence_mean",
    "val_mu_raw": "valence_mean_raw",
    "val_mu_cal": "valence_mean_calibrated",
    "val_var": "valence_variance",
    "val_var_raw": "valence_variance_raw",
    "val_var_cal": "valence_variance_calibrated",
    "val_var_input_cal": "valence_input_variance_calibrated",
    "aro_mu": "arousal_mean",
    "aro_mu_raw": "arousal_mean_raw",
    "aro_mu_cal": "arousal_mean_calibrated",
    "aro_var": "arousal_variance",
    "aro_var_raw": "arousal_variance_raw",
    "aro_var_cal": "arousal_variance_calibrated",
    "aro_var_input_cal": "arousal_input_variance_calibrated",
}
_INVERSE_TAIL_REPLACEMENTS = {value: key for key, value in _TAIL_REPLACEMENTS.items()}


def canonical_column_name(column_name: str) -> str:
    """Translate a v1 analysis column name to its v2 semantic equivalent."""
    name = str(column_name)
    for legacy_prefix, canonical_prefix in _COLUMN_PREFIXES:
        if name == legacy_prefix:
            return canonical_prefix
        prefix = f"{legacy_prefix}_"
        if name.startswith(prefix):
            tail = name[len(prefix):]
            tail = _TAIL_REPLACEMENTS.get(tail, tail)
            tail = tail.replace("_val_", "_valence_").replace("_aro_", "_arousal_")
            return f"{canonical_prefix}_{tail}"
    return name


def to_v2_analysis_schema(frame: Any) -> Any:
    """Return a copy of a pandas-like analysis frame in the version-2 schema.

    The function intentionally accepts a duck-typed dataframe to avoid a hard
    pandas dependency in this terminology module.  Existing v2 columns are
    preserved and a ``schema_version`` marker is added when absent.
    """
    result = frame.copy()
    rename_map = {
        column: canonical_column_name(column)
        for column in result.columns
        if canonical_column_name(column) != column
    }
    duplicate_targets = [
        target for target in rename_map.values() if target in result.columns
    ]
    if duplicate_targets:
        joined = ", ".join(sorted(set(duplicate_targets)))
        raise ValueError(f"Cannot map legacy analysis columns onto existing v2 columns: {joined}")
    result = result.rename(columns=rename_map)
    if "schema_version" not in result.columns:
        result.insert(0, "schema_version", SCHEMA_VERSION)
    return result


def legacy_column_name(column_name: str, *, acoustic_prefix: str = "w2v") -> str:
    """Translate a v2 semantic column name back to a chosen v1 convention.

    Turn-level analysis used ``w2v_*`` whereas fold prediction files used
    ``wav2vec2_*``. The v2 semantic schema deliberately collapses that naming
    accident, so callers must choose the legacy contract they are serving.
    """
    if acoustic_prefix not in {"w2v", "wav2vec2"}:
        raise ValueError("acoustic_prefix must be 'w2v' or 'wav2vec2'")
    name = str(column_name)
    inverse_prefixes = sorted(
        (
            (canonical, legacy)
            for legacy, canonical in _COLUMN_PREFIXES
            if not (
                canonical == ACOUSTIC_LIKELIHOOD
                and legacy in {"w2v", "wav2vec2"}
                and legacy != acoustic_prefix
            )
        ),
        key=lambda item: len(item[0]),
        reverse=True,
    )
    for canonical_prefix, legacy_prefix in inverse_prefixes:
        if name == canonical_prefix:
            return legacy_prefix
        prefix = f"{canonical_prefix}_"
        if name.startswith(prefix):
            tail = name[len(prefix):]
            tail = tail.replace("_valence_", "_val_").replace("_arousal_", "_aro_")
            tail = _INVERSE_TAIL_REPLACEMENTS.get(tail, tail)
            return f"{legacy_prefix}_{tail}"
    return name


def _to_v1_schema(frame: Any, *, acoustic_prefix: str) -> Any:
    """Return a legacy-schema copy with the requested acoustic field prefix."""
    result = frame.copy()
    if "schema_version" in result.columns:
        result = result.drop(columns=["schema_version"])
    rename_map = {
        column: legacy_column_name(column, acoustic_prefix=acoustic_prefix)
        for column in result.columns
        if legacy_column_name(column, acoustic_prefix=acoustic_prefix) != column
    }
    duplicate_targets = [
        target for target in rename_map.values() if target in result.columns
    ]
    if duplicate_targets:
        joined = ", ".join(sorted(set(duplicate_targets)))
        raise ValueError(f"Cannot map v2 analysis columns onto existing v1 columns: {joined}")
    return result.rename(columns=rename_map)


def to_v1_analysis_schema(frame: Any) -> Any:
    """Adapt v2 data for historic turn-level analysis code (``w2v_*`` fields)."""
    return _to_v1_schema(frame, acoustic_prefix="w2v")


def to_v1_prediction_schema(frame: Any) -> Any:
    """Adapt v2 data for historic fold-prediction readers (``wav2vec2_*``)."""
    return _to_v1_schema(frame, acoustic_prefix="wav2vec2")


def write_v2_analysis_exports(frame: Any, csv_path: Path, json_path: Path) -> Any:
    """Write versioned CSV and JSON exports without mutating legacy artifacts."""
    result = to_v2_analysis_schema(frame)
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(csv_path, index=False)
    result.to_json(json_path, orient="records", indent=2)
    return result


def prediction_artifact_candidates(
    directory: Path,
    model_id: str,
    fold: int,
    split: str = "test",
) -> tuple[Path, ...]:
    """Return v2-first and v1-compatible prediction-file candidates.

    ``split`` is ``"test"`` or ``"validation"``.  New training runs write
    the first path; readers may continue to consume historical second paths.
    """
    if split not in {"test", "validation"}:
        raise ValueError("split must be 'test' or 'validation'")
    legacy_stem_by_id = {
        DIALOGUE_CONTEXT_PRIOR: "gru",
        ACOUSTIC_LIKELIHOOD: "wav2vec2",
        VISUAL_LIKELIHOOD: "visual_likelihood_emonet",
    }
    try:
        legacy_stem = legacy_stem_by_id[model_id]
    except KeyError as exc:
        raise KeyError(f"No prediction artifact convention for {model_id!r}") from exc

    suffix = "_val_predictions.csv" if split == "validation" else "_predictions.csv"
    stem_tag = "gru" if model_id == DIALOGUE_CONTEXT_PRIOR else ("wav2vec2" if model_id == ACOUSTIC_LIKELIHOOD else "emonet")
    v2_stem = f"{model_id}_{stem_tag}"
    candidates = [
        directory / f"{v2_stem}_fold_{fold}{suffix}",
        directory / f"{legacy_stem}_fold_{fold}{suffix}",
    ]
    if model_id == VISUAL_LIKELIHOOD:
        candidates.extend([
            directory / f"emonet_visual_likelihood_fold_{fold}{suffix}",
            directory / f"visual_fold_{fold}{suffix}",
        ])
    return tuple(candidates)


def resolve_prediction_artifact(directory: Path, model_id: str, fold: int, split: str = "test") -> Path:
    """Resolve a v2 prediction artifact first, falling back to the v1 file."""
    candidates = prediction_artifact_candidates(directory, model_id, fold, split)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    candidates_text = ", ".join(str(path) for path in candidates)
    raise FileNotFoundError(f"No prediction artifact found. Checked: {candidates_text}")


def write_v2_prediction_export(frame: Any, directory: Path, model_id: str, fold: int, split: str = "test") -> Path:
    """Write a semantic v2 prediction export alongside the legacy v1 file."""
    destination = prediction_artifact_candidates(directory, model_id, fold, split)[0]
    destination.parent.mkdir(parents=True, exist_ok=True)
    to_v2_analysis_schema(frame).to_csv(destination, index=False)
    return destination


_DEMO_LEGACY_FIELDS = {
    DIALOGUE_CONTEXT_PRIOR: "expect",
    ACOUSTIC_LIKELIHOOD: "sensory",
    DEPLOYABLE_SPEAKER_HISTORY_PRIOR: "speaker_rolling",
}


def demo_distribution(record: Mapping[str, Any], model_id: str, dimension: str) -> tuple[float, float]:
    """Read a demo mean/standard deviation from v2 or historical v1 JSON.

    Version 2 stores distributions under ``record['distributions'][model_id]``;
    version 1 keeps the historical flat fields. This boundary adapter lets the
    interface move to semantic IDs without rewriting existing demo data.
    """
    if dimension not in {"valence", "arousal"}:
        raise ValueError("dimension must be 'valence' or 'arousal'")

    distributions = record.get("distributions")
    if isinstance(distributions, Mapping) and model_id in distributions:
        component = distributions[model_id][dimension]
        return float(component["mean"]), float(component["sigma"])

    try:
        legacy_prefix = _DEMO_LEGACY_FIELDS[model_id]
    except KeyError as exc:
        raise KeyError(f"No demo-data schema mapping for {model_id!r}") from exc
    return (
        float(record[f"{legacy_prefix}_{dimension}_mu"]),
        float(record[f"{legacy_prefix}_{dimension}_sigma"]),
    )


def demo_preceding_turn_count(record: Mapping[str, Any], model_id: str) -> int:
    """Read an optional same-speaker history count from either demo schema."""
    if model_id != DEPLOYABLE_SPEAKER_HISTORY_PRIOR:
        raise KeyError(f"No preceding-turn count is defined for {model_id!r}")
    distributions = record.get("distributions")
    if isinstance(distributions, Mapping) and model_id in distributions:
        return int(distributions[model_id].get("n_preceding_turns", 0))
    return int(record.get("speaker_rolling_n_prior", 0))


def to_v2_demo_record(record: Mapping[str, Any]) -> dict[str, Any]:
    """Convert one v1 demo-data record into the semantic v2 JSON structure."""
    if isinstance(record.get("distributions"), Mapping):
        result = dict(record)
        result.setdefault("schema_version", SCHEMA_VERSION)
        return result

    result = dict(record)
    distributions: dict[str, dict[str, dict[str, float]]] = {}
    for model_id, legacy_prefix in _DEMO_LEGACY_FIELDS.items():
        distributions[model_id] = {}
        for dimension in ("valence", "arousal"):
            distributions[model_id][dimension] = {
                "mean": float(result.pop(f"{legacy_prefix}_{dimension}_mu")),
                "sigma": float(result.pop(f"{legacy_prefix}_{dimension}_sigma")),
            }
    n_prior = result.pop("speaker_rolling_n_prior", None)
    if n_prior is not None:
        distributions[DEPLOYABLE_SPEAKER_HISTORY_PRIOR]["n_preceding_turns"] = int(n_prior)
    result["schema_version"] = SCHEMA_VERSION
    result["distributions"] = distributions
    return result


def to_v2_demo_records(records: Iterable[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Convert a v1/v2 demo-data sequence into version-2 records."""
    return [to_v2_demo_record(record) for record in records]


def forbidden_public_terms() -> tuple[str, ...]:
    """Return deprecated display names for lightweight terminology checks."""
    return (
        "Global Prior",
        "GRU Context Prior",
        "Context History (GRU)",
        "Speaker History (Rolling Mean)",
        "GRU Expectation Prior",
        "Sensory (Wav2Vec2)",
        "Equal-Weight Fusion (GRU + Wav2Vec2)",
        "Precision-Weighted Posterior (GRU + Wav2Vec2)",
    )
