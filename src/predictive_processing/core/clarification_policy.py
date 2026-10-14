"""Transparent interaction policy for uncertainty-aware robot responses.

The policy treats disagreement as a limitation of the robot's current model,
not as a judgement about the person speaking.  It is intentionally small:
normal responses continue when the estimate is clear, a neutral attentive cue
is used when confidence is moderate, and the robot asks for more context when
its sources conflict or the posterior remains broad.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from math import sqrt
from typing import Iterable


CAUTION_DISCREPANCY_THRESHOLD = 1.0
CLARIFY_DISCREPANCY_THRESHOLD = 1.5
CAUTION_POSTERIOR_SIGMA_THRESHOLD = 0.32
CLARIFY_POSTERIOR_SIGMA_THRESHOLD = 0.36
# The policy waits through the reply to the third human utterance.  This gives
# the dialogue model enough conversational evidence before uncertainty can
# change the robot's social response.
DIALOGUE_CONTEXT_WARMUP_SPEAKER_TURNS = 3


class InteractionPolicy(StrEnum):
    """The action the robot should take after an affect inference."""

    RESPOND = "respond"
    CAUTIOUS = "cautious"
    CLARIFY = "clarify"


@dataclass(frozen=True)
class DimensionPolicyDecision:
    """Policy evidence for one affective dimension."""

    dimension: str
    discrepancy: float
    standardized_discrepancy: float
    posterior_sigma: float
    action: InteractionPolicy
    triggers: tuple[str, ...]


@dataclass(frozen=True)
class PolicyDecision:
    """An inspectable, user-facing interaction-policy recommendation."""

    action: InteractionPolicy
    dimensions: tuple[DimensionPolicyDecision, ...]
    triggers: tuple[str, ...]
    gesture_override: str | None
    response_guidance: str
    context_warmup: bool = False
    context_turn_count: int | None = None

    @property
    def label(self) -> str:
        return {
            InteractionPolicy.RESPOND: "Respond normally",
            InteractionPolicy.CAUTIOUS: "Respond cautiously",
            InteractionPolicy.CLARIFY: "Ask for clarification",
        }[self.action]

    @property
    def icon(self) -> str:
        return {
            InteractionPolicy.RESPOND: ":material/check_circle:",
            InteractionPolicy.CAUTIOUS: ":material/visibility:",
            InteractionPolicy.CLARIFY: ":material/psychology:",
        }[self.action]

    @property
    def badge_color(self) -> str:
        return {
            InteractionPolicy.RESPOND: "green",
            InteractionPolicy.CAUTIOUS: "orange",
            InteractionPolicy.CLARIFY: "red",
        }[self.action]

    @property
    def clarification_prompt(self) -> str:
        return (
            "I may be reading that incorrectly. Would you like to tell me "
            "a little more about how that felt?"
        )


def standardized_discrepancy(
    mu_prior: float,
    sigma_prior: float,
    mu_likelihood: float,
    sigma_likelihood: float,
) -> float:
    """Return the uncertainty-scaled prior--likelihood discrepancy, ``δ*``."""
    if sigma_prior <= 0 or sigma_likelihood <= 0:
        raise ValueError("Prior and likelihood standard deviations must be positive.")
    return abs(mu_likelihood - mu_prior) / sqrt(sigma_prior**2 + sigma_likelihood**2)


def decide_dimension_policy(
    *,
    dimension: str,
    mu_prior: float,
    sigma_prior: float,
    mu_likelihood: float,
    sigma_likelihood: float,
    posterior_sigma: float,
) -> DimensionPolicyDecision:
    """Classify one dimension from source conflict and posterior uncertainty."""
    if posterior_sigma <= 0:
        raise ValueError("Posterior standard deviation must be positive.")

    discrepancy = abs(mu_likelihood - mu_prior)
    delta_star = standardized_discrepancy(
        mu_prior, sigma_prior, mu_likelihood, sigma_likelihood
    )
    triggers: list[str] = []

    if delta_star >= CLARIFY_DISCREPANCY_THRESHOLD:
        triggers.append("source disagreement")
    if posterior_sigma >= CLARIFY_POSTERIOR_SIGMA_THRESHOLD:
        triggers.append("high posterior uncertainty")
    if triggers:
        action = InteractionPolicy.CLARIFY
    else:
        if delta_star >= CAUTION_DISCREPANCY_THRESHOLD:
            triggers.append("moderate source disagreement")
        if posterior_sigma >= CAUTION_POSTERIOR_SIGMA_THRESHOLD:
            triggers.append("moderate posterior uncertainty")
        action = InteractionPolicy.CAUTIOUS if triggers else InteractionPolicy.RESPOND

    return DimensionPolicyDecision(
        dimension=dimension,
        discrepancy=discrepancy,
        standardized_discrepancy=delta_star,
        posterior_sigma=posterior_sigma,
        action=action,
        triggers=tuple(triggers),
    )


def combine_policy_decisions(
    decisions: Iterable[DimensionPolicyDecision],
) -> PolicyDecision:
    """Combine dimension-level evidence into one conservative interaction action."""
    dimension_decisions = tuple(decisions)
    if not dimension_decisions:
        raise ValueError("At least one dimension decision is required.")

    action = max(
        (decision.action for decision in dimension_decisions),
        key={
            InteractionPolicy.RESPOND: 0,
            InteractionPolicy.CAUTIOUS: 1,
            InteractionPolicy.CLARIFY: 2,
        }.get,
    )
    triggered = tuple(
        f"{decision.dimension}: {trigger}"
        for decision in dimension_decisions
        for trigger in decision.triggers
        if decision.action == action
    )
    gesture_override = {
        InteractionPolicy.RESPOND: None,
        InteractionPolicy.CAUTIOUS: "tilt_head",
        InteractionPolicy.CLARIFY: "think",
    }[action]
    response_guidance = {
        InteractionPolicy.RESPOND: (
            "Continue the conversation normally. Do not state an affect label as fact."
        ),
        InteractionPolicy.CAUTIOUS: (
            "Respond gently and tentatively. Avoid making a strong claim about the "
            "speaker's affect."
        ),
        InteractionPolicy.CLARIFY: (
            "Ask one brief, non-leading clarification question. Explain that the robot "
            "may be reading the situation incorrectly; do not diagnose or label the speaker."
        ),
    }[action]
    return PolicyDecision(
        action=action,
        dimensions=dimension_decisions,
        triggers=triggered,
        gesture_override=gesture_override,
        response_guidance=response_guidance,
    )


def dialogue_context_warmup_policy(
    decisions: Iterable[DimensionPolicyDecision],
    *,
    speaker_turn_count: int,
    warmup_speaker_turns: int = DIALOGUE_CONTEXT_WARMUP_SPEAKER_TURNS,
) -> PolicyDecision:
    """Keep an initially uninformative dialogue prior from triggering caution.

    A dialogue-context prior has no conversational evidence before the opening
    turn.  Its broad uncertainty is therefore expected rather than a reason to
    alter the robot's social response.  The card still shows the model outputs,
    but the clarification thresholds begin only after sufficient history exists.
    """
    if speaker_turn_count < 1:
        raise ValueError("Speaker turn count must be at least one.")
    if warmup_speaker_turns < 1:
        raise ValueError("Warm-up speaker turns must be at least one.")

    dimension_decisions = tuple(decisions)
    if not dimension_decisions:
        raise ValueError("At least one dimension decision is required.")

    return PolicyDecision(
        action=InteractionPolicy.RESPOND,
        dimensions=dimension_decisions,
        triggers=(
            "dialogue-context warm-up: broad prior is expected before enough "
            "conversation history is available",
        ),
        gesture_override=None,
        response_guidance=(
            "Continue normally. The dialogue-context prior is still warming up, "
            "so its broad uncertainty does not trigger a cautious response."
        ),
        context_warmup=True,
        context_turn_count=speaker_turn_count,
    )


def dialogue_context_is_warming_up(speaker_turn_count: int) -> bool:
    """Return whether the current speaker turn is within the warm-up period."""
    if speaker_turn_count < 1:
        raise ValueError("Speaker turn count must be at least one.")
    return speaker_turn_count <= DIALOGUE_CONTEXT_WARMUP_SPEAKER_TURNS
