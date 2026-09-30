"""Coherence: the reply follows from the preceding turns. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator


class CoherenceEvaluator(LLMJudgeEvaluator):
    name = "Coherence"
    prompt_version = "coherence/v1"
    definition = (
        "The reply logically follows from the preceding turns of the conversation, without abrupt topic "
        "jumps or non-sequiturs."
    )
    fail_criteria = (
        "The reply changes subject abruptly relative to the user's last message and the turns before it.",
        "The reply answers an earlier or different question than the one the conversation has reached.",
        "The reply contradicts something established in the immediately preceding turns.",
        "No sentence of the reply follows from what was just said.",
    )
    pass_note = "With no earlier turns, judge whether the reply follows from the user's message alone."
