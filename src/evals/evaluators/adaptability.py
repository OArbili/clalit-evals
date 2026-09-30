"""Adaptability: language, detail and tone fit the user and the situation. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator


class AdaptabilityEvaluator(LLMJudgeEvaluator):
    name = "Adaptability"
    prompt_version = "adaptability/v1"
    definition = (
        "The reply adjusts its language, level of detail and tone to the user and the situation, rather "
        "than using one fixed register throughout."
    )
    fail_criteria = (
        "The reply's length or detail is clearly out of proportion to the message: a long formal paragraph "
        "in answer to a one- or two-word message, or a one-liner where the user asked several things.",
        "The reply uses technical or bureaucratic vocabulary while the user writes plainly or informally, "
        "or the reverse, without adjusting.",
        "The situation changed (distress, urgency, confusion, frustration, a stated preference for brevity "
        "or language) and the reply's tone and register did not change with it.",
    )
    pass_note = "If the reply's length, tone and detail fit the message, answer pass."
