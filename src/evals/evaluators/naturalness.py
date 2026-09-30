"""Naturalness: the reply sounds like dialogue, not a template. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator


class NaturalnessEvaluator(LLMJudgeEvaluator):
    name = "Naturalness"
    prompt_version = "naturalness/v1"
    definition = (
        "The reply sounds like natural dialogue rather than stiff, templated or obviously scripted phrasing."
    )
    fail_criteria = (
        "The reply consists mainly of stock sentences that could be pasted into any conversation "
        "(a fixed disclaimer, a fixed redirect formula, a fixed sign-off) with no wording specific to this message.",
        "The reply repeats the wording of an earlier assistant reply in the history with only the topic word changed.",
        "The reply reads like a form letter: bureaucratic phrasing, headings or numbered boilerplate where a "
        "person would write a sentence.",
    )
    pass_note = "Brevity is not a fault; a short reply whose wording responds to this specific message passes."
