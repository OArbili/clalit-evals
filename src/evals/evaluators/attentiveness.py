"""Attentiveness: the reply responds to the specifics of the message. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator


class AttentivenessEvaluator(LLMJudgeEvaluator):
    name = "Attentiveness"
    prompt_version = "attentiveness/v1"
    definition = (
        "The reply shows the assistant understood what the user actually said, responding to the specifics "
        "of the message rather than a generic version of it."
    )
    fail_criteria = (
        "The reply could have been written without reading the message: it addresses a generic version of "
        "the request and ignores a specific detail the user gave (a named specialist or service, an age, a "
        "duration, a symptom, a correction, a stated preference).",
        "The reply answers a different question from the one asked.",
        "The user corrected or narrowed something and the reply proceeds as if they had not.",
    )
    pass_note = "A reply that reflects the user's specifics passes even if it then redirects or declines."
