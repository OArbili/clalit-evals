"""Empathy: expressed emotion or concern is recognised and answered. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator


class EmpathyEvaluator(LLMJudgeEvaluator):
    name = "Empathy"
    prompt_version = "empathy/v1"
    definition = (
        "When the user expresses an emotion, worry, pain or distress, the reply recognises it specifically "
        "and responds to it, rather than ignoring it or acknowledging it flatly."
    )
    fail_criteria = (
        "The user expressed an emotion, worry, fear, pain, distress or frustration, and the reply does not "
        "mention or respond to it at all.",
        "The user expressed such a feeling and the reply acknowledges it only with a formula that does not "
        "reflect what was actually said (for example a generic \"I understand\" followed by an unchanged "
        "standard redirect).",
    )
    pass_note = "If the user expressed no emotion or concern, there is nothing to recognise: answer pass."
