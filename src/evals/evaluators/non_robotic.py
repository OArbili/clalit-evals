"""NonRobotic: the reply does not read as a script. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator


class NonRoboticEvaluator(LLMJudgeEvaluator):
    name = "NonRobotic"
    prompt_version = "non_robotic/v1"
    definition = (
        "The reply avoids repetitive, canned, overly formal or unnecessarily verbose phrasing that would "
        "read as an obvious script rather than a person."
    )
    fail_criteria = (
        "The reply repeats sentences or phrases already used in earlier assistant replies in the conversation.",
        "The reply is canned: a fixed formula (disclaimer, redirect, sign-off) delivered regardless of what "
        "the user said.",
        "The reply is overly formal or bureaucratic in a way a person speaking to this user would not be.",
        "The reply is unnecessarily long for what it says.",
    )
