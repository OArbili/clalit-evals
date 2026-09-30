"""AppropriateQuestioning: follow-up questions are relevant and proportionate. LLM-as-judge with a pre-check."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator
from .base import EvalResult, Turn


class AppropriateQuestioningEvaluator(LLMJudgeEvaluator):
    name = "AppropriateQuestioning"
    prompt_version = "appropriate_questioning/v1"
    definition = (
        "When the reply asks follow-up questions, they are relevant and proportionate to the situation, "
        "rather than mechanical or interrogation-like."
    )
    fail_criteria = (
        "A question in the reply is unrelated to the problem the user described.",
        "A question repeats one the user already answered in the conversation.",
        "The reply is a mechanical checklist: several questions at once, or a list of symptoms to tick, "
        "where one targeted question would do.",
        "The questions interrogate rather than help: they demand details without any acknowledgement of "
        "what the user said.",
    )

    def precheck(self, turn: Turn) -> EvalResult | None:
        if "?" not in turn.agent_output:
            return EvalResult.passed(self.name, "The reply asks no follow-up question, so there is nothing to judge.")
        return None
