"""FocusedQuestioning: the questions in one reply are about one subject. LLM-as-judge with a pre-check."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator
from .base import EvalResult, Turn


class FocusedQuestioningEvaluator(LLMJudgeEvaluator):
    """Did the agent ask several unrelated questions in one iteration? Fails a reply that does."""

    name = "FocusedQuestioning"
    prompt_version = "focused_questioning/v1"
    definition = "When the reply asks more than one question, the questions are about the same subject."
    fail_criteria = (
        "The reply asks two or more questions about subjects unrelated to each other in the same message (for example "
        "a question about a symptom together with a question about scheduling, contact details, payment or a "
        "different complaint).",
    )
    pass_note = (
        "Several questions about the same symptom, complaint or request pass. A reply with one question or none passes."
    )

    def precheck(self, turn: Turn) -> EvalResult | None:
        if "?" not in turn.agent_output:
            return EvalResult.passed(self.name, "The reply asks no question, so there is nothing to compare.")
        return None
