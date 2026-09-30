"""ContextRetention: earlier information is remembered and used. LLM-as-judge with a pre-check."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator
from .base import EvalResult, Turn


class ContextRetentionEvaluator(LLMJudgeEvaluator):
    name = "ContextRetention"
    prompt_version = "context_retention/v1"
    definition = (
        "The reply remembers and uses information the user provided earlier in the conversation, rather "
        "than re-asking for it or contradicting it."
    )
    fail_criteria = (
        "The reply asks for information the user already provided earlier in the conversation (for example "
        "asking about pregnancy after the user said they are not pregnant, or asking the reason for the "
        "visit after it was stated).",
        "The reply contradicts information the user provided earlier in the conversation.",
        "The reply ignores earlier information where it was clearly needed to respond correctly.",
    )

    def precheck(self, turn: Turn) -> EvalResult | None:
        if not any(m.role == "user" for m in turn.history):
            return EvalResult.passed(self.name, "No earlier user messages, so there is no information to retain.")
        return None
