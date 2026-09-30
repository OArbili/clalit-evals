"""Relevance: at least part of the reply relates to the user's message. LLM-as-judge."""

from __future__ import annotations

from ._judge import LLMJudgeEvaluator

PROMPT_VERSION = "relevance/v2"


class RelevanceEvaluator(LLMJudgeEvaluator):
    """Checks that at least part of the agent answer relates to the user input.

    Deliberately lenient: this detects non-sequiturs, not partial or weak answers.
    History is shown to the judge only to resolve references in the last message.
    """

    name = "Relevance"
    prompt_version = PROMPT_VERSION
    definition = "At least part of the assistant's reply relates to what the user just said."
    fail_criteria = (
        "The reply is entirely unrelated to the user's message: a non-sequitur about a "
        "different subject, with no sentence that addresses what the user said.",
    )
    pass_note = (
        "A reply that is incomplete, partial, generic or only partly on topic still passes: "
        "this dimension detects non-sequiturs, not quality. Use the conversation history only "
        "to resolve references in the user's last message (for example \"yes, book it\")."
    )
