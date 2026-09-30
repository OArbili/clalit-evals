"""Repetition: the agent repeated the same exact sentence. Deterministic."""

from __future__ import annotations

import re

from .. import vocab
from .base import EvalResult, Evaluator, Turn

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n+")
_TRAILING_PUNCT = re.compile(r"[.!?،,;:\s]+$")
_LIST_MARKER = re.compile(r"^\s*(?:[-*•]|\d+[.)])\s+")


def normalize(sentence: str) -> str:
    """Strip a leading list marker, collapse whitespace, lowercase, strip trailing punctuation."""
    collapsed = " ".join(_LIST_MARKER.sub("", sentence).split())
    return _TRAILING_PUNCT.sub("", collapsed).lower()


def sentences(text: str) -> list[str]:
    return [n for s in _SENTENCE_SPLIT.split(text) if (n := normalize(s))]


class RepetitionEvaluator(Evaluator):
    """Detects the agent repeating the same exact sentence.

    Deterministic -- never calls a model. Sentences end at . ! ? or a line
    break, so repeated lines and bullet items are caught as well as repeated
    prose. Mode is selected by whether `history` is empty: intra-response when
    it is, cross-turn when it isn't. In cross-turn mode only assistant turns are
    compared, since a member echoing themselves is not an agent defect.
    """

    name = "Repetition"

    def evaluate(self, turn: Turn) -> EvalResult:
        if not turn.agent_output:
            return EvalResult.errored(
                self.name, vocab.ERR_MISSING_INPUT, "gen_ai.output.messages is absent or empty."
            )

        current = sentences(turn.agent_output)
        seen: dict[str, str] = {}

        if turn.history:  # cross-turn mode
            for message in turn.history:
                if message.role != "assistant":
                    continue
                for s in sentences(message.text):
                    seen.setdefault(s, "prior")

        for s in current:
            if s in seen:
                where = (
                    "twice in this response"
                    if seen[s] == "current"
                    else "in this response and in an earlier agent turn"
                )
                return EvalResult.failed(self.name, f'The sentence "{s}" occurs {where}.')
            seen[s] = "current"

        mode = "cross-turn" if turn.history else "intra-response"
        count = f"{len(current)} sentence{'' if len(current) == 1 else 's'}"
        return EvalResult.passed(self.name, f"No sentence is repeated ({mode} comparison over {count}).")
