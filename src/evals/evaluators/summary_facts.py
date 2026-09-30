"""SummaryFactCapture and SummaryFactPrecision: an LLM summary against a human reference, fact by fact.

Specification: docs/SUMMARY_FACT_CAPTURE.md. Both evaluators run the same two
judge prompts. Capture extracts facts from the reference and checks them against
the LLM summary (recall: did it miss anything?). Precision extracts claims from
the LLM summary and checks them against the reference (did it invent anything?).
The reference's facts are cached per instance so every summary of one
conversation is measured against the same list.
"""

from __future__ import annotations

import hashlib
import threading


from .. import vocab
from ..llm import Provider
from ..types import JudgeCall
from ._judge import JudgeBacked, judge_error
from .base import EvalResult, Turn

PROMPT_VERSION_EXTRACT = "summary_facts/extract/v1"
PROMPT_VERSION_VERIFY = "summary_facts/verify/v1"

EXTRACT_SCHEMA = {
    "type": "object",
    "properties": {
        "statements": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Every checkable statement in the text, atomic and self-contained, in order.",
        }
    },
    "required": ["statements"],
    "additionalProperties": False,
}

VERIFY_SCHEMA = {
    "type": "object",
    "properties": {
        "results": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "statement": {"type": "integer", "description": "The statement's number in the list."},
                    "stated": {"type": "boolean", "description": "Whether the target text states it."},
                    "evidence": {"type": "string", "description": "The target's words that state it, or empty."},
                },
                "required": ["statement", "stated", "evidence"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["results"],
    "additionalProperties": False,
}

EXTRACT_PROMPT = """\
You extract atomic statements from ONE text: a summary of a health fund's triage conversation.

Return every checkable statement the text makes, as a list, in order of appearance.
- One statement per fact: a symptom, its onset or duration, a measurement, a negation \
("not pregnant"), an action taken, a recommendation given. A sentence with two facts yields two statements.
- Each statement is self-contained: resolve pronouns and references ("she" -> "the member").
- Keep numbers, dates, units and negations exactly as written.
- Keep the text's own language (Hebrew or English); do not translate.
- Do not add, infer, merge or judge anything. Identical texts must yield identical lists.
- A text with no checkable statement yields an empty list.
"""

VERIFY_PROMPT = """\
You are a strict, consistent checker. You receive a TARGET text and a numbered list of STATEMENTS. \
For each statement decide whether the target text states it.

A statement is stated by the target only if the target says the same thing, or an equivalent \
paraphrase of it, with the same numbers, dates, units and polarity:
- "sore throat" and "throat pain" are the same statement; "38.5" and "38" are not; "no fever" and "fever" are not.
- A statement the target does not mention at all is not stated.
- A statement the target contradicts is not stated.
- Do not use outside knowledge and do not infer what the target probably meant.
- Judge each statement on its own. Identical inputs must yield identical answers.

Return one result per statement, in the same order, with the statement's number, `stated` true or \
false, and `evidence`: the target's words that state it, quoted as plain text, or an empty string \
when not stated. The texts may be Hebrew or English.
"""


class _SummaryFactsEvaluator(JudgeBacked):
    """Shared machinery: extract statements from the source text, verify each against the target,
    score the ratio. Subclasses set the direction, the wording and the default threshold."""

    name = ""
    default_threshold: float = 1.0
    source_field = ""        # the text statements are extracted from
    target_field = ""        # the text they are checked against
    unit = ""                # "reference facts" | "claims"
    total_attr = stated_attr = not_stated_attr = ""

    def __init__(
        self,
        provider: Provider | object | None = None,
        model: str | None = None,
        effort: str = "low",
        timeout: float = 30.0,
        *,
        threshold: float | None = None,
        capture_content: bool = False,
    ) -> None:
        super().__init__(provider, model, effort, timeout)
        self.threshold = self.default_threshold if threshold is None else float(threshold)
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError(f"threshold must be within [0, 1], got {self.threshold}")
        self._capture_content = capture_content
        self._cache: dict[str, list[str]] = {}
        self._lock = threading.Lock()

    # -- the shared path -------------------------------------------------------------

    def evaluate(self, turn: Turn) -> EvalResult:
        if not turn.agent_output:
            return EvalResult.errored(self.name, vocab.ERR_MISSING_INPUT,
                                      "the LLM summary (gen_ai.output.messages) is absent or empty.")
        if not turn.reference:
            return EvalResult.errored(self.name, vocab.ERR_MISSING_INPUT, "the reference summary is absent.")

        source = getattr(turn, self.source_field)
        target = getattr(turn, self.target_field)
        calls: list[JudgeCall] = []
        try:
            statements = self._extract(source, calls)
            if not statements:
                return EvalResult.errored(self.name, vocab.ERR_MISSING_INPUT, self._nothing_to_check).with_calls(calls)
            stated = self._verify(target, statements, calls)
        except Exception as exc:
            err = judge_error(self.name, exc)
            if err is None:
                raise
            return err.with_calls(calls)

        kept = [s for s, ok in zip(statements, stated) if ok]
        lost = [s for s, ok in zip(statements, stated) if not ok]
        ratio = round(len(kept) / len(statements), 4)
        attributes: dict[str, object] = {self.total_attr: len(statements), self.stated_attr: len(kept)}
        if self._capture_content:
            attributes[self.not_stated_attr] = lost
        return EvalResult.scored(
            self.name, ratio, passed=ratio >= self.threshold,
            explanation=self._explain(len(kept), len(statements), ratio, lost), attributes=attributes,
        ).with_calls(calls)

    def _extract(self, text: str, calls: list[JudgeCall] | None = None) -> list[str]:
        calls = [] if calls is None else calls
        cache = self.source_field == "reference"   # the reference is shared by every summary of a conversation
        key = hashlib.sha256(text.encode("utf-8")).hexdigest() if cache else None
        if cache:
            with self._lock:
                hit = self._cache.get(key)
            if hit is not None:
                return list(hit)
        reply = self.ask_judge(
            system=EXTRACT_PROMPT, user=f"<text>\n{text}\n</text>", schema=EXTRACT_SCHEMA,
            prompt_version=PROMPT_VERSION_EXTRACT, calls=calls, max_tokens=4096,
        )
        statements = reply["statements"]
        if not isinstance(statements, list) or not all(isinstance(s, str) for s in statements):
            raise ValueError("extraction reply is not a list of strings")
        statements = [s.strip() for s in statements if s.strip()]
        if cache:
            with self._lock:
                self._cache[key] = list(statements)
        return statements

    def _verify(self, target: str, statements: list[str], calls: list[JudgeCall] | None = None) -> list[bool]:
        calls = [] if calls is None else calls
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(statements, 1))
        reply = self.ask_judge(
            system=VERIFY_PROMPT,
            user=f"<target>\n{target}\n</target>\n\n<statements>\n{listing}\n</statements>",
            schema=VERIFY_SCHEMA, prompt_version=PROMPT_VERSION_VERIFY, calls=calls, max_tokens=4096,
        )
        results = reply["results"]
        if not isinstance(results, list):
            raise ValueError("verification reply is not a list")
        by_number: dict[int, bool] = {}
        for r in results:
            n = r["statement"]
            if not isinstance(n, int) or isinstance(n, bool) or not 1 <= n <= len(statements) or n in by_number:
                raise ValueError(f"verification reply has an invalid or duplicate statement number {n!r}")
            if not isinstance(r["stated"], bool):
                raise ValueError(f"verification reply has a non-boolean verdict for statement {n}")
            by_number[n] = r["stated"]
        if len(by_number) != len(statements):
            raise ValueError(f"verification reply has {len(by_number)} results for {len(statements)} statements")
        return [by_number[i] for i in range(1, len(statements) + 1)]

    # -- wording -----------------------------------------------------------------------

    _nothing_to_check = ""

    def _explain(self, kept: int, total: int, ratio: float, lost: list[str]) -> str:
        raise NotImplementedError


class SummaryFactCaptureEvaluator(_SummaryFactsEvaluator):
    """Recall: the share of the reference's facts that the LLM summary states."""

    name = "SummaryFactCapture"
    default_threshold = 0.8
    source_field, target_field = "reference", "agent_output"
    unit = "reference facts"
    total_attr, stated_attr, not_stated_attr = vocab.FACTS_TOTAL, vocab.FACTS_CAPTURED, vocab.FACTS_MISSING
    _nothing_to_check = "the reference contains no checkable facts."

    def _explain(self, kept, total, ratio, lost):
        if not lost:
            return f"All {total} reference facts are captured."
        return (f"{kept} of {total} reference facts captured ({ratio:.2f}). Missing: "
                + "; ".join(f"'{s}'" for s in lost) + ".")


class SummaryFactPrecisionEvaluator(_SummaryFactsEvaluator):
    """Precision: the share of the LLM summary's claims that the reference supports."""

    name = "SummaryFactPrecision"
    default_threshold = 1.0
    source_field, target_field = "agent_output", "reference"
    unit = "claims"
    total_attr, stated_attr, not_stated_attr = vocab.CLAIMS_TOTAL, vocab.CLAIMS_SUPPORTED, vocab.CLAIMS_UNSUPPORTED
    _nothing_to_check = "the LLM summary contains no checkable claims."

    def _explain(self, kept, total, ratio, lost):
        if not lost:
            return f"All {total} claims are supported by the reference."
        return (f"{kept} of {total} claims are supported by the reference ({ratio:.2f}). Unsupported: "
                + "; ".join(f"'{s}'" for s in lost) + ".")
