"""The data types the package speaks. Plain dataclasses, standard library only.

What goes in
    Turn        one agent turn: the reply under evaluation (agent_output), the member's
                message, earlier messages, an optional human reference, and the identifiers
                that pass through untouched (response, conversation, the agent's trace and
                span ids, model, tenant, group). Turn.coerce() accepts a Turn or any dict with
                these field names, ignores extra keys, reads None / NaN / pandas missing values
                as absent, and rejects non-text content with a clear error.
    Message     one earlier message (role, text); as_message() accepts the common spellings.

What comes out
    JudgeCall   the record of one model call an evaluator made: provider, model, prompt
                version, timing, tokens, finish reason, and the error if it failed.
    Verdict     one evaluator's answer for one turn: label, score and explanation, or an
                error type instead of a score; the turn's ids; the evaluator's extra
                attributes; and the record (started_at, ended_at, calls). to_dict() is the
                serialisable form.
    Verdicts    all verdicts for one turn, indexable by name, with ok / failed / errors and a
                printable table.

The package knows nothing about where turns come from or where verdicts go. Whatever a
host needs to report or store must be a field here, because these objects are all the
package hands back; batches (Report) live in the service, evals_service.batch.
"""

from __future__ import annotations

import math
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import asdict, dataclass, field, fields
from html import escape
from typing import Any


@dataclass(frozen=True)
class Message:
    """One prior message. `text` is plain text; content blocks are flattened before evaluation."""

    role: str
    text: str


HistoryItem = Message | tuple[str, str] | Mapping[str, str]

_MISSING_TYPES = {"NAType", "NaTType"}  # pandas.NA / pandas.NaT without importing pandas


def _clean(value: Any) -> Any:
    """Normalise the ways a missing value arrives from a store or a DataFrame to None."""
    if value is None:
        return None
    if isinstance(value, float) and math.isnan(value):
        return None
    if type(value).__name__ in _MISSING_TYPES:
        return None
    return value


def as_message(item: HistoryItem) -> Message:
    """Message | (role, text) | {"role": ..., "content" | "text": ...} -> Message."""
    if isinstance(item, Message):
        return item
    if isinstance(item, Mapping):
        if "role" not in item:
            raise ValueError(f"history item {item!r} has no 'role'")
        text = item.get("content", item.get("text"))
        if text is None:
            raise ValueError(f"history item {item!r} has neither 'content' nor 'text'")
        if not isinstance(text, str):
            raise TypeError(
                "history content must be plain text; flatten content-block lists "
                "(tool results, images) before evaluation"
            )
        return Message(str(item["role"]), text)
    if isinstance(item, (tuple, list)) and len(item) == 2 and not isinstance(item[0], Mapping):
        return Message(str(item[0]), str(item[1]))
    if isinstance(item, str):
        raise TypeError("history must be a list of messages, not a single string")
    raise TypeError(f"cannot interpret {item!r} as a message")


@dataclass
class Turn:
    """One agent turn, as the agent application produced it or as it was read back from storage.

    Every id is a plain string. `trace_id` (32 hex) / `span_id` (16 hex) are the
    evaluated GenAI span's ids as your trace store keeps them: when both are present
    the gen_ai.evaluation.result event is parented to that span; otherwise it is
    unparented and `response_id` (gen_ai.response.id) is the only link. `sampled`
    is whether that span was sampled -- live mode fills it, stored rows default to
    True. `model` is the AGENT's model, a metric dimension only. `group` is the
    organisational unit that owns the agent (service.namespace), a reporting
    dimension like `tenant_id`; both are inputs carried through unchanged to the
    span, the event and the metrics -- Evals holds neither and no evaluator reads them.
    `finish_reason`
    is normalised to stop | length | tool_calls | content_filter | error; a value
    outside that set passes through unchanged rather than being lost. `reference`
    is a human-written reference for `agent_output` (the ground-truth summary
    for the summary fact evaluators); it comes from an annotation store, not a
    span, and only reference-based evaluators read it.
    """

    agent_output: str
    user_input: str | None = None
    history: list[Message] = field(default_factory=list)
    reference: str | None = None
    response_id: str | None = None
    conversation_id: str | None = None
    model: str | None = None
    tenant_id: str | None = None
    group: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    finish_reason: str | None = None
    sampled: bool = True

    _TEXT_FIELDS = (
        "agent_output", "user_input", "reference", "response_id", "conversation_id",
        "model", "tenant_id", "group", "trace_id", "span_id", "finish_reason",
    )

    def __post_init__(self) -> None:
        for name in self._TEXT_FIELDS:
            value = _clean(getattr(self, name))
            if value is not None and not isinstance(value, str):
                hint = " -- flatten content blocks to text first" if isinstance(value, list) else ""
                raise TypeError(f"{name} must be plain text, got {type(value).__name__}{hint}")
            setattr(self, name, value)
        if self.agent_output is None:
            self.agent_output = ""
        history = _clean(self.history)
        if isinstance(history, str):
            raise TypeError("history must be a list of messages, not a single string")
        self.history = [as_message(m) for m in (history or ())]
        self.sampled = bool(self.sampled)

    @property
    def text(self) -> str:
        """The agent's reply -- alias of `agent_output` for live-mode readability."""
        return self.agent_output

    @classmethod
    def coerce(cls, value: Turn | Mapping[str, Any], **overrides: Any) -> Turn:
        """A Turn passes through; a dict/row with Turn's field names becomes one.

        Unknown keys are ignored so `df.to_dict("records")` / JSONL rows pass
        straight in; NaN / pandas.NA / None all read as missing. A missing
        agent_output becomes "" and surfaces as a missing_input verdict.
        Explicit overrides win. Raises TypeError / ValueError for a row that
        cannot be understood (Evals turns that into error verdicts).
        """
        names = {f.name for f in fields(cls)}
        if isinstance(value, Turn):
            if not overrides:
                return value
            data = {n: getattr(value, n) for n in names}
        elif isinstance(value, Mapping):
            data = {k: v for k, v in value.items() if k in names}
        else:
            raise TypeError(f"expected a Turn or a mapping, got {type(value).__name__}")
        data.update(overrides)
        data.setdefault("agent_output", "")
        return cls(**data)


@dataclass(frozen=True)
class JudgeCall:
    """One model call an evaluator made: what the host needs to report it (a span, a cost line, a log)."""

    provider: str
    model: str
    prompt_version: str
    started_at: float                 # epoch seconds
    ended_at: float
    response_model: str | None = None
    response_id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None
    error_type: str | None = None     # one of the five spec error types when the call failed
    error: str | None = None

    @property
    def duration_s(self) -> float:
        return self.ended_at - self.started_at


@dataclass(frozen=True)
class Verdict:
    """One evaluator's answer for one turn, plus the record of how it was produced.

    `label` and `score` are None exactly when `error_type` is set. `attributes`
    holds the evaluator's extra attributes (fact counts for the summary
    evaluators); empty for the others. `started_at` / `ended_at` bound the
    evaluator's run and `calls` lists its model calls, so whoever hosts the
    package (the service) can report spans, events, metrics or costs from the
    Verdict alone; the package itself emits nothing.
    """

    name: str
    label: str | None
    score: float | None
    explanation: str | None
    error_type: str | None
    response_id: str | None = None
    trace_id: str | None = None
    span_id: str | None = None
    attributes: Mapping[str, Any] = field(default_factory=dict)
    started_at: float | None = None
    ended_at: float | None = None
    calls: tuple[JudgeCall, ...] = ()

    @property
    def passed(self) -> bool | None:
        """label == "pass" -- the derivation the spec mandates; None on error."""
        return None if self.error_type else self.label == "pass"

    @property
    def status(self) -> str:
        """'pass' | 'fail' | 'error:<type>' -- for printing."""
        return f"error:{self.error_type}" if self.error_type else (self.label or "?")

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "score": self.score,
            "passed": self.passed,
            "explanation": self.explanation,
            "error_type": self.error_type,
            "attributes": dict(self.attributes),
            "started_at": self.started_at,
            "ended_at": self.ended_at,
            "calls": [asdict(c) for c in self.calls],
        }


class Verdicts(Sequence[Verdict]):
    """All verdicts for one turn, plus the turn itself. What `evaluate()` returns.

    One `Verdict` per evaluator, in the configured order, never modified after
    construction; `.turn` is the Turn the evaluators saw. Index by position
    (`verdicts[0]`) or by evaluator name, case-insensitively (`verdicts["relevance"]`;
    an unknown name raises KeyError listing the names present). `ok` is true only
    when every evaluator passed: an error is not a pass. `failed` (the evaluator
    said fail) and `errors` (it could not answer) are kept apart because they need
    different follow-up and the pass-rate metric excludes errors. `to_dict()` is the
    write-back shape: the turn's ids on top, then each Verdict.to_dict() with its
    judge calls; the service returns exactly this as JSON. `repr()` prints a table,
    one row per evaluator; `_repr_html_` does the same in a notebook. Aggregates
    across turns are not here: that is the service's Report.
    """

    def __init__(self, verdicts: Sequence[Verdict], *, turn: Turn) -> None:
        self._items = list(verdicts)
        self._turn = turn

    def __len__(self) -> int:
        return len(self._items)

    def __iter__(self) -> Iterator[Verdict]:
        return iter(self._items)

    def __getitem__(self, key):  # type: ignore[override]
        if isinstance(key, str):
            for v in self._items:
                if v.name.lower() == key.lower():
                    return v
            raise KeyError(f"no verdict named {key!r}; have {[v.name for v in self._items]}")
        return self._items[key]

    @property
    def turn(self) -> Turn:
        return self._turn

    @property
    def ok(self) -> bool:
        """True iff every evaluator passed. An errored evaluator is not a pass."""
        return all(v.passed is True for v in self._items)

    @property
    def failed(self) -> list[Verdict]:
        """Verdicts with passed is False -- errors excluded, as evals.score excludes them."""
        return [v for v in self._items if v.passed is False]

    @property
    def errors(self) -> list[Verdict]:
        return [v for v in self._items if v.error_type]

    def to_dict(self) -> dict[str, Any]:
        """Write-back shape, keyed by the ids the turn store already has."""
        t = self._turn
        return {
            "response_id": t.response_id,
            "trace_id": t.trace_id,
            "span_id": t.span_id,
            "conversation_id": t.conversation_id,
            "verdicts": [v.to_dict() for v in self._items],
        }

    def __repr__(self) -> str:
        if not self._items:
            return "Verdicts()"
        w = max(len(v.name) for v in self._items)
        return "\n".join(
            f"{v.name:<{w}}  {v.status:<20}  {v.explanation or ''}" for v in self._items
        )

    def _repr_html_(self) -> str:
        rows = "".join(
            f"<tr><td>{escape(v.name)}</td>"
            f"<td style='background:{_colour(v)}'>{escape(v.status)}</td>"
            f"<td>{escape(v.explanation or '')}</td></tr>"
            for v in self._items
        )
        return f"<table><tr><th>evaluator</th><th>status</th><th>explanation</th></tr>{rows}</table>"


def _colour(v: Verdict) -> str:
    if v.error_type:
        return "#e0e0e0"
    return "#c8f0c8" if v.passed else "#f5c6c6"
