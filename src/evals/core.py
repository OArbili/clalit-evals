"""The entry point: `Evals`. Evaluates one turn per call and returns Verdicts.

Flow of one call, ev.evaluate(turn):

    1. select     which evaluators run: all of them, or the ones named in `only=`
    2. coerce     the input (a Turn, a dict row, or keyword fields) into a Turn;
                  a row that cannot be understood becomes a skeleton Turn (ids only)
                  plus a preset missing_input result, so the caller still gets
                  attributable error verdicts; strict=True raises instead
    3. run each   _run_one(evaluator, turn), independently:
         a. start the clock
         b. evaluator.evaluate(turn) -> EvalResult (a judge evaluator calls the model
            here and records every call as a JudgeCall on the result)
         c. anything raised -> an error EvalResult with one of the five spec error
            types (_classify); strict=True re-raises
         d. stop the clock, log a WARNING if the result is an error
         e. build the Verdict: the answer, the turn's ids, the evaluator's
            attributes, and the record (started_at, ended_at, calls)
    4. return     Verdicts(one Verdict per evaluator, in configured order, turn=t)

Construction resolves the judge (provider + model) once and builds the
evaluators; that is the whole state. The package holds only the evaluation
logic: it emits no telemetry and stores nothing. Every Verdict carries the
record of how it was produced, and whoever hosts the package (the REST
service, or your own application) reports spans, events and metrics from that
record. Batches are the host's too (evals_service.batch). Logging goes through
the standard `logging` module under the "evals" logger; no handlers installed.
"""

from __future__ import annotations

import json
import logging
import os
import time
from collections.abc import Mapping, Sequence
from dataclasses import replace
from typing import Any

from . import vocab
from .evaluators import EvalResult, Evaluator, build
from .llm import LLMError, Provider
from .providers import as_provider, default_provider
from .types import Turn, Verdict, Verdicts

log = logging.getLogger("evals")
log.addHandler(logging.NullHandler())


class Evals:
    """The one object an end user creates. Evaluates turns.

        ev = Evals()                                   # judge from the environment (ANTHROPIC_API_KEY or OPENAI_*)
        ev = Evals(judge_provider=OpenAICompatibleProvider("http://vllm:8000/v1", default_model="judge"))
        verdicts = ev.evaluate(user_input="...", agent_output="...", response_id="msg_1")

    `judge_provider` is a `Provider` (evals.providers), an Anthropic-style client,
    or a callable.
    Left at None, EVALS_PROVIDER / the API key in the environment decide.
    `judge_model` defaults to EVALS_JUDGE_MODEL, then the provider's default.
    `evaluators` are registry names or Evaluator instances (an instance keeps its
    own options, e.g. SummaryFactCaptureEvaluator(capture_content=True)).
    `strict=True` re-raises instead of turning failures into error verdicts.
    The package evaluates one turn at a time; batching, concurrency and
    aggregates belong to the host (evals_service.batch).

    Tenant, group and every other identifier are fields of the Turn; the
    instance holds no organisational state and no destination of any kind.
    """

    def __init__(
        self,
        *,
        judge_provider: Provider | Any | None = None,
        judge_model: str | None = None,
        evaluators: Sequence[str | Evaluator] = ("repetition", "relevance"),
        strict: bool = False,
    ) -> None:
        """Resolve the judge once (argument, else the environment; model argument, else EVALS_JUDGE_MODEL,
        else the provider\'s default), build the evaluators (registry names through their factory, instances as
        they are), bind the judge into every evaluator, log one line. That is the whole state."""
        env = os.environ.get
        self._strict = strict
        judge_prov = as_provider(judge_provider) or default_provider()
        judge = judge_model or env("EVALS_JUDGE_MODEL") or judge_prov.default_model
        if not judge:
            raise ValueError(f"judge_model= is required: provider {judge_prov.name!r} declares no default model")
        self.judge_provider, self.judge_model = judge_prov, judge

        self._evaluators: list[Evaluator] = []
        for e in evaluators:
            if isinstance(e, Evaluator):
                self._evaluators.append(e)
            elif isinstance(e, str):
                self._evaluators.append(build(e, judge_provider=judge_prov, judge_model=judge))
            else:
                raise TypeError(f"evaluators must be names or Evaluator instances, got {type(e).__name__}")
        for e in self._evaluators:   # built or passed in: fill in whatever they were not given explicitly
            e.bind(judge_provider=judge_prov, judge_model=judge)
        log.info("Evals ready: evaluators=%s judge=%s/%s", [e.name for e in self._evaluators], judge_prov.name, judge)

    @property
    def evaluators(self) -> list[Evaluator]:
        """A copy of the configured evaluators, for inspection."""
        return list(self._evaluators)

    def evaluate(
        self,
        turn: Turn | Mapping[str, Any] | None = None,
        /,
        *,
        only: str | Sequence[str] | None = None,
        **fields: Any,
    ) -> Verdicts:
        """The only verb: run every configured evaluator (or `only=` some) on one turn.

        `turn` is a Turn, a dict/row with Turn's field names, or nothing (then
        the keyword fields are the turn). Select, coerce, run each evaluator
        through _run_one, return Verdicts with the Turn attached. Never raises
        unless strict=True: a row that cannot be understood, or an evaluator that
        fails, becomes verdicts with error_type set and no score.
        """
        # 1. select: all configured evaluators, or the ones named in `only=`
        selected = self._select(only)

        # 2. coerce: whatever came in becomes a Turn; a row that cannot be read
        #    becomes a skeleton Turn (its ids) plus a preset missing_input result
        raw = turn if turn is not None else {}
        try:
            t = Turn.coerce(raw, **fields)
        except (TypeError, ValueError) as exc:
            if self._strict:
                raise
            log.warning("row could not be evaluated: %s", exc)
            t = _skeleton(raw, fields)
            preset = EvalResult.errored("", vocab.ERR_MISSING_INPUT, f"{type(exc).__name__}: {exc}")
        else:
            preset = None

        # 3. run each evaluator in isolation (see _run_one), 4. return them together
        verdicts = Verdicts(
            [self._run_one(e, t, preset=preset and replace(preset, name=e.name)) for e in selected],
            turn=t,
        )
        log.debug("evaluated response_id=%s: %s", t.response_id, ", ".join(f"{v.name}={v.status}" for v in verdicts))
        return verdicts

    # -- internals ----------------------------------------------------------------

    def _select(self, only) -> list[Evaluator]:
        """`only` -> the evaluators to run: all when None, else the named ones (case-insensitive).
        An unconfigured name is a ValueError even without strict: a caller mistake, not a data problem."""
        if only is None:
            return list(self._evaluators)
        names = [only] if isinstance(only, str) else list(only)
        by_name = {e.name.lower(): e for e in self._evaluators}
        out = []
        for n in names:
            if n.lower() not in by_name:
                raise ValueError(
                    f"evaluator {n!r} is not configured; have {[e.name for e in self._evaluators]}"
                )
            out.append(by_name[n.lower()])
        return out

    def _run_one(self, evaluator: Evaluator, turn: Turn, *, preset: EvalResult | None = None) -> Verdict:
        """One evaluator, isolated and timed: call it (or use `preset`, for a bad row), turn anything raised
        into an error result with a spec error type, log an error verdict, and build the Verdict with the
        answer, the turn's ids, the attributes and the record. A failure here never affects another evaluator."""
        started = time.time()                                                    # a. start the clock
        try:
            r = preset if preset is not None else evaluator.evaluate(turn)       # b. the evaluator (may call the judge)
            if not isinstance(r, EvalResult):
                raise TypeError(f"{evaluator.name}.evaluate returned {type(r).__name__}, expected EvalResult")
        except Exception as exc:                                                 # c. any failure -> an error result
            if self._strict:
                raise
            kind = _classify(exc)
            log.exception("%s raised; recorded as %s", evaluator.name, kind)
            r = EvalResult.errored(evaluator.name, kind, f"{type(exc).__name__}: {exc}")
        ended = time.time()                                                      # d. stop the clock
        if r.is_error:
            log.warning("%s on response_id=%s: error %s (%s)", r.name, turn.response_id, r.error_type, r.explanation)
        linked = bool(turn.trace_id and turn.span_id)
        return Verdict(                                                          # e. the answer + the record
            r.name, r.label, r.score, r.explanation, r.error_type,
            response_id=turn.response_id,
            trace_id=turn.trace_id if linked else None,
            span_id=turn.span_id if linked else None,
            attributes=dict(r.attributes),
            started_at=started, ended_at=ended, calls=tuple(r.calls),
        )


def _skeleton(raw: Any, fields: Mapping[str, Any]) -> Turn:
    """For a row that failed coercion: a Turn with an empty reply and whatever ids the row and the keyword
    fields carried, so the error verdicts can still be written back against the right response and trace."""
    src: dict[str, Any] = {}
    if isinstance(raw, Turn):
        src = {"response_id": raw.response_id, "conversation_id": raw.conversation_id,
               "trace_id": raw.trace_id, "span_id": raw.span_id}
    elif isinstance(raw, Mapping):
        src = dict(raw)
    src.update(fields)
    keep = {}
    for k in ("response_id", "conversation_id", "trace_id", "span_id"):
        v = src.get(k)
        if isinstance(v, str):
            keep[k] = v
    return Turn(agent_output="", **keep)


def _classify(exc: BaseException) -> str:
    """Exception -> spec error type: an LLMError keeps its own (timeout, rate_limit, provider_error);
    JSON, key, type and value errors are invalid_response; anything else is provider_error."""
    if isinstance(exc, LLMError):
        return exc.error_type
    if isinstance(exc, (json.JSONDecodeError, KeyError, TypeError, ValueError)):
        return vocab.ERR_INVALID_RESPONSE
    return vocab.ERR_PROVIDER
