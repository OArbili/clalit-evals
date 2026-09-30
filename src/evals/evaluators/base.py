"""The evaluator contract. Subclass `Evaluator`; see docs/ADDING_AN_EVALUATOR.md."""

from __future__ import annotations

import abc
from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from .. import vocab
from ..types import JudgeCall, Turn

__all__ = ["Evaluator", "EvalResult", "Turn"]


@dataclass(frozen=True)
class EvalResult:
    """One evaluator's verdict on one turn.

    `score` and `label` are None exactly when `error_type` is set (on the wire
    the score attributes are then omitted, never null). `attributes` are extra
    attributes an evaluator wants recorded (the summary evaluators' fact
    counts); they never override the standard ones. `calls` are the model calls
    the evaluator made, for the host to report.
    """

    name: str
    score: float | None = None
    label: str | None = None
    explanation: str | None = None
    error_type: str | None = None
    attributes: Mapping[str, object] = field(default_factory=dict)
    calls: tuple[JudgeCall, ...] = ()

    @classmethod
    def passed(cls, name: str, explanation: str) -> "EvalResult":
        return cls(name, score=1.0, label=vocab.LABEL_PASS, explanation=explanation)

    @classmethod
    def failed(cls, name: str, explanation: str) -> "EvalResult":
        return cls(name, score=0.0, label=vocab.LABEL_FAIL, explanation=explanation)

    @classmethod
    def scored(
        cls, name: str, score: float, *, passed: bool, explanation: str,
        attributes: Mapping[str, object] | None = None,
    ) -> "EvalResult":
        """A ratio score with a label decided by the caller (its threshold)."""
        label = vocab.LABEL_PASS if passed else vocab.LABEL_FAIL
        return cls(name, score=float(score), label=label, explanation=explanation, attributes=dict(attributes or {}))

    @classmethod
    def errored(cls, name: str, error_type: str, detail: str) -> "EvalResult":
        return cls(name, error_type=error_type, explanation=detail)

    def with_calls(self, calls) -> "EvalResult":
        return replace(self, calls=tuple(calls))

    @property
    def is_error(self) -> bool:
        return self.error_type is not None


class Evaluator(abc.ABC):
    """One check on one agent turn.

    Set `name` -- it becomes `gen_ai.evaluation.name` wherever the verdict is
    reported -- and implement `evaluate`, returning one of `EvalResult.passed`,
    `.failed`, `.scored` or `.errored`. Expected failures (missing input, a judge
    timing out) are returned, not raised; anything you do raise is caught by the
    core.py and recorded as an errored verdict.
    """

    name: str

    @abc.abstractmethod
    def evaluate(self, turn: Turn) -> EvalResult: ...

    def bind(self, *, judge_provider=None, judge_model=None) -> None:
        """Receive the Evals instance's shared judge dependencies.

        Called by Evals on every evaluator, built or passed in. The default
        does nothing; judge evaluators fill in whatever they were not given
        explicitly, so a custom judge works with just `Evals(evaluators=[MyJudge()])`.
        """
