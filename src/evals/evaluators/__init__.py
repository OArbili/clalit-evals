"""Evaluators -- one per file.

    base.py                     the Evaluator contract
    _judge.py                   JudgeBacked (how an evaluator holds its judge) and LLMJudgeEvaluator, the
                                one-prompt judge: one template, schema, call record, error map
    repetition.py               Repetition (deterministic)
    relevance.py                Relevance
    naturalness.py              Naturalness
    empathy.py                  Empathy
    attentiveness.py            Attentiveness
    coherence.py                Coherence
    context_retention.py        ContextRetention (pre-check: no earlier user turns -> pass)
    appropriate_questioning.py  AppropriateQuestioning (pre-check: no question asked -> pass)
    adaptability.py             Adaptability
    non_robotic.py              NonRobotic
    summary_facts.py            SummaryFactCapture, SummaryFactPrecision (reference-based, ratio scores)

To add one: create a module here with an `Evaluator` subclass (a one-prompt
judge: subclass `LLMJudgeEvaluator`; any other use of the judge: `JudgeBacked`
and `ask_judge()`), then either pass an instance to
`Evals(evaluators=[...])` or register a name in BUILTIN below.
Full guide: docs/ADDING_AN_EVALUATOR.md.
"""

from __future__ import annotations

from collections.abc import Callable

from .base import EvalResult
from ..types import Message, Turn
from ._judge import JudgeBacked, LLMJudgeEvaluator
from .adaptability import AdaptabilityEvaluator
from .appropriate_questioning import AppropriateQuestioningEvaluator
from .attentiveness import AttentivenessEvaluator
from .base import Evaluator
from .coherence import CoherenceEvaluator
from .context_retention import ContextRetentionEvaluator
from .empathy import EmpathyEvaluator
from .naturalness import NaturalnessEvaluator
from .non_robotic import NonRoboticEvaluator
from .relevance import RelevanceEvaluator
from .repetition import RepetitionEvaluator
from .summary_facts import SummaryFactCaptureEvaluator, SummaryFactPrecisionEvaluator

__all__ = [
    "BUILTIN", "EvalResult", "Evaluator", "JudgeBacked", "LLMJudgeEvaluator", "Message", "Turn", "build",
    "RepetitionEvaluator", "RelevanceEvaluator", "NaturalnessEvaluator", "EmpathyEvaluator",
    "AttentivenessEvaluator", "CoherenceEvaluator", "ContextRetentionEvaluator",
    "AppropriateQuestioningEvaluator", "AdaptabilityEvaluator", "NonRoboticEvaluator",
    "SummaryFactCaptureEvaluator", "SummaryFactPrecisionEvaluator",
]


def _judge(cls) -> Callable[..., Evaluator]:
    """Factory for an evaluator that consults the judge: built with Evals' shared judge dependencies.
    Anything else (a threshold, capture_content) is set on an instance you build yourself."""
    return lambda *, judge_provider=None, judge_model=None, **_: cls(judge_provider, model=judge_model)


# name -> factory. Factories receive Evals' shared dependencies as keyword
# arguments (judge_provider, judge_model) and must accept **_ so that a
# dependency added later does not break existing entries.
BUILTIN: dict[str, Callable[..., Evaluator]] = {
    "repetition": lambda **_: RepetitionEvaluator(),
    "relevance": _judge(RelevanceEvaluator),
    "naturalness": _judge(NaturalnessEvaluator),
    "empathy": _judge(EmpathyEvaluator),
    "attentiveness": _judge(AttentivenessEvaluator),
    "coherence": _judge(CoherenceEvaluator),
    "context_retention": _judge(ContextRetentionEvaluator),
    "appropriate_questioning": _judge(AppropriateQuestioningEvaluator),
    "adaptability": _judge(AdaptabilityEvaluator),
    "non_robotic": _judge(NonRoboticEvaluator),
    "summary_fact_capture": _judge(SummaryFactCaptureEvaluator),
    "summary_fact_precision": _judge(SummaryFactPrecisionEvaluator),
}


def build(name: str, **deps) -> Evaluator:
    """Instantiate a registered evaluator by name (case-insensitive)."""
    try:
        factory = BUILTIN[name.lower()]
    except KeyError:
        raise ValueError(
            f"unknown evaluator {name!r}; choose from {sorted(BUILTIN)} or pass an Evaluator instance"
        ) from None
    return factory(**deps)
