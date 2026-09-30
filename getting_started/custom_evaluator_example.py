"""A custom evaluator, written exactly as docs/ADDING_AN_EVALUATOR.md describes."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run from a clone without pip install -e .

from evals import Evals
from evals import vocab
from evals.evaluators.base import EvalResult, Evaluator, Turn
from stub_client import StubClient


class AnswerLengthEvaluator(Evaluator):
    """Fails when the answer exceeds `max_words` words."""

    name = "AnswerLength"

    def __init__(self, max_words: int = 80) -> None:
        self.max_words = max_words

    def evaluate(self, turn: Turn) -> EvalResult:
        if not turn.agent_output:
            return EvalResult.errored(self.name, vocab.ERR_MISSING_INPUT, "gen_ai.output.messages is absent or empty.")
        n = len(turn.agent_output.split())
        if n > self.max_words:
            return EvalResult.failed(self.name, f"The answer is {n} words; the limit is {self.max_words}.")
        return EvalResult.passed(self.name, f"The answer is {n} words, within the {self.max_words}-word limit.")


ev = Evals(evaluators=["repetition", AnswerLengthEvaluator(max_words=12)],   # by name, or an instance
           judge_provider=StubClient("judge"))

verdicts = ev.evaluate(
    agent_output="Book a same-day appointment at your Clalit clinic. If breathing becomes difficult, go to the ER.",
    response_id="msg_01Ab3Cd5Ef",
    tenant_id="clalit",
)
print(verdicts)
print("\nrecord:", [(v.name, v.label, len(v.calls)) for v in verdicts])   # neither evaluator called a model
