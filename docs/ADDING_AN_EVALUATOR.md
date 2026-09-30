# Adding an evaluator

An evaluator is one check on one agent turn. Each lives in its own module under
`src/evals/evaluators/`, and `Evals` gives every evaluator the same treatment:
timing, failure isolation, the record of its judge calls on the `Verdict`, and
a place in `Verdicts` (and in the service's `Report` for a batch). The service then reports each verdict from
that record as its own `evaluate {name}` span, a `gen_ai.evaluation.result`
event parented to the evaluated turn, and the `evals.score` / `evals.errors`
metrics. You write the check; the package and the service do the rest.

## The contract

```python
from evals.evaluators.base import EvalResult, Evaluator, Turn

class MyEvaluator(Evaluator):
    name = "MyCheck"                      # becomes gen_ai.evaluation.name

    def evaluate(self, turn: Turn) -> EvalResult:
        ...
        return EvalResult.passed(self.name, "one sentence saying why")
```

- `name` is the identity of the evaluator everywhere: on the event, the metric
  dimension, the span name, and `verdicts["MyCheck"]`. Use one word in
  PascalCase, as the specification's examples do (`Repetition`, `Relevance`),
  and keep it unique.
- `evaluate` returns exactly one of four things:

  | Constructor | When |
  |---|---|
  | `EvalResult.passed(name, explanation)` | the turn meets the check; score 1, label `pass` |
  | `EvalResult.failed(name, explanation)` | it does not; score 0, label `fail` |
  | `EvalResult.scored(name, score, *, passed, explanation, attributes=None)` | a ratio score with a label decided by your threshold (the summary fact evaluators) |
  | `EvalResult.errored(name, error_type, detail)` | the check could not be made |

  `EvalResult` lives in `evals.evaluators.base` (also exported from
  `evals.evaluators`). If the evaluator called a model, attach the calls with
  `.with_calls(calls)` so they reach the `Verdict` (see *Step 1*).

- **Return errors, don't raise them.** A missing input, a judge that timed out,
  a response that could not be parsed: these are `errored` results, with
  `error_type` one of `missing_input`, `timeout`, `rate_limit`,
  `provider_error`, `invalid_response` (constants in `evals.vocab`). Anything
  you *do* raise is still caught by `Evals` and recorded as an errored
  verdict, so a bug never kills a batch, but a deliberate `errored` result
  carries a better explanation than a stack trace.
- Release 1 scores are binary: 1 or 0. The constructors enforce that.
- The explanation is user-facing: one sentence, stating the reason, in plain
  language. It is what a reviewer reads in Elasticsearch.

What the turn gives you: `turn.agent_output` (always a string; empty means
missing), `turn.user_input`, `turn.history` (a list of `Message(role, text)`),
`turn.reference` for a reference-based check, and the ids. Never read anything
else. There is no OpenTelemetry to touch: the package has none. If your
evaluator calls a model, subclass `JudgeBacked` (`src/evals/evaluators/_judge.py`)
and make every call through `self.ask_judge(...)`: it holds the judge (provider,
model, `bind`) and records each call as a `JudgeCall` for you; for a call made
some other way, build a `JudgeCall` and attach it with `with_calls`. Either way
the calls travel on the result, so the
service can report it as a `chat {model}` span and a cost report can read it.

## Step 1 — write the module

Create `src/evals/evaluators/<name>.py`. A complete deterministic example:

```python
"""AnswerLength: the answer is short enough to read on a phone."""

from __future__ import annotations

from .. import vocab
from .base import EvalResult, Evaluator, Turn


class AnswerLengthEvaluator(Evaluator):
    """Fails when the answer exceeds `max_words` words."""

    name = "AnswerLength"

    def __init__(self, max_words: int = 80) -> None:
        self.max_words = max_words

    def evaluate(self, turn: Turn) -> EvalResult:
        if not turn.agent_output:
            return EvalResult.errored(
                self.name, vocab.ERR_MISSING_INPUT, "gen_ai.output.messages is absent or empty."
            )
        n = len(turn.agent_output.split())
        if n > self.max_words:
            return EvalResult.failed(self.name, f"The answer is {n} words; the limit is {self.max_words}.")
        return EvalResult.passed(self.name, f"The answer is {n} words, within the {self.max_words}-word limit.")
```

For an **LLM-as-judge** evaluator, subclass `LLMJudgeEvaluator` and write no
plumbing at all. The base (through `JudgeBacked`) already accepts `provider`,
`model`, `effort` and `timeout`; makes the call through `ask_judge(*, system,
user, schema, prompt_version, calls, max_tokens=1024)`, which records it as a
`JudgeCall` on the result (the service reports it as a
`chat {model}` CLIENT span under the `evaluate {name}` span); asks for
structured output; checks for a refusal before reading content; maps the
provider's `LLMError`s to the error types; and renders the same
`<conversation_history>` / `<user_message>` / `<assistant_reply>` layout for
every dimension. You supply the definition and the fail criteria, phrased as
literal, observable conditions, because that is what makes verdicts consistent
across same or similar inputs. (Different wording for the rubric or the turn: override
`system_prompt()` / `user_prompt()`; a different answer shape: subclass
`JudgeBacked` and call `ask_judge()` — both shown in DEVELOPER_GUIDE.md §10.)

```python
from .._judge import LLMJudgeEvaluator   # from a module inside evals/evaluators/: from ._judge import ...


class PolitenessEvaluator(LLMJudgeEvaluator):
    name = "Politeness"
    prompt_version = "politeness/v1"
    definition = "The reply is courteous to the user."
    fail_criteria = (
        "The reply contains an insult, sarcasm or a dismissive remark directed at the user.",
        "The reply orders the user to do something with no courtesy marker where one is customary.",
    )
    pass_note = "Brevity is not impoliteness."
```

Override `precheck(turn)` to answer without a model call when the dimension's
situation does not arise (see `appropriate_questioning.py`: no question in the
reply means pass). Keep criteria literal: "the reply asks for information the user
already gave" is checkable; "the reply is unhelpful" is not, and will drift between
runs.

## Step 2 — make it available

Two ways; the first needs no changes to the package.

**Pass an instance.** Works for evaluators that live anywhere, including in
your own project:

```python
ev = Evals(evaluators=["repetition", "relevance", AnswerLengthEvaluator(max_words=60)])
```

**Register a name.** So that `Evals(evaluators=["answer_length"])` and
`EVALS_*` configuration can refer to it, add a factory to `BUILTIN` in
`src/evals/evaluators/__init__.py`:

```python
from .answer_length import AnswerLengthEvaluator

BUILTIN = {
    ...
    "answer_length": lambda **_: AnswerLengthEvaluator(),
}
```

Factories are called with Evals' shared dependencies as keyword
arguments, `judge_provider` and `judge_model`, and must accept `**_` so that new dependencies added
later do not break existing entries. A judge evaluator picks the ones it
needs, as the `relevance` entry does.

**Dependencies.** `Evals` calls `bind(judge_provider=, judge_model=)` on
every evaluator, built or passed in, and a judge fills in whatever it was not
given explicitly. So `Evals(evaluators=[MyJudge()])` uses the
`Evals` judge provider and model, and a registry entry can be as short as
`"my_judge": lambda **_: MyJudge()`.

## Step 3 — test it

`strict=True` turns any exception into a test failure instead of an error
verdict. The helpers in `tests/helpers.py` are reusable: `make()` builds an
`Evals` on the stub judge with `strict=True`; `FixedJudge` / `RaisingJudge` are
fake clients; `ScriptedJudge` replays the dataset's labels. Put the unit test
in `tests/unit/`:

```python
def test_answer_length():
    ev = make(evaluators=[AnswerLengthEvaluator(max_words=3)])
    v = ev.evaluate(agent_output="one two three four.", response_id="r1")
    assert v["AnswerLength"].passed is False
    assert "4 words" in v["AnswerLength"].explanation
    assert v["AnswerLength"].calls == ()             # no model call on the record

    e = events(v)[0]                                  # what the service would report
    assert e["attributes"][sc.EVAL_NAME] == "AnswerLength"
    assert e["attributes"][sc.EVAL_SCORE_LABEL] == "fail"
    assert spans("evaluate AnswerLength", v)
```

Test the error path too: an empty `agent_output` must yield `error_type ==
"missing_input"` with no score attributes on the event. For a judge, check
the record: one `JudgeCall` per model call, with the `prompt_version` you set.

Then give it ground truth. Every registered evaluator has a label on every turn
of `tests/data/synthetic_conversations.jsonl` (see `tests/data/README.md`): add
your evaluator's key to each turn's `expected` map, with at least two clear
`fail` cases and five clear `pass` cases, and `null` wherever the case is not
clear-cut. Add the key to `DIMENSIONS` in `tests/data/synthetic.py` and, for a
judge, its display name to `ScriptedJudge._KEY` in `tests/helpers.py`. The
offline e2e test then checks that the pipeline reproduces your labels, and the
live test (`EVALS_LIVE=1 pytest -m live`) measures how often the real judge
agrees with them.

## Step 4 — document it

- Add a row to the evaluator table in `README.md` and `docs/DEVELOPER_GUIDE.md` §2,
  and a `CHANGELOG.md` entry.
- Give it a README in `docs/evaluators/`: add a Hebrew fail and pass example to
  `EXAMPLES` in `tools/build_evaluator_docs.py` (for a non-judge, also its `what` and
  `rule` text in `HAND_WRITTEN`) and run
  `python3 tools/build_evaluator_docs.py`. The rule section is rendered from
  your class, and `tests/unit/test_evaluator_docs.py` fails until the file on
  disk matches.

## Checklist

- [ ] one module in `src/evals/evaluators/`, one `Evaluator` subclass, a unique PascalCase `name`
- [ ] `evaluate` returns `passed` / `failed` / `errored`; expected failures are returned, not raised
- [ ] reads only `agent_output`, `user_input`, `history` and the ids
- [ ] judge evaluators subclass `LLMJudgeEvaluator`: a `definition`, literal `fail_criteria`, a `prompt_version`, and a `precheck` where the situation may not arise
- [ ] registered in `BUILTIN` if it should be addressable by name
- [ ] tests for the pass, fail and `missing_input` paths, on the returned verdict (label, score, explanation, `calls`)
- [ ] ground-truth labels for the evaluator on every turn of the synthetic dataset
- [ ] Hebrew examples in `tools/build_evaluator_docs.py` and the regenerated `docs/evaluators/<name>.md`
- [ ] README table, developer guide table and CHANGELOG updated
