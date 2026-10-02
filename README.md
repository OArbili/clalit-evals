# clalit-evals

An evals framework for LLM agents, as a Python package: turn-level evaluators, one
object and one verb — `Evals().evaluate(turn)` returns a verdict per evaluator, each with
the record of how it was produced — built to be reported as OpenTelemetry without
depending on it. Written for a Hebrew-speaking health-fund triage assistant and usable
for any conversational agent. The package holds the evaluation logic only: no telemetry,
no storage, no dependencies.

This is a proof of concept. The framework is this package; the proof around it — a REST
service that hosts it and reports every verdict as OpenTelemetry, sinks for Elastic,
Langfuse and MLflow, browser UIs, the specification, runs on real exports — is in the
companion repository, `OArbili/noy`, which is private.

The repository is public and the code is MIT-licensed. The package is installed from here, by release
tag; no account, key or token is needed:

```bash
pip install "clalit-evals @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"
pip install "clalit-evals[anthropic] @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"   # + the Anthropic SDK
pip install "clalit-evals[openai] @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"      # + the openai SDK client
```

Without git, install the wheel attached to the [release](https://github.com/OArbili/clalit-evals/releases):

```bash
pip install https://github.com/OArbili/clalit-evals/releases/download/v0.3.0/clalit_evals-0.3.0-py3-none-any.whl
```

The package is not on PyPI: `pip install clalit-evals` without the URL gets an old placeholder, not this code.

The judge over any OpenAI-compatible endpoint needs nothing beyond the package. The import name is `evals`.

```python
from evals import Evals

ev = Evals()      # the judge from ANTHROPIC_API_KEY, or OPENAI_API_KEY / OPENAI_BASE_URL; or Evals(judge_provider=...)
verdicts = ev.evaluate(
    user_input="I've had a fever of 38.5 and a sore throat since yesterday. Should I come in?",
    agent_output="Book a same-day appointment at your clinic. If breathing becomes difficult, go to the ER.",
    response_id="msg_01Ab3Cd5Ef",
)
print(verdicts)
```

```
Repetition  pass   No sentence is repeated (intra-response comparison over 2 sentences).
Relevance   pass   The reply recommends a care pathway for the fever and sore throat the member described.
```

## The evaluators

| Name | Kind | Checks |
|---|---|---|
| `repetition` | deterministic | the same sentence twice — in the reply, or against an earlier agent turn |
| `relevance` | judge | at least part of the reply relates to the member's message |
| `naturalness` | judge | reads like a person, not a template |
| `empathy` | judge | an expressed worry or pain is recognised specifically |
| `attentiveness` | judge | the reply reacts to what the member actually said |
| `coherence` | judge | the reply is consistent with itself |
| `context_retention` | judge | earlier information is remembered, not re-asked |
| `appropriate_questioning` | judge | follow-up questions are relevant and proportionate |
| `adaptability` | judge | the reply adjusts to a change in the member's situation |
| `non_robotic` | judge | no mechanical, checklist-like phrasing |
| `questions_before_routing` | judge, clinical | the information a routing depends on was collected before the reply routes the member (protocol list is a parameter) |
| `red_flags_ruled_out` | judge, clinical | red flags were ruled out before a non-urgent routing (protocol list is a parameter) |
| `focused_questioning` | judge, clinical | the questions in one reply are about one subject |
| `summary_fact_capture` | judge, reference-based | recall: the share of a human reference's facts an LLM summary states |
| `summary_fact_precision` | judge, reference-based | precision: the share of the LLM summary's claims the reference supports |

The judge dimensions share one rubric and one answer schema, so their verdicts are
comparable; each contributes only its definition and its literal fail criteria. The two
routing checks take your clinical protocol:
`RedFlagsRuledOutEvaluator(red_flags=[...], require_all=True)`; their defaults are generic. Hebrew
and English are judged in their own language; explanations are one English sentence.

```python
ev = Evals(evaluators=["repetition", "relevance", "empathy", "context_retention"])
```

## The judge

Any of: the environment (`ANTHROPIC_API_KEY`; `OPENAI_API_KEY` / `OPENAI_BASE_URL` for
OpenAI, Azure OpenAI, vLLM, Ollama, LiteLLM; `EVALS_JUDGE_MODEL`), an explicit provider,
an SDK client, or a function — the last one makes every evaluator testable offline.

```python
from evals.providers import AnthropicProvider, OpenAICompatibleProvider

Evals(judge_provider=AnthropicProvider(), judge_model="claude-sonnet-5-5")
Evals(judge_provider=OpenAICompatibleProvider("http://vllm.internal:8000/v1"), judge_model="gemma-3-27b")
Evals(judge_provider=openai.OpenAI(), judge_model="gpt-5")            # an openai SDK client, as is
Evals(judge_provider=fake_judge, judge_model="fake-judge")            # def fake_judge(*, system, messages, schema, **_): ...
```

## What comes back

```python
verdicts.ok                     # every evaluator passed (an error is not a pass)
verdicts["relevance"].label     # 'pass' | 'fail', or None with .error_type set: missing_input, timeout,
                                # rate_limit, provider_error, invalid_response -- evaluate() never raises for these
verdicts["relevance"].calls     # one JudgeCall per model call: provider, model, prompt version, tokens, timing
verdicts.to_dict()              # the serialisable form, ids and record included
```

The record is what a host needs to report or bill each verdict — as OpenTelemetry
spans and a `gen_ai.evaluation.result` event, a Langfuse score, an MLflow feedback, a
row in Elastic. The package produces none of those; a host does (the POC's REST service,
in the private companion repository, is one such host).

## Your own evaluator

```python
from evals.evaluators import LLMJudgeEvaluator

class PolitenessEvaluator(LLMJudgeEvaluator):          # a new dimension on the shared rubric: data only
    name = "Politeness"
    prompt_version = "politeness/v1"
    definition = "The reply is courteous to the member."
    fail_criteria = ("The reply contains an insult, sarcasm or a dismissive remark directed at the member.",)

ev = Evals(evaluators=["repetition", PolitenessEvaluator()])
```

Own wording: override `system_prompt()` / `user_prompt()`. Own answer shape (a graded
score, several findings): subclass `JudgeBacked` and call `ask_judge()`. No judge at all:
subclass `Evaluator`. [docs/DEVELOPER_GUIDE.md](docs/DEVELOPER_GUIDE.md) walks through all of
them, from the first call; [docs/ADDING_AN_EVALUATOR.md](docs/ADDING_AN_EVALUATOR.md) is the
checklist for adding one to this repository; [docs/evaluators/](docs/evaluators/) has one page
per built-in evaluator with Hebrew examples; [docs/SUMMARY_FACT_CAPTURE.md](docs/SUMMARY_FACT_CAPTURE.md)
specifies the two summary evaluators.

## Requirements

Python 3.12 or later. No runtime dependencies; `anthropic` or `openai` only if you use
that SDK. MIT licence.

## Examples

[examples/](examples/) has seven standalone scripts for the installed package — copy one and run it,
`--offline` for no key: the first call; choosing evaluators and reading every field of a verdict;
the judge through an `openai` client or an OpenAI-compatible endpoint; your own evaluators three
ways; fifty turns through a thread pool; an LLM summary against a human reference with the two summary
evaluators; the three clinical checks with a protocol of your own. The test suite runs them against a freshly built wheel in
a clean virtual environment, so they are exactly what the installed package gives you.

## Development

```bash
git clone https://github.com/OArbili/clalit-evals.git && cd clalit-evals
python3 -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest                                  # offline: scripted judges, no key
python3 getting_started/simple_example.py
```

Layout: `src/evals/` the package (`core.py` is `Evals`; `types.py` the data; `evaluators/` one module per
evaluator; `providers/` the judge adapters; `vocab.py` the values a verdict is written with);
`examples/` standalone scripts for the installed package; `getting_started/` walkthroughs from a checkout; `tests/` unit and offline end-to-end tests on a
labelled Hebrew/English dataset (`tests/data/`); `tools/build_evaluator_docs.py` regenerates
`docs/evaluators/` from the prompts in the code. Releases: bump `version` in `pyproject.toml`, add a
`CHANGELOG.md` entry, commit, then `gh release create vX.Y.Z`: `.github/workflows/release.yml` builds the wheel
and sdist and attaches them to the release. Nothing is uploaded to PyPI.
