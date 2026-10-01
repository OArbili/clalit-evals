# Examples

Standalone scripts for the installed package: `pip install "clalit-evals @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"` (or a wheel from a GitHub release), copy one, run it. Nothing
here imports from this repository. Each runs with `--offline` (a function stands in for the judge, no
key) or against the judge your environment names (`ANTHROPIC_API_KEY`, or `OPENAI_API_KEY` /
`OPENAI_BASE_URL` with `EVALS_JUDGE_MODEL`).

| Script | Shows | Needs |
|---|---|---|
| `01_evaluate_one_turn.py` | the first call: one turn in, verdicts out | `clalit-evals` |
| `02_choose_evaluators_and_read_verdicts.py` | choosing evaluators, `only=`, every field of a verdict, the judge-call record, how an empty reply comes back | `clalit-evals` |
| `03_openai_client.py` | the judge through `openai.OpenAI()` (`--offline`: a client-shaped double), or any OpenAI-compatible endpoint over the standard library (`--http`) | `clalit-evals[openai]` |
| `04_custom_evaluator.py` | your own evaluators: a new dimension on the shared rubric, your own prompts, your own answer shape, a deterministic check | `clalit-evals` |
| `05_many_turns.py` | fifty turns through a thread pool, pass rates and the rows to look at | `clalit-evals` |
| `06_summary_evaluation.py` | an LLM summary against a human reference: SummaryFactCapture (recall) and SummaryFactPrecision (precision) on three summaries — facts missed, a claim invented, faithful — with `threshold=`, `capture_content=`, the count attributes and the two judge calls per verdict | `clalit-evals` |
| `07_clinical_evaluators.py` | three clinical checks on a triage conversation — questions before routing, red flags ruled out, focused questioning — with a protocol of your own, and the protocol's version on the call record | `clalit-evals` |

`../getting_started/` has the walkthroughs that run from a checkout of this repository (they share a
stub Anthropic client and print more of what happens inside); `../docs/DEVELOPER_GUIDE.md` explains
everything the scripts do, in order.
