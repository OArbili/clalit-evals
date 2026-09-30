# Synthetic evaluation dataset

`synthetic_conversations.jsonl` — one conversation per line, Hebrew and English,
written to exercise every evaluator with clear-cut cases.

Each assistant message carries `expected`: a label per evaluator key
(`repetition`, `relevance`, `naturalness`, `empathy`, `attentiveness`,
`coherence`, `context_retention`, `appropriate_questioning`, `adaptability`,
`non_robotic`), plus `why`. Labels are:

- `"pass"` / `"fail"` — ground truth a competent reviewer would agree with;
- `null` — the case is not clear-cut for that dimension, so tests do not assert it.

A message with `"skip": true` is the automatic inactivity notice; the pipeline
excludes it and tests expect it to be excluded.

Used by:

- `tests/e2e/test_pipeline_offline.py` — the whole pipeline (export CSV in, CSVs
  out) with a scripted judge that replays these labels; asserts the pipeline
  reproduces them exactly. Runs in CI, no network.
- `tests/e2e/test_judges_live.py` — the real judges against the labels
  (`EVALS_LIVE=1`, an API key, and credit); asserts accuracy thresholds and
  verdict consistency across repeated runs.

`synthetic.py` converts the file into the Elastic export shape
(`conversation`, `metadata`, `session_id`, `repetition_flag`, `relevancy_flag`)
so `tools/run_csv_evals.py` is tested on the same format it sees in production.

## Summaries: `synthetic_summaries.jsonl`

One record per (reference summary, LLM summary) pair for the summary fact
evaluators: the two texts, the reference's facts and the LLM summary's claims as
a reviewer would extract them, a `captured` / `supported` flag per statement,
and the `expected` capture and precision scores and labels (or an expected
error). Cases cover a faithful paraphrase, a changed number, a contradiction, a
pure addition, omissions only, a shared reference (the fact cache), an empty
summary, a missing reference, and scores just above and below the capture
threshold, in Hebrew and English.

Used by `tests/e2e/test_summaries_offline.py` (a scripted judge replays the
labelled statements from the real prompts) and `tests/e2e/test_summaries_live.py`
(the real judges; scores within a tolerance, labels where the expected score is
clear of the threshold, and consistency across runs).
