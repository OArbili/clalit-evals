# Summary fact evaluators — requirements (draft v0.2)

**Status:** approved 5 September 2026 and implemented in
`src/evals/evaluators/summary_facts.py` (thresholds 0.8 / 1.0, empty LLM summary =
`missing_input`, as proposed in section 9). Summarised as *Summary fact
evaluators (release 3)* in the POC's requirements document (`OArbili/noy`, docs/EVALS_REQUIREMENTS.md); this document remains the
full specification. v0.2 added the precision evaluator (v0.1 had capture only).

## 1. Purpose

An LLM writes a summary of a triage conversation. A person wrote a reference
summary of the same conversation (the ground truth, GT). Two evaluators compare
them, fact by fact, in opposite directions:

| Evaluator | Question | Score |
|---|---|---|
| **`SummaryFactCapture`** (key `summary_fact_capture`) | Did the LLM summary miss anything the GT says? | **capture ratio** = GT facts that the LLM summary states ÷ all GT facts (recall) |
| **`SummaryFactPrecision`** (key `summary_fact_precision`) | Did the LLM summary say anything the GT does not support? | **precision** = LLM claims supported by the GT ÷ all LLM claims |

Each is a separate evaluator with its own `gen_ai.evaluation.result` event,
because an event carries one score and the two answers call for different
actions (a missed fact is a completeness defect, an invented one is a safety
defect). A summary is clean when both pass, which the SDK's `Verdicts.ok`
already expresses. Both are one summary at a time, like every existing
evaluator.

## 2. Definitions

| Term | Meaning |
|---|---|
| Fact / claim | One atomic, self-contained statement that can be checked on its own: a symptom, its onset or duration, a measurement, a negation ("not pregnant"), an action taken, a recommendation given. A sentence with two statements yields two. "Fact" is used for statements extracted from the GT, "claim" for statements extracted from the LLM summary; the extraction rule is the same. |
| Stated by / supported by a text | The text states the statement, or an equivalent paraphrase of it, with the same numbers, dates and polarity. "Sore throat" vs "throat pain" counts; "38.5" vs "38" does not; "no fever" vs "fever" does not. |
| Captured / missing | A GT fact that the LLM summary states / does not state. |
| Supported / unsupported | An LLM claim that the GT states / does not state. An unsupported claim is either an addition the GT never mentions or a contradiction of a GT fact. |
| Capture ratio | captured ÷ GT facts, a double in [0, 1]. |
| Precision | supported ÷ LLM claims, a double in [0, 1]. |

A contradiction counts on both sides: the GT fact is missing (capture) and the
LLM claim is unsupported (precision). A pure addition only lowers precision.

## 3. Inputs

Identical for both evaluators. The LLM summary is the output of a GenAI
operation, so it arrives exactly as `agent_output` does for every existing
evaluator. The GT summary is not on any span — OpenTelemetry's GenAI
conventions have no attribute for a human reference — so it is a new field on
`Turn`, read from the annotation store and joined on the same identifiers.

### Correlation fields

Unchanged from the existing spec: Service (`service.name`), Group (`service.namespace`), Conversation id
(`gen_ai.conversation.id`), Trace id and Span id of the **summarisation** span.
The summarisation span is a normal `chat {model}` CLIENT span
(`gen_ai.operation.name = "chat"`).

### Content fields

| Field on `Turn` | Read from | Example | Required |
|---|---|---|---|
| `agent_output` — the LLM summary | `gen_ai.output.messages` of the summarisation span (text of the assistant message) | "34-year-old woman, fever 38.5 since yesterday, sore throat. Advised same-day GP appointment." | yes |
| `reference` — the GT summary **(new)** | the annotation store, keyed by `gen_ai.response.id` of the summarisation span (fallback: `gen_ai.conversation.id`) | "Female, 34. Fever 38.5 since yesterday and sore throat. No breathing difficulty. Not pregnant. Advised same-day appointment with family doctor; ER if breathing worsens." | yes |
| `response_id` | `gen_ai.response.id` of the summarisation span | `msg_01Ab3Cd5Ef` | yes |
| `model` | `gen_ai.request.model` of the summarisation span — a metric dimension only | `claude-opus-4-8` | optional |
| `user_input`, `history` | not read by either evaluator | — | no |

### Configuration (per evaluator instance, not per turn)

| Option | Capture default | Precision default | Meaning |
|---|---|---|---|
| `threshold` | `0.8` | `1.0` | `score.label = "pass"` when the score ≥ threshold. Proposed: a missed fact is tolerable up to a point, an invented clinical fact is not. **Decision needed** — section 9. |
| `judge_model` | the SDK's judge model | same | Model for both steps of section 5. |
| `capture_content` | `False` | `False` | Whether the missing facts / unsupported claims themselves are put on the verdict (they are member medical facts). An option of the evaluator instance: `SummaryFactCaptureEvaluator(capture_content=True)`; registry names build it off. |

## 4. Outputs

Each evaluator emits one `gen_ai.evaluation.result` event, exactly as the
existing evaluators do, plus extension attributes in our own `eval.` namespace
(the namespace `eval.judge.prompt_version` already uses; they are not in the
GenAI registry and are marked as such). The two evaluators use different
extension names so their events are never confused in the same index.

### Common attributes

| Attribute | Type | Requirement | Value |
|---|---|---|---|
| `gen_ai.evaluation.name` | string | required | `SummaryFactCapture` or `SummaryFactPrecision` |
| `gen_ai.evaluation.score.value` | double | conditionally required (absent on error) | the ratio, `0.0`–`1.0` |
| `gen_ai.evaluation.score.label` | string | conditionally required (absent on error) | `pass` if score ≥ that evaluator's threshold, else `fail` |
| `gen_ai.evaluation.explanation` | string | recommended | see below |
| `gen_ai.response.id` | string | recommended | the summarisation response id |
| `gen_ai.conversation.id` | string | recommended | the conversation id |
| `error.type` | string | conditionally required (only on error) | one of the five spec error types |

### Extension attributes — SummaryFactCapture

| Attribute | Type | Requirement | Value |
|---|---|---|---|
| `eval.facts.total` | int | recommended | number of GT facts |
| `eval.facts.captured` | int | recommended | number the LLM summary states |
| `eval.facts.missing` | string[] | opt-in (`capture_content=True`) | the missing GT facts, verbatim |

Explanation: "7 of 9 reference facts captured (0.78). Missing: '…'; '…'." or
"All 9 reference facts are captured."

### Extension attributes — SummaryFactPrecision

| Attribute | Type | Requirement | Value |
|---|---|---|---|
| `eval.claims.total` | int | recommended | number of claims in the LLM summary |
| `eval.claims.supported` | int | recommended | number the GT states |
| `eval.claims.unsupported` | string[] | opt-in (`capture_content=True`) | the unsupported claims, verbatim |

Explanation: "7 of 8 claims are supported by the reference (0.88). Unsupported:
'…'." or "All 8 claims are supported by the reference."

The event is parented to the summarisation span when its trace/span ids are
known, otherwise unparented with `gen_ai.response.id` as the link — the existing
rule.

**Spans.** Per evaluator, one `evaluate {name}` INTERNAL span linked to the
summarisation span, with two child `chat {judge_model}` CLIENT spans (one per
step in section 5), each carrying `eval.judge.prompt_version`
(`summary_facts/extract/v1` and `summary_facts/verify/v1` — the same two
prompts serve both evaluators). The service produces them from the verdict's
`JudgeCall` records.

**Metrics.** Unchanged names and dimensions. `evals.score` records the ratio, so
for these evaluators the histogram's mean is the average capture ratio or
precision rather than a pass rate; its description changes from "1 = pass and
0 = fail" to "evaluation score in [0, 1]; binary evaluators record 1 or 0".
`evals.errors` as today.

**SDK result.** The existing `Verdict` type carries both unchanged: `score` is
the ratio, `label`/`passed` follow the threshold, `explanation` as above. The
CSV run script gains a `<key>_score` column next to `_label`.

## 5. How it works, step by step

Both evaluators run the same two prompts. Capture extracts from the GT and
verifies against the LLM summary; precision extracts from the LLM summary and
verifies against the GT.

1. **Validate.** `agent_output` empty → `missing_input` ("the LLM summary is
   absent or empty"). `reference` empty → `missing_input` ("the reference summary
   is absent"). No model is called.
2. **Extract** (judge call 1) from the *source* text — the GT for capture, the
   LLM summary for precision. The judge receives only that text and returns, as
   structured output, `{"statements": [string, ...]}`: atomic statements in the
   text's own language, in order of appearance, each self-contained (pronouns
   resolved), numbers and negations verbatim.
   For capture the result is **cached by the SHA-256 of the reference text** for
   the life of the `Evals` instance: every LLM summary of the same conversation
   is measured against the same fact list, and a re-run costs one call per
   summary, not two. Precision cannot cache — every LLM summary is different.
3. **Guard.** Zero statements from a non-empty source → `missing_input`
   ("the reference contains no checkable facts" / "the LLM summary contains no
   checkable claims"), since the input, not the judge, is at fault. A malformed
   extraction reply → `invalid_response`.
4. **Verify** (judge call 2) against the *target* text — the LLM summary for
   capture, the GT for precision. The judge receives the target text and the
   numbered statement list, and returns `{"results": [{"statement": n,
   "stated": bool, "evidence": string}, ...]}` — one entry per statement, in
   one call, so every statement is judged with the same view of the target.
   `evidence` quotes the target's words that state it, or is empty. The decision
   rule is the literal one of section 2.
5. **Guard.** Exactly one result per statement, otherwise `invalid_response`.
6. **Score.** ratio = stated ÷ total. `label = pass` if ratio ≥ that
   evaluator's threshold, else `fail`. Explanation as in section 4.
7. **Return.** The verdict with its record: the two judge calls as
   `JudgeCall`s (`prompt_version` `summary_facts/extract/v1` and
   `summary_facts/verify/v1`) and the fact / claim counts as attributes. The
   service reports the event, span and metric of section 4 from that record.
   Judge failures map to the five spec error types exactly as for the
   dimension judges; an errored turn carries no score attributes and is not
   recorded in `evals.score`.

Running both evaluators on one summary costs three judge calls when the
reference's facts are cached (capture: verify; precision: extract + verify),
four on the first summary of a conversation.

## 6. Consistency rules

- Same source text → same statement list (the cache for the GT side; a fixed
  extraction prompt with low effort and a closed schema on both sides).
- Same (reference, LLM summary) → same verdicts: all statements are verified in
  one call under the literal rule; the judge is told that identical inputs must
  yield identical verdicts, as the dimension judges are.
- The judge never sees the source conversation, so it cannot credit the LLM
  summary for facts the GT omits, nor the GT for facts the LLM summary adds.
  The GT is the sole authority in both directions.
- The same two prompts serve both evaluators, so "stated by" means the same
  thing whichever direction is being judged.
- Hebrew and English texts are handled in their own language; statements are
  extracted in that language; the explanation is English, quoting statements
  as plain text.

## 7. Error handling

Identical to the existing spec: `missing_input`, `timeout`, `rate_limit`,
`provider_error`, `invalid_response`; score attributes omitted, never null; the
error counted in `evals.errors`; other turns in the batch unaffected. A failure
in either judge call fails that evaluator's whole evaluation — there is no
partial score. The two evaluators fail independently: a precision error does not
void a capture verdict on the same summary.

## 8. Worked example

Reference (GT) summary:

> Female, 34. Fever 38.5 since yesterday and sore throat. No breathing
> difficulty. Not pregnant. Advised same-day appointment with family doctor; ER
> if breathing worsens.

LLM summary under evaluation:

> 34-year-old woman with fever 38.5 since yesterday and throat pain, breathing
> normally. Known diabetic. Advised a same-day GP appointment.

**SummaryFactCapture.** GT facts (step 2, cached): 1 female · 2 aged 34 ·
3 fever 38.5 · 4 since yesterday · 5 sore throat · 6 no breathing difficulty ·
7 not pregnant · 8 advised same-day family-doctor appointment · 9 advised ER if
breathing worsens. Verification against the LLM summary (step 4): 1–6 and 8
captured ("throat pain" ≈ sore throat, "breathing normally" ≈ no breathing
difficulty); 7 and 9 missing. Score 7 ÷ 9 = **0.78** → `fail` at 0.8.

**SummaryFactPrecision.** LLM claims (step 2): 1 female · 2 aged 34 · 3 fever
38.5 · 4 since yesterday · 5 throat pain · 6 breathing normally · 7 known
diabetic · 8 advised same-day GP appointment. Verification against the GT:
1–6 and 8 supported; 7 unsupported (the GT never mentions diabetes). Score
7 ÷ 8 = **0.88** → `fail` at 1.0 (it would pass at 0.8).

Two events on the same summarisation span:

```json
{
  "event_name": "gen_ai.evaluation.result",
  "attributes": {
    "gen_ai.evaluation.name": "SummaryFactCapture",
    "gen_ai.evaluation.score.value": 0.78,
    "gen_ai.evaluation.score.label": "fail",
    "gen_ai.evaluation.explanation": "7 of 9 reference facts captured (0.78). Missing: 'The member is not pregnant'; 'The member was advised to go to the ER if breathing worsens'.",
    "gen_ai.response.id": "msg_01Ab3Cd5Ef",
    "gen_ai.conversation.id": "conv_8b21f4",
    "eval.facts.total": 9,
    "eval.facts.captured": 7
  }
}
{
  "event_name": "gen_ai.evaluation.result",
  "attributes": {
    "gen_ai.evaluation.name": "SummaryFactPrecision",
    "gen_ai.evaluation.score.value": 0.88,
    "gen_ai.evaluation.score.label": "fail",
    "gen_ai.evaluation.explanation": "7 of 8 claims are supported by the reference (0.88). Unsupported: 'The member is a known diabetic'.",
    "gen_ai.response.id": "msg_01Ab3Cd5Ef",
    "gen_ai.conversation.id": "conv_8b21f4",
    "eval.claims.total": 8,
    "eval.claims.supported": 7
  }
}
```

`eval.facts.missing` and `eval.claims.unsupported` are added only with
`capture_content=True`.

`python3 getting_started/summary_example.py --offline` reproduces this example and three variations of
it (facts missed only, a claim invented only, faithful and complete), printing
the verdicts and the emitted events.

## 9. Out of scope and decisions needed

- **Thresholds.** Proposed 0.8 for capture and 1.0 for precision. Both ratios
  are always recorded, so a threshold only moves the label; the report's pass
  rate and the histogram mean read differently, though, so the defaults deserve
  a deliberate choice.
- **Empty LLM summary.** Proposed: `missing_input` for both evaluators,
  consistent with every other evaluator. The alternative is capture 0 / `fail`
  and precision undefined. To be decided.
- **No combined score.** A single F1-style number would hide which defect a
  summary has; the two events, and `Verdicts.ok` for "both pass", replace it.
- **Fact weighting.** All statements count equally. Weighting safety-relevant
  ones (red flags, negations, medication) higher is a possible v2.
- **Reference source.** The SDK reads `reference` from the row it is given; how
  the annotation store is joined to the export is the caller's job, as it is for
  the export's existing flags.

## 10. Fit with the current SDK

| Change | Where | Compatibility |
|---|---|---|
| `Turn.reference: str \| None` | `evals/types.py` | optional field; existing rows unaffected; `Turn.coerce` picks it up from a `reference` column |
| One new module, two evaluators | `src/evals/evaluators/summary_facts.py` | a shared base that owns the two prompts, the cache, the call record and the error mapping (reusing `call_judge` and the mapping of `_judge.py`, not its single-verdict template); `SummaryFactCaptureEvaluator` and `SummaryFactPrecisionEvaluator` set only the direction and the threshold |
| Registry entries | `src/evals/evaluators/__init__.py` | `summary_fact_capture`, `summary_fact_precision`; both receive the shared judge dependencies |
| `evals.score` description | `evals_service/telemetry.py` | wording only; the metric name and dimensions stay |
| `<key>_score` column | `tools/run_csv_evals.py` | added for every evaluator; 1/0 for the binary ones |
| Synthetic dataset | `tests/data/` | a `reference` and an LLM summary per case, with expected capture and precision ratios; cases for a pure addition, a contradiction, a paraphrase and a changed number; offline e2e with a scripted judge, live e2e against the real judges as today |
| Spec and reference docs | the POC's requirements document and reference | new evaluator section, the extension attributes, the histogram wording |
