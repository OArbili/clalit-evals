# Evaluators

One README per evaluator: what it checks, the inputs it reads, the rule as the judge prompt states it, the OpenTelemetry output, and a Hebrew pass and fail example. The rule sections are generated from the evaluator classes by `tools/build_evaluator_docs.py`, so they always match the prompts that run.

| Evaluator | Registry name | Type | Score | Reads |
|---|---|---|---|---|
| [Repetition](repetition.md) | `repetition` | deterministic | 1 / 0 | `agent_output`, `history` |
| [Relevance](relevance.md) | `relevance` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [Naturalness](naturalness.md) | `naturalness` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [Empathy](empathy.md) | `empathy` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [Attentiveness](attentiveness.md) | `attentiveness` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [Coherence](coherence.md) | `coherence` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [ContextRetention](context_retention.md) | `context_retention` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [AppropriateQuestioning](appropriate_questioning.md) | `appropriate_questioning` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [Adaptability](adaptability.md) | `adaptability` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [NonRobotic](non_robotic.md) | `non_robotic` | LLM judge | 1 / 0 | `agent_output`, `user_input`, `history` |
| [SummaryFactCapture](summary_fact_capture.md) | `summary_fact_capture` | reference-based LLM judge | ratio, pass ≥ 0.8 | `agent_output`, `reference` |
| [SummaryFactPrecision](summary_fact_precision.md) | `summary_fact_precision` | reference-based LLM judge | ratio, pass ≥ 1.0 | `agent_output`, `reference` |

Every evaluator returns the same kind of verdict, which the service reports as the same `gen_ai.evaluation.result` event and the same `evals.score` / `evals.errors` metrics; all share the five error types. Specification: [`../EVALS_REQUIREMENTS.md`](../EVALS_REQUIREMENTS.md); summary evaluators in full: [`../SUMMARY_FACT_CAPTURE.md`](../SUMMARY_FACT_CAPTURE.md); how to add one: [`../ADDING_AN_EVALUATOR.md`](../ADDING_AN_EVALUATOR.md).
