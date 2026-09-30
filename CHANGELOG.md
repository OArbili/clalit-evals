# Changelog

All notable changes to the `clalit-evals` package are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/); the project follows [Semantic Versioning](https://semver.org/).

## [0.2.0] - 2026-09-30

The first release of the evaluators; the package moved here from the evals platform repository, where
its history up to this point lives. Released as a wheel on the GitHub release, not on PyPI: this repository
is private. (0.1.0 was a placeholder with a single `hello_world()`, and is what the PyPI name still holds.)

### Added

- `Evals`: one object, one verb — `evaluate(turn)` returns `Verdicts`, one `Verdict` per evaluator, each
  carrying its record (`started_at`, `ended_at`, and one `JudgeCall` per model call).
- Twelve evaluators: Repetition (deterministic); Relevance, Naturalness, Empathy, Attentiveness, Coherence,
  ContextRetention, AppropriateQuestioning, Adaptability, NonRobotic (LLM-as-judge on one shared rubric);
  SummaryFactCapture and SummaryFactPrecision (an LLM summary against a human reference, fact by fact).
- Providers for the judge: Anthropic (SDK client), any OpenAI-compatible endpoint over the standard library,
  an `openai` SDK client (`OpenAI()`, `AzureOpenAI()`), or a plain function.
- Bases for your own evaluators: `Evaluator`, `JudgeBacked` (holds the judge, `ask_judge()`),
  `LLMJudgeEvaluator` (the pass/fail rubric; override `system_prompt()` / `user_prompt()` for your own wording).
- No dependencies. No telemetry and no storage inside the package: the record on each verdict is what a host
  reports; the companion REST service in the repository turns it into OpenTelemetry and stores it.

## [0.1.0] - 2026-09-24

### Added

- `hello_world()`, which prints `Hello, World!` (placeholder release).
