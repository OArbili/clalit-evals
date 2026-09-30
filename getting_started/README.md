# Getting started

Runnable examples, in learning order. Every one runs offline by default (a stub judge), so no key and no
network are needed:

```bash
source .venv/bin/activate
python3 getting_started/simple_example.py
```

| # | Script | Shows |
|---|---|---|
| 1 | `simple_example.py` | Evaluate one turn and print the verdicts: the minimum call, then the same turn with every identifier. |
| 2 | `example.py` | One turn with logging on and the record behind each verdict (timing, judge calls). `--offline`, or a real judge from the environment. |
| 3 | `custom_evaluator_example.py` | A deterministic evaluator of your own next to the built-in ones. |
| 4 | `custom_judge_example.py` | Three ways to write an evaluator that uses the judge: a new dimension on the standard rubric, your own prompts, your own answer shape (docs/DEVELOPER_GUIDE.md section 10). |
| 5 | `summary_example.py` | The two reference-based summary evaluators on four summaries of one reference, and the attributes on their verdicts. |

`stub_client.py` is the fake judge the offline runs use. Each script's docstring states how to run it for real
(the environment variables).
