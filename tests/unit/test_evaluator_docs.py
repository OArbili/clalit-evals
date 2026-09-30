"""docs/evaluators/ must match what tools/build_evaluator_docs.py renders from the current prompts."""

from pathlib import Path

import pytest

import build_evaluator_docs as b
from evals.evaluators import BUILTIN

ROOT = Path(__file__).resolve().parents[2]


def test_every_registered_evaluator_has_a_readme_and_examples():
    for key in BUILTIN:
        assert key in b.EXAMPLES, f"add Hebrew examples for {key} to build_evaluator_docs.EXAMPLES"
        assert set(b.EXAMPLES[key]) == {"fail", "pass"}
        assert (ROOT / "docs" / "evaluators" / f"{key}.md").exists()


def test_examples_are_hebrew():
    hebrew = lambda s: any("א" <= ch <= "ת" for ch in s)
    for key, ex in b.EXAMPLES.items():
        for e in ex.values():
            assert hebrew(e.get("agent", "") + e.get("summary", "")), key


@pytest.mark.parametrize("name", ["README.md", *[f"{k}.md" for k in BUILTIN]])
def test_docs_on_disk_are_up_to_date(name):
    rendered = b.render_all()[name]
    on_disk = (ROOT / "docs" / "evaluators" / name).read_text(encoding="utf-8")
    assert on_disk == rendered, f"docs/evaluators/{name} is stale: run python3 tools/build_evaluator_docs.py"


def test_deterministic_examples_are_what_the_evaluator_really_says():
    """Repetition never calls a model, so its documented verdicts must be exact."""
    from evals.evaluators.repetition import RepetitionEvaluator
    from evals.types import Turn

    for label, e in b.EXAMPLES["repetition"].items():
        r = RepetitionEvaluator().evaluate(Turn(agent_output=e["agent"], user_input=e["user"], history=e.get("history", [])))
        assert (r.label, r.explanation) == (label, e["explanation"])
