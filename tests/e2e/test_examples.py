"""Every example script in getting_started/ runs offline and prints what its docstring promises."""

import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[2]
EXAMPLES = ROOT / "getting_started"
SCRIPTS = ["simple_example.py", "example.py", "custom_evaluator_example.py", "custom_judge_example.py", "summary_example.py"]


def run(script):
    r = subprocess.run([sys.executable, str(EXAMPLES / script), "--offline"], capture_output=True, text=True,
                       timeout=120, cwd=ROOT, env={"PATH": "", "PYTHONPATH": str(ROOT)})
    assert r.returncode == 0, f"{script} failed:\n{r.stderr[-3000:]}"
    return r.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_example_runs_offline_without_credentials(script):
    run(script)


def test_summary_example_shows_all_four_outcomes():
    out = " ".join(run("summary_example.py").split())   # column padding is not the point
    assert "SummaryFactCapture fail 0.78 7 of 9 reference facts captured (0.78)" in out
    assert "SummaryFactPrecision pass 1.00 All 7 claims are supported by the reference." in out
    assert "SummaryFactCapture pass 1.00 All 9 reference facts are captured." in out
    assert "SummaryFactPrecision fail 0.90 9 of 10 claims are supported by the reference (0.90). Unsupported: 'The member is a known diabetic'." in out
    assert out.count("both pass: no") == 3 and out.count("both pass: yes") == 1
    assert "eval.facts.missing" in out and "eval.claims.unsupported" in out


def test_simple_example_prints_only_the_verdicts():
    out = run("simple_example.py")
    lines = [l for l in out.splitlines() if l.strip()]
    assert lines[0] == "--- 1. the minimum ---" and lines[4] == "--- 2. with every identifier ---"
    for i in (1, 5):                                                  # both calls: the same verdicts, nothing else
        assert lines[i].startswith("Repetition  pass") and lines[i + 1].startswith("Relevance   pass") and lines[i + 2] == "ok: True"
    assert len(lines) == 8                                            # no telemetry, nothing else


