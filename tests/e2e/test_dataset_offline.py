"""Every dimension on the labelled synthetic dataset (tests/data/), with a judge that replays the labels.

The labels are the ground truth of the evaluators; the scripted judge answers each dimension
the way the dataset says, so this checks the plumbing end to end -- turn construction, pre-checks,
the shared judge path, the verdicts -- not the judge's own accuracy (the live tests do that).
"""

import json

import pytest

import synthetic as syn
from evals import Evals
from evals.evaluators import build
from helpers import ScriptedJudge

DISPLAY = {d: build(d).name for d in syn.DIMENSIONS}


@pytest.fixture(scope="module")
def dataset():
    convs = syn.load()
    return convs, syn.expected_turns(convs)


def test_every_label_is_reproduced(dataset):
    convs, expected = dataset
    ev = Evals(evaluators=syn.DIMENSIONS, judge_provider=ScriptedJudge(expected), strict=True)
    report = [ev.evaluate(row) for row in syn.evals_rows(convs)]
    assert len(report) == len(expected) and not any(vs.errors for vs in report)
    for vs, e in zip(report, expected):
        assert vs.turn.response_id == e["response_id"] and vs.turn.conversation_id == e["conversation_id"]
        for d in syn.DIMENSIONS:
            want = e["expected"][d]
            if want is not None:
                assert vs[DISPLAY[d]].label == want, (e["response_id"], d)
        for v in vs:
            assert v.score == (1.0 if v.label == "pass" else 0.0)


def test_pass_rates_match_ground_truth(dataset):
    convs, expected = dataset
    ev = Evals(evaluators=syn.DIMENSIONS, judge_provider=ScriptedJudge(expected))
    report = [ev.evaluate(row) for row in syn.evals_rows(convs)]
    for d in syn.DIMENSIONS:
        passed = sum(vs[DISPLAY[d]].passed is True for vs in report)
        n_fail = sum(e["expected"][d] == "fail" for e in expected)
        assert passed / len(report) == pytest.approx(1 - n_fail / len(expected))


def test_dataset_is_well_formed(dataset):
    convs, expected = dataset
    assert len(convs) >= 15 and len(expected) >= 20
    for e in expected:
        assert set(e["expected"]) == set(syn.DIMENSIONS)
        assert all(v in ("pass", "fail", None) for v in e["expected"].values())
    for d in syn.DIMENSIONS:   # every dimension has clear positives and negatives
        assert sum(e["expected"][d] == "fail" for e in expected) >= 2
        assert sum(e["expected"][d] == "pass" for e in expected) >= 5
    json.dumps(convs, ensure_ascii=False)
