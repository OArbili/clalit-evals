"""End-to-end, offline: the labelled summary dataset through both summary evaluators with a scripted judge."""

import pytest

import synthetic as syn
from evals import Evals
from evals import vocab as sc
from helpers import ScriptedSummaryJudge
from stub_client import StubClient

NAMES = {"capture": "SummaryFactCapture", "precision": "SummaryFactPrecision"}


@pytest.fixture(scope="module")
def cases():
    return syn.load_summaries()


def run(cases, capture_content=False):
    from evals.evaluators import SummaryFactCaptureEvaluator, SummaryFactPrecisionEvaluator
    judge = ScriptedSummaryJudge(cases)
    evaluators = [SummaryFactCaptureEvaluator(capture_content=capture_content), SummaryFactPrecisionEvaluator(capture_content=capture_content)]
    ev = Evals(evaluators=evaluators, judge_provider=judge, strict=True)
    report = [ev.evaluate(row) for row in syn.summary_rows(cases)]
    attrs = [(vs.turn.response_id, v.name, v) for vs in report for v in vs]
    return report, attrs, judge


def test_every_case_reproduces_its_expected_scores_and_labels(cases):
    report, _, _ = run(cases)
    assert len(report) == len(cases)
    for c, vs in zip(cases, report):
        for side, name in NAMES.items():
            v, want = vs[name], c["expected"][side]
            if "error" in want:
                assert v.error_type == want["error"], (c["id"], side)
            else:
                assert v.error_type is None and (v.score, v.label) == (want["score"], want["label"]), (c["id"], side, v)


def test_attributes_on_the_verdicts(cases):
    report, attrs, _ = run(cases, capture_content=True)
    by = {(rid, name): v for rid, name, v in attrs}
    assert len(by) == 2 * len(cases)
    for c, vs in zip(cases, report):
        capv, prev = by[(c["response_id"], NAMES["capture"])], by[(c["response_id"], NAMES["precision"])]
        cap, pre = capv.attributes, prev.attributes
        if "error" in c["expected"]["capture"]:
            assert capv.error_type == "missing_input" and capv.score is None and sc.FACTS_TOTAL not in cap
            continue
        assert cap[sc.FACTS_TOTAL] == len(c["reference_facts"]) and cap[sc.FACTS_CAPTURED] == sum(c["captured"])
        assert list(cap[sc.FACTS_MISSING]) == [f for f, ok in zip(c["reference_facts"], c["captured"]) if not ok]
        assert pre[sc.CLAIMS_TOTAL] == len(c["summary_claims"]) and pre[sc.CLAIMS_SUPPORTED] == sum(c["supported"])
        assert list(pre[sc.CLAIMS_UNSUPPORTED]) == [s for s, ok in zip(c["summary_claims"], c["supported"]) if not ok]
        assert vs.turn.conversation_id == c["conversation_id"]


def test_missing_lists_are_off_by_default(cases):
    _, attrs, _ = run(cases)
    assert not any(sc.FACTS_MISSING in v.attributes or sc.CLAIMS_UNSUPPORTED in v.attributes for _, _, v in attrs)


def test_reference_facts_are_extracted_once_per_reference(cases):
    _, _, judge = run(cases)
    refs = [c["reference"] for c in cases if c["reference"] and c["llm_summary"]]
    extracted_refs = [t for t in judge.extract_calls if t in refs]
    assert sorted(extracted_refs) == sorted(set(refs))          # sum-01 and sum-07 share one reference
    assert len(judge.extract_calls) == len(set(refs)) + len([c for c in cases if c["llm_summary"] and c["reference"]])


def test_dataset_covers_the_spec_cases(cases):
    ids = {c["id"]: c for c in cases}
    assert ids["sum-01"]["expected"] == {"capture": {"score": 0.7778, "label": "fail"}, "precision": {"score": 0.875, "label": "fail"}}
    assert ids["sum-05"]["expected"]["capture"]["label"] == "pass" and ids["sum-05"]["expected"]["precision"]["label"] == "fail"
    assert ids["sum-06"]["expected"]["capture"]["label"] == "fail" and ids["sum-06"]["expected"]["precision"]["label"] == "pass"
    assert ids["sum-04"]["captured"][4] == 0 and ids["sum-04"]["supported"][4] == 0    # contradiction on both sides
    assert {c["lang"] for c in cases} == {"en", "he"}
    for c in cases:
        assert len(c["reference_facts"]) == len(c["captured"]) and len(c["summary_claims"]) == len(c["supported"])
