"""Unit tests for SummaryFactCapture / SummaryFactPrecision."""

import json

import anthropic
import httpx2 as httpx
import pytest

from evals import vocab as sc
from evals.evaluators import BUILTIN, build
from evals.evaluators.summary_facts import (
    EXTRACT_PROMPT, EXTRACT_SCHEMA, PROMPT_VERSION_EXTRACT, PROMPT_VERSION_VERIFY, VERIFY_PROMPT, VERIFY_SCHEMA,
    SummaryFactCaptureEvaluator, SummaryFactPrecisionEvaluator,
)
from evals.types import Turn
from helpers import RaisingJudge, make
from stub_client import _Message, _TextBlock

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
REF = "Female, 34. Fever 38.5 since yesterday. Not pregnant."
SUM = "34-year-old woman with fever 38.5 since yesterday. Known diabetic."
FACTS = ["The member is female", "The member is 34", "Fever 38.5", "Since yesterday", "Not pregnant"]
CLAIMS = ["The member is 34", "The member is a woman", "Fever 38.5", "Since yesterday", "Known diabetic"]


class TableJudge:
    """extract: text -> statements; verify: fixed verdict list (or per-target table). Records calls."""

    def __init__(self, facts, verdicts):
        self.facts, self.verdicts, self.calls = facts, verdicts, []

    @property
    def messages(self):
        outer = self

        class M:
            @staticmethod
            def create(**kw):
                outer.calls.append(kw)
                if kw["system"] is EXTRACT_PROMPT:
                    text = kw["messages"][0]["content"].split("<text>\n", 1)[1].rsplit("\n</text>", 1)[0]
                    body = outer.facts(text) if callable(outer.facts) else outer.facts
                    return _Message(content=[_TextBlock(json.dumps({"statements": body}))])
                n = kw["messages"][0]["content"].count("\n") + 1
                v = outer.verdicts(kw) if callable(outer.verdicts) else outer.verdicts
                results = v if isinstance(v, list) and v and isinstance(v[0], dict) else [
                    {"statement": i, "stated": bool(ok), "evidence": "x" if ok else ""} for i, ok in enumerate(v, 1)]
                return _Message(content=[_TextBlock(json.dumps({"results": results}))])

        return M


def turn(**kw):
    kw.setdefault("agent_output", SUM)
    kw.setdefault("reference", REF)
    return Turn(**kw)


class TestRegistry:
    def test_registered_with_spec_defaults(self):
        cap = build("summary_fact_capture")
        pre = build("summary_fact_precision")
        assert isinstance(cap, SummaryFactCaptureEvaluator) and cap.threshold == 0.8
        assert isinstance(pre, SummaryFactPrecisionEvaluator) and pre.threshold == 1.0
        assert (cap.name, pre.name) == ("SummaryFactCapture", "SummaryFactPrecision")

    def test_factory_passes_the_model_and_content_capture_stays_off(self):
        ev = build("summary_fact_capture", judge_model="claude-x")
        assert ev.model == "claude-x" and ev._capture_content is False       # the texts need an instance built with the flag

    def test_threshold_validation(self):
        with pytest.raises(ValueError, match="threshold"):
            SummaryFactCaptureEvaluator(threshold=1.5)


class TestPrompts:
    def test_schemas_are_closed(self):
        assert EXTRACT_SCHEMA["required"] == ["statements"] and EXTRACT_SCHEMA["additionalProperties"] is False
        item = VERIFY_SCHEMA["properties"]["results"]["items"]
        assert item["required"] == ["statement", "stated", "evidence"] and item["additionalProperties"] is False

    def test_prompts_state_the_literal_rule_and_consistency(self):
        assert "Identical texts must yield identical lists" in EXTRACT_PROMPT
        assert '"38.5" and "38" are not' in VERIFY_PROMPT and "Identical inputs must yield identical answers" in VERIFY_PROMPT

    def test_extract_then_verify_with_the_expected_payloads(self):
        j = TableJudge(FACTS, [1, 1, 1, 1, 0])
        SummaryFactCaptureEvaluator(j).evaluate(turn())
        extract, verify = j.calls
        assert extract["system"] is EXTRACT_PROMPT and extract["messages"][0]["content"] == f"<text>\n{REF}\n</text>"
        assert extract["output_config"]["format"]["schema"] is EXTRACT_SCHEMA
        assert verify["system"] is VERIFY_PROMPT and verify["output_config"]["format"]["schema"] is VERIFY_SCHEMA
        listing = "\n".join(f"{i}. {s}" for i, s in enumerate(FACTS, 1))
        assert verify["messages"][0]["content"] == f"<target>\n{SUM}\n</target>\n\n<statements>\n{listing}\n</statements>"

    def test_precision_swaps_source_and_target(self):
        j = TableJudge(CLAIMS, [1, 1, 1, 1, 0])
        SummaryFactPrecisionEvaluator(j).evaluate(turn())
        extract, verify = j.calls
        assert f"<text>\n{SUM}\n</text>" == extract["messages"][0]["content"]
        assert verify["messages"][0]["content"].startswith(f"<target>\n{REF}\n</target>")


class TestScoring:
    def test_capture_ratio_label_explanation_and_attributes(self):
        r = SummaryFactCaptureEvaluator(TableJudge(FACTS, [1, 1, 1, 1, 0])).evaluate(turn())
        assert (r.score, r.label) == (0.8, sc.LABEL_PASS)
        assert r.explanation == "4 of 5 reference facts captured (0.80). Missing: 'Not pregnant'."
        assert r.attributes == {sc.FACTS_TOTAL: 5, sc.FACTS_CAPTURED: 4}

    def test_capture_below_threshold_fails_and_rounds_to_four_places(self):
        facts = FACTS + ["Advised ER", "Advised GP", "No cough", "No rash"]
        r = SummaryFactCaptureEvaluator(TableJudge(facts, [1, 1, 1, 1, 1, 1, 1, 0, 0])).evaluate(turn())
        assert (r.score, r.label) == (0.7778, sc.LABEL_FAIL)
        assert r.explanation.startswith("7 of 9 reference facts captured (0.78). Missing: 'No cough'; 'No rash'.")

    def test_all_captured_wording(self):
        r = SummaryFactCaptureEvaluator(TableJudge(FACTS, [1] * 5)).evaluate(turn())
        assert (r.score, r.label, r.explanation) == (1.0, sc.LABEL_PASS, "All 5 reference facts are captured.")

    def test_precision_default_threshold_is_strict(self):
        r = SummaryFactPrecisionEvaluator(TableJudge(CLAIMS, [1, 1, 1, 1, 0])).evaluate(turn())
        assert (r.score, r.label) == (0.8, sc.LABEL_FAIL)
        assert r.explanation == "4 of 5 claims are supported by the reference (0.80). Unsupported: 'Known diabetic'."
        assert r.attributes == {sc.CLAIMS_TOTAL: 5, sc.CLAIMS_SUPPORTED: 4}
        ok = SummaryFactPrecisionEvaluator(TableJudge(CLAIMS, [1] * 5)).evaluate(turn())
        assert ok.label == sc.LABEL_PASS and ok.explanation == "All 5 claims are supported by the reference."

    def test_custom_threshold(self):
        r = SummaryFactPrecisionEvaluator(TableJudge(CLAIMS, [1, 1, 1, 1, 0]), threshold=0.8).evaluate(turn())
        assert r.label == sc.LABEL_PASS

    def test_capture_content_adds_the_missing_statements(self):
        r = SummaryFactCaptureEvaluator(TableJudge(FACTS, [1, 1, 1, 1, 0]), capture_content=True).evaluate(turn())
        assert r.attributes[sc.FACTS_MISSING] == ["Not pregnant"]
        r = SummaryFactPrecisionEvaluator(TableJudge(CLAIMS, [1, 1, 1, 1, 0]), capture_content=True).evaluate(turn())
        assert r.attributes[sc.CLAIMS_UNSUPPORTED] == ["Known diabetic"]

    def test_result_order_in_the_reply_does_not_matter(self):
        shuffled = [{"statement": 5, "stated": False, "evidence": ""}] + [
            {"statement": i, "stated": True, "evidence": "x"} for i in (3, 1, 4, 2)]
        r = SummaryFactCaptureEvaluator(TableJudge(FACTS, shuffled)).evaluate(turn())
        assert r.score == 0.8 and "Not pregnant" in r.explanation


class TestCache:
    def test_reference_facts_are_extracted_once_per_reference(self):
        j = TableJudge(FACTS, [1] * 5)
        ev = SummaryFactCaptureEvaluator(j)
        ev.evaluate(turn(agent_output="summary one"))
        ev.evaluate(turn(agent_output="summary two"))
        ev.evaluate(turn(agent_output="summary three", reference="Another reference."))
        extracts = [c for c in j.calls if c["system"] is EXTRACT_PROMPT]
        assert [c["messages"][0]["content"] for c in extracts] == [f"<text>\n{REF}\n</text>", "<text>\nAnother reference.\n</text>"]
        assert len(j.calls) == 5   # 2 extractions + 3 verifications

    def test_precision_never_caches(self):
        j = TableJudge(CLAIMS, [1] * 5)
        ev = SummaryFactPrecisionEvaluator(j)
        ev.evaluate(turn()); ev.evaluate(turn())
        assert sum(c["system"] is EXTRACT_PROMPT for c in j.calls) == 2

    def test_cached_list_is_copied(self):
        ev = SummaryFactCaptureEvaluator(TableJudge(FACTS, [1] * 5))
        ev._extract(REF).append("mutated by a caller")
        assert ev._extract(REF) == FACTS                             # the cache hands out copies


class TestInputs:
    never = RaisingJudge(RuntimeError("judge must not be called"))

    @pytest.mark.parametrize("cls", [SummaryFactCaptureEvaluator, SummaryFactPrecisionEvaluator])
    def test_empty_summary_and_missing_reference(self, cls):
        r = cls(self.never).evaluate(turn(agent_output=""))
        assert r.error_type == sc.ERR_MISSING_INPUT and "LLM summary" in r.explanation
        r = cls(self.never).evaluate(turn(reference=None))
        assert r.error_type == sc.ERR_MISSING_INPUT and r.explanation == "the reference summary is absent."

    def test_reference_without_facts(self):
        r = SummaryFactCaptureEvaluator(TableJudge([], [])).evaluate(turn())
        assert r.error_type == sc.ERR_MISSING_INPUT and r.explanation == "the reference contains no checkable facts."
        r = SummaryFactPrecisionEvaluator(TableJudge([], [])).evaluate(turn())
        assert r.error_type == sc.ERR_MISSING_INPUT and r.explanation == "the LLM summary contains no checkable claims."

    def test_blank_statements_are_dropped(self):
        r = SummaryFactCaptureEvaluator(TableJudge(["a", "  ", "b"], [1, 0])).evaluate(turn())
        assert r.attributes[sc.FACTS_TOTAL] == 2

    def test_turn_rows_carry_reference(self):
        t = Turn.coerce({"agent_output": "s", "reference": "r", "response_id": "x"})
        assert t.reference == "r"
        assert Turn.coerce({"agent_output": "s", "reference": float("nan")}).reference is None


class TestErrors:
    @pytest.mark.parametrize("exc, expected", [
        (anthropic.APITimeoutError(request=REQ), sc.ERR_TIMEOUT),
        (anthropic.RateLimitError("slow", response=httpx.Response(429, request=REQ), body=None), sc.ERR_RATE_LIMIT),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None), sc.ERR_PROVIDER),
    ])
    def test_provider_failures(self, exc, expected):
        r = SummaryFactCaptureEvaluator(RaisingJudge(exc)).evaluate(turn())
        assert r.error_type == expected and r.score is None and r.attributes == {}

    @pytest.mark.parametrize("verdicts", [
        [1, 1, 1, 1],                                                          # one result short
        [{"statement": 1, "stated": True, "evidence": ""}] * 5,                # duplicate numbers
        [{"statement": i, "stated": True, "evidence": ""} for i in range(0, 5)],   # out of range
        [{"statement": i, "stated": "yes", "evidence": ""} for i in range(1, 6)],  # non-boolean
        "not a list",
    ])
    def test_malformed_verification(self, verdicts):
        r = SummaryFactCaptureEvaluator(TableJudge(FACTS, lambda kw: verdicts)).evaluate(turn())
        assert r.error_type == sc.ERR_INVALID_RESPONSE

    def test_malformed_extraction(self):
        r = SummaryFactCaptureEvaluator(TableJudge(["ok", 3], [1, 1])).evaluate(turn())
        assert r.error_type == sc.ERR_INVALID_RESPONSE and "list of strings" in r.explanation

    def test_unexpected_exception_propagates(self):
        with pytest.raises(RuntimeError):
            SummaryFactCaptureEvaluator(RaisingJudge(RuntimeError("bug"))).evaluate(turn())


class TestThroughTheFacade:
    def test_verdicts_attributes_and_calls(self):
        j = TableJudge(lambda text: FACTS if text == REF else CLAIMS, [1, 1, 1, 1, 0])
        ev = make(evaluators=[SummaryFactCaptureEvaluator(j, capture_content=True), SummaryFactPrecisionEvaluator(j, capture_content=True)], judge_provider=j)
        vs = ev.evaluate(agent_output=SUM, reference=REF, response_id="msg_1", conversation_id="c1")
        cap, pre = vs["SummaryFactCapture"], vs["SummaryFactPrecision"]
        assert (cap.score, cap.label, pre.score, pre.label) == (0.8, "pass", 0.8, "fail")
        assert cap.attributes == {sc.FACTS_TOTAL: 5, sc.FACTS_CAPTURED: 4, sc.FACTS_MISSING: ["Not pregnant"]}
        assert vs.to_dict()["verdicts"][1]["attributes"][sc.CLAIMS_UNSUPPORTED] == ["Known diabetic"]
        versions = sorted(c.prompt_version for v in vs for c in v.calls)          # extract + verify, per evaluator
        assert versions == [PROMPT_VERSION_EXTRACT, PROMPT_VERSION_EXTRACT, PROMPT_VERSION_VERIFY, PROMPT_VERSION_VERIFY]
