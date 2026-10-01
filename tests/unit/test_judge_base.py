"""Unit tests for the shared LLM-judge base and the registered judge evaluators' contracts."""

import json

import anthropic
import httpx2 as httpx  # anthropic 1.3.0 is built on httpx2
import pytest

from evals import vocab
from evals.evaluators import BUILTIN, LLMJudgeEvaluator, build
from evals.evaluators._judge import SCHEMA, TEMPLATE, _unescape
from evals.evaluators.appropriate_questioning import AppropriateQuestioningEvaluator
from evals.evaluators.context_retention import ContextRetentionEvaluator
from evals.evaluators.relevance import RelevanceEvaluator
from evals.types import Turn
from helpers import FixedJudge, RaisingJudge
from stub_client import _Message, _TextBlock

JUDGES = [n for n in BUILTIN if isinstance(build(n), LLMJudgeEvaluator)]
REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")


def turn(**kw):
    kw.setdefault("agent_output", "Book an appointment with your family doctor.")
    kw.setdefault("user_input", "I have a sore throat.")
    return Turn(**kw)


class TestRegistry:
    def test_fifteen_evaluators_registered(self):
        assert list(BUILTIN) == ["repetition", "relevance", "naturalness", "empathy", "attentiveness", "coherence",
                                 "context_retention", "appropriate_questioning", "adaptability", "non_robotic",
                                 "questions_before_routing", "red_flags_ruled_out", "focused_questioning",
                                 "summary_fact_capture", "summary_fact_precision"]
        assert len(JUDGES) == 12

    def test_build_is_case_insensitive_and_passes_judge_deps(self):
        client = FixedJudge("pass")
        ev = build("Relevance", judge_provider=client, judge_model="claude-x")
        assert isinstance(ev, RelevanceEvaluator) and ev.model == "claude-x" and ev.provider.client is client

    def test_build_ignores_unknown_deps(self):
        assert build("empathy", judge_provider=FixedJudge("pass"), future_dep=1).name == "Empathy"

    def test_build_unknown_name(self):
        with pytest.raises(ValueError, match="unknown evaluator 'nope'"):
            build("nope")

    @pytest.mark.parametrize("name", JUDGES)
    def test_every_judge_declares_definition_criteria_and_version(self, name):
        cls = type(build(name))
        assert issubclass(cls, LLMJudgeEvaluator)
        assert cls.definition.strip() and len(cls.fail_criteria) >= 1
        assert cls.prompt_version != LLMJudgeEvaluator.prompt_version
        assert cls.name and cls.name[0].isupper()

    def test_prompt_versions_are_distinct(self):
        versions = [type(build(n)).prompt_version for n in JUDGES]
        assert len(set(versions)) == len(versions)


class TestPrompt:
    def test_schema_is_a_closed_pass_fail_object(self):
        assert SCHEMA["properties"]["verdict"]["enum"] == ["pass", "fail"]
        assert SCHEMA["required"] == ["verdict", "explanation"] and SCHEMA["additionalProperties"] is False

    @pytest.mark.parametrize("name", JUDGES)
    def test_system_prompt_is_built_from_the_one_template(self, name):
        cls = type(build(name))
        p = cls.system_prompt()
        assert f"Dimension under judgement: {cls.name}\n" in p
        assert f"Definition: {cls.definition}" in p
        for c in cls.fail_criteria:
            assert f"- {c}" in p
        assert "Identical inputs must yield identical verdicts" in p
        assert "answer \"pass\" and say so" in p

    def test_pass_note_is_appended_after_the_decision_rule(self):
        class J(LLMJudgeEvaluator):
            name = "J"; definition = "d"; fail_criteria = ("c1",); pass_note = "  Extra note. "
        assert 'Otherwise answer "pass". Extra note.\n' in J.system_prompt()

    def test_template_has_no_dimension_specific_text(self):
        for word in ("Relevance", "Empathy", "Naturalness"):
            assert word not in TEMPLATE

    def test_user_prompt_layout_with_and_without_history(self):
        ev = RelevanceEvaluator(FixedJudge("pass"))
        t = turn(history=[("user", "hi"), ("assistant", "hello")])
        text = ev.user_prompt(t)
        assert text.startswith("<conversation_history>\nuser: hi\nassistant: hello\n</conversation_history>\n\n<user_message>")
        assert text.endswith("<assistant_reply>\nBook an appointment with your family doctor.\n</assistant_reply>")
        assert "<conversation_history>" not in ev.user_prompt(turn())

    def test_user_prompt_omits_history_when_the_dimension_does_not_use_it(self):
        class J(RelevanceEvaluator):
            uses_history = False
        assert "<conversation_history>" not in J(FixedJudge("pass")).user_prompt(turn(history=[("user", "hi")]))


class TestVerdicts:
    def test_pass_and_fail_map_to_score_and_label(self):
        assert RelevanceEvaluator(FixedJudge("pass", "fine")).evaluate(turn()).score == 1.0
        r = RelevanceEvaluator(FixedJudge("fail", "off topic")).evaluate(turn())
        assert (r.label, r.score, r.explanation) == (vocab.LABEL_FAIL, 0.0, "off topic")

    def test_stray_unicode_escapes_in_explanation_are_decoded(self):
        r = RelevanceEvaluator(FixedJudge("pass", "quoted \\u05e9\\u05dc\\u05d5\\u05dd here")).evaluate(turn())
        assert r.explanation == "quoted שלום here"

    def test_unescape_leaves_ordinary_text_alone(self):
        assert _unescape("no escapes \\n here") == "no escapes \\n here"

    def test_judge_provider_is_called_with_structured_output_and_timeout(self):
        seen = {}

        class Spy:
            class messages:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return _Message(content=[_TextBlock(json.dumps({"verdict": "pass", "explanation": "ok"}))])

        RelevanceEvaluator(Spy(), model="claude-m", effort="high", timeout=7.5).evaluate(turn())
        assert seen["model"] == "claude-m" and seen["timeout"] == 7.5
        assert seen["output_config"] == {"format": {"type": "json_schema", "schema": SCHEMA}, "effort": "high"}
        assert seen["messages"][0]["role"] == "user" and "<assistant_reply>" in seen["messages"][0]["content"]


class TestErrors:
    @pytest.mark.parametrize("exc, expected", [
        (anthropic.APITimeoutError(request=REQ), vocab.ERR_TIMEOUT),
        (anthropic.RateLimitError("slow down", response=httpx.Response(429, request=REQ), body=None), vocab.ERR_RATE_LIMIT),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None), vocab.ERR_PROVIDER),
        (anthropic.APIConnectionError(request=REQ), vocab.ERR_PROVIDER),
    ])
    def test_anthropic_exceptions_map_to_the_spec_error_types(self, exc, expected):
        r = RelevanceEvaluator(RaisingJudge(exc)).evaluate(turn())
        assert r.is_error and r.error_type == expected and r.score is None and r.label is None

    def test_refusal_is_a_provider_error(self):
        r = RelevanceEvaluator(FixedJudge("pass", stop_reason="refusal")).evaluate(turn())
        assert r.error_type == vocab.ERR_PROVIDER and "declined" in r.explanation

    @pytest.mark.parametrize("body", ["not json", '{"verdict": "pass"}', '{"verdict": "maybe", "explanation": "x"}',
                                      '{"verdict": "pass", "explanation": 5}', '[]'])
    def test_malformed_judge_replies_are_invalid_response(self, body):
        class Bad:
            class messages:
                @staticmethod
                def create(**kw):
                    return _Message(content=[_TextBlock(body)])

        r = RelevanceEvaluator(Bad()).evaluate(turn())
        assert r.error_type == vocab.ERR_INVALID_RESPONSE

    def test_unexpected_exception_propagates(self):
        with pytest.raises(RuntimeError):
            RelevanceEvaluator(RaisingJudge(RuntimeError("bug"))).evaluate(turn())

    def test_missing_inputs_never_reach_the_judge(self):
        ev = RelevanceEvaluator(RaisingJudge(RuntimeError("must not be called")))
        assert ev.evaluate(turn(agent_output="")).explanation == "gen_ai.output.messages is absent or empty."
        assert ev.evaluate(turn(user_input=None)).explanation == "the last user message is absent or empty."
        assert ev.evaluate(turn(user_input="")).error_type == vocab.ERR_MISSING_INPUT


class TestPrechecks:
    never = RaisingJudge(RuntimeError("judge must not be called"))

    def test_no_question_passes_without_a_call(self):
        r = AppropriateQuestioningEvaluator(self.never).evaluate(turn(agent_output="Rest and drink."))
        assert r.label == vocab.LABEL_PASS and "asks no follow-up question" in r.explanation

    def test_question_goes_to_the_judge(self):
        r = AppropriateQuestioningEvaluator(FixedJudge("fail", "checklist")).evaluate(turn(agent_output="Fever? Rash?"))
        assert r.label == vocab.LABEL_FAIL

    def test_no_earlier_user_message_passes_without_a_call(self):
        ev = ContextRetentionEvaluator(self.never)
        assert ev.evaluate(turn()).label == vocab.LABEL_PASS
        assert ev.evaluate(turn(history=[("assistant", "Hello!")])).label == vocab.LABEL_PASS

    def test_earlier_user_message_goes_to_the_judge(self):
        ev = ContextRetentionEvaluator(FixedJudge("fail", "re-asked"))
        assert ev.evaluate(turn(history=[("user", "I am not pregnant"), ("assistant", "ok")])).label == vocab.LABEL_FAIL

    def test_precheck_runs_after_the_missing_input_check(self):
        r = AppropriateQuestioningEvaluator(self.never).evaluate(turn(agent_output="Rest.", user_input=None))
        assert r.error_type == vocab.ERR_MISSING_INPUT


class TestBind:
    def test_judge_instance_passed_to_evals_uses_its_judge_provider(self):
        from helpers import make

        class Politeness(RelevanceEvaluator):
            name = "Politeness"
            prompt_version = "politeness/v1"

        vs = make(evaluators=[Politeness()]).evaluate(agent_output="Rest.", user_input="hi", response_id="r1")   # no provider given to the instance
        assert vs["Politeness"].label == "pass"           # answered by the Evals stub judge, not the environment
        (call,) = vs["Politeness"].calls
        assert call.prompt_version == "politeness/v1" and call.provider == "anthropic"

    def test_bind_never_overrides_what_the_instance_was_given(self):
        own = FixedJudge("fail", "own client")
        ev = RelevanceEvaluator(own, model="claude-own")
        ev.bind(judge_provider=FixedJudge("pass"), judge_model="other")
        assert ev.provider.client is own and ev.model == "claude-own"

    def test_registry_factory_may_ignore_the_dependencies(self):
        from helpers import make

        class Short(RelevanceEvaluator):
            name = "Short"

        BUILTIN["short_test"] = lambda **_: Short()
        try:
            assert make(evaluators=["short_test"]).evaluate(agent_output="Rest.", user_input="hi", response_id="r1")["Short"].label == "pass"
        finally:
            del BUILTIN["short_test"]
