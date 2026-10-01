"""Behavioural tests for Evals (core.py): verdicts, error semantics, records. The package emits no telemetry."""

from __future__ import annotations

import json
import logging

import anthropic
import pytest

from evals import Evals, JudgeCall, Turn, Verdicts
from evals import vocab
from helpers import RaisingJudge, make
from stub_client import StubClient


# --- inputs -----------------------------------------------------------------------------


def test_dict_rows_and_keyword_fields_are_equivalent():
    ev = make()
    a = ev.evaluate({"user_input": "q", "agent_output": "a", "ignored_column": 1}, only="repetition")
    b = ev.evaluate(user_input="q", agent_output="a", only="repetition")
    c = ev.evaluate(Turn(agent_output="a", user_input="q"), only="repetition")
    strip = lambda v: {k: x for k, x in v.to_dict().items() if k not in ("started_at", "ended_at")}
    assert strip(a[0]) == strip(b[0]) == strip(c[0])


def test_evaluate_does_not_mutate_the_callers_turn():
    turn = Turn(agent_output="a.", response_id="r")
    v = make().evaluate(turn, tenant_id="clalit", only="repetition")
    assert turn.tenant_id is None          # caller state untouched: the override went on a copy
    assert v.turn.tenant_id == "clalit"    # the turn the evaluators saw


def test_stored_ids_are_kept_on_the_verdict_only_when_both_are_present():
    tid, sid = "4bf92f3577b34da6a3ce929d0e0e4736", "00f067aa0ba902b7"
    ev = make()
    assert ev.evaluate(agent_output="a", trace_id=tid, span_id=sid, only="repetition")[0].trace_id == tid
    v = ev.evaluate(agent_output="a", trace_id=tid, only="repetition")[0]
    assert v.trace_id is None and v.span_id is None


def test_nan_and_pandas_missing_values_read_as_absent():
    nan = float("nan")
    v = make(strict=False).evaluate({"user_input": "q", "agent_output": "a.", "response_id": nan, "model": nan,
                                     "history": nan, "trace_id": nan, "span_id": nan}, only="repetition")
    assert v[0].passed is True
    assert v.turn.response_id is None and v.turn.model is None and v.turn.history == []


def test_a_bad_row_becomes_error_verdicts_and_the_batch_continues():
    blocks = [{"type": "text", "text": "not flattened"}]
    rows = [
        {"user_input": "q", "agent_output": "fine."},
        {"user_input": "q", "agent_output": blocks, "response_id": "bad-1"},                 # non-text output
        {"user_input": "q", "agent_output": "x.", "history": [{"role": "user", "content": blocks}]},  # non-text history
        {"user_input": "q", "agent_output": "also fine."},
    ]
    ev = make(strict=False)
    results = [ev.evaluate(r, only="repetition") for r in rows]
    assert [vs[0].passed for vs in results] == [True, None, None, True]
    bad = results[1][0]
    assert bad.error_type == vocab.ERR_MISSING_INPUT and "agent_output must be plain text" in bad.explanation
    assert results[1].turn.response_id == "bad-1"     # ids salvaged for write-back
    with pytest.raises(TypeError):
        make(strict=True).evaluate(agent_output=blocks)


# --- error semantics ------------------------------------------------------------------


def test_missing_input_omits_score_and_sets_error_type():
    v = make().evaluate(agent_output="", only="repetition")
    assert v[0].error_type == vocab.ERR_MISSING_INPUT
    assert v[0].passed is None and v[0].score is None and v[0].label is None and v.ok is False
    assert v[0].calls == ()


def test_judge_timeout_becomes_a_timeout_verdict_with_its_call_record():
    v = make(judge_provider=RaisingJudge(anthropic.APITimeoutError(request=None))).evaluate(user_input="q", agent_output="a", only="relevance")
    assert v[0].error_type == vocab.ERR_TIMEOUT
    (call,) = v[0].calls
    assert call.error_type == vocab.ERR_TIMEOUT and call.model == "claude-opus-4-8" and call.ended_at >= call.started_at


def test_unexpected_evaluator_exception_is_isolated_unless_strict():
    from evals.evaluators import Evaluator

    class Broken(Evaluator):
        name = "Broken"

        def evaluate(self, turn):
            raise RuntimeError("boom")

    v = make(strict=False, evaluators=[Broken()]).evaluate(agent_output="a")
    assert v[0].error_type == vocab.ERR_PROVIDER and "RuntimeError: boom" in v[0].explanation
    with pytest.raises(RuntimeError):
        make(strict=True, evaluators=[Broken()]).evaluate(agent_output="a")


def test_evaluator_returning_garbage_is_isolated():
    from evals.evaluators import Evaluator

    class Garbage(Evaluator):
        name = "Garbage"

        def evaluate(self, turn):
            return {"passed": True}

    v = make(strict=False, evaluators=[Garbage()]).evaluate(agent_output="a.")
    assert v[0].error_type == vocab.ERR_INVALID_RESPONSE and "expected EvalResult" in v[0].explanation


# --- the record every verdict carries ---------------------------------------------------


def test_verdict_carries_timing_and_judge_calls():
    v = make().evaluate(user_input="q", agent_output="a.", response_id="r")
    rep, rel = v["Repetition"], v["Relevance"]
    assert rep.calls == () and rep.started_at <= rep.ended_at
    (call,) = rel.calls
    assert isinstance(call, JudgeCall)
    assert (call.provider, call.model, call.prompt_version) == ("anthropic", "claude-opus-4-8", "relevance/v2")
    assert (call.input_tokens, call.output_tokens, call.finish_reason, call.error_type) == (412, 38, "stop", None)
    assert rel.started_at <= call.started_at <= call.ended_at <= rel.ended_at


def test_verdicts_repr_and_to_dict():
    v = make().evaluate(user_input="q", agent_output="a", response_id="r", conversation_id="c")
    assert isinstance(v, Verdicts)
    text = repr(v)
    assert "Repetition" in text and "Relevance" in text and "pass" in text
    d = v.to_dict()
    assert d["response_id"] == "r" and d["conversation_id"] == "c"
    assert {x["name"] for x in d["verdicts"]} == {"Repetition", "Relevance"}
    rel = [x for x in d["verdicts"] if x["name"] == "Relevance"][0]
    assert rel["calls"][0]["model"] == "claude-opus-4-8" and rel["started_at"] <= rel["ended_at"]
    json.dumps(d)  # serialisable for write-back
    with pytest.raises(KeyError):
        v["Nope"]


def test_only_rejects_unconfigured_evaluator():
    with pytest.raises(ValueError):
        make().evaluate(agent_output="a", only="toxicity")


# --- construction ------------------------------------------------------------------------


def test_only_a_judge_provider_is_needed(monkeypatch):
    for var in ("ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL", "EVALS_PROVIDER"):
        monkeypatch.delenv(var, raising=False)
    ev = Evals(judge_provider=StubClient("judge"))
    assert ev.judge_provider.name == "anthropic" and ev.judge_model == "claude-opus-4-8"
    assert ev.evaluate(agent_output="Rest.", user_input="hi", response_id="r1").ok
    with pytest.raises(RuntimeError, match="no LLM provider configured"):
        Evals()                                                  # nothing given, nothing in the environment


def test_the_package_has_no_state_beyond_the_evaluators():
    """No tenant, no group, no destination, no agent: those belong to the turn or to the host."""
    for kw in ({"tenant_id": "clalit"}, {"telemetry": "console"}, {"service": "x"}, {"provider": StubClient("triage")}, {"integrations": []}):
        with pytest.raises(TypeError):
            Evals(judge_provider=StubClient("judge"), **kw)
    ev = make()
    assert not hasattr(ev, "chat") and not hasattr(ev, "captured") and not hasattr(ev, "close") and not hasattr(ev, "evaluate_many")
    assert [e.name for e in ev.evaluators] == ["Repetition", "Relevance"]


def test_default_client_sends_the_workspace_header_when_configured(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
    monkeypatch.setenv("ANTHROPIC_WORKSPACE_ID", "wrkspc_123")
    assert Evals().judge_provider.client.default_headers.get("anthropic-workspace-id") == "wrkspc_123"
    monkeypatch.delenv("ANTHROPIC_WORKSPACE_ID")
    assert "anthropic-workspace-id" not in Evals().judge_provider.client.default_headers


def test_logging_reports_setup_and_errors(caplog):
    caplog.set_level(logging.DEBUG, logger="evals")
    ev = make(strict=False)
    assert any("Evals ready: evaluators=['Repetition', 'Relevance']" in r.message for r in caplog.records)
    ev.evaluate(agent_output="", response_id="r9", only="repetition")
    warnings = [r for r in caplog.records if r.levelno == logging.WARNING]
    assert any("Repetition on response_id=r9: error missing_input" in r.message for r in warnings)
    assert any(r.name == "evals" and r.levelno == logging.DEBUG and "evaluated response_id=r9" in r.message for r in caplog.records)


# --- evaluator package ------------------------------------------------------------------


def test_repetition_catches_newline_and_bullet_repeats():
    from evals.evaluators import RepetitionEvaluator

    rep = RepetitionEvaluator()
    assert rep.evaluate(Turn(agent_output="Book now\nBook now")).label == "fail"
    assert rep.evaluate(Turn(agent_output="- Book now\n- Book now")).label == "fail"
    assert rep.evaluate(Turn(agent_output="1. Book now\n2. Book now")).label == "fail"
    assert rep.evaluate(Turn(agent_output="Book now\nThen rest.")).label == "pass"


def test_finish_reason_map_covers_every_sdk_stop_reason():
    from typing import get_args

    import anthropic.types as t

    from evals.providers.anthropic import STOP_REASONS

    values = set(get_args(t.message.StopReason))
    assert values <= set(STOP_REASONS)
    assert set(STOP_REASONS.values()) <= set(vocab.FINISH_REASONS)


def test_each_evaluator_lives_in_its_own_module():
    from evals.evaluators import relevance, repetition
    from evals.evaluators.base import Evaluator

    assert repetition.RepetitionEvaluator.__module__ == "evals.evaluators.repetition"
    assert relevance.RelevanceEvaluator.__module__ == "evals.evaluators.relevance"
    assert issubclass(repetition.RepetitionEvaluator, Evaluator)


def test_names_resolve_through_the_registry_and_unknown_names_fail_early():
    from evals.evaluators import BUILTIN, build

    assert {"repetition", "relevance"} <= set(BUILTIN) and len(BUILTIN) == 15
    assert build("Repetition").name == "Repetition"
    with pytest.raises(ValueError, match="unknown evaluator"):
        Evals(evaluators=["toxicity"], judge_provider=StubClient("judge"))


def test_a_custom_evaluator_from_its_own_module_runs_end_to_end(tmp_path, monkeypatch):
    import textwrap

    (tmp_path / "wordcount.py").write_text(textwrap.dedent('''
        from evals.evaluators.base import EvalResult, Evaluator

        class WordCount(Evaluator):
            name = "WordCount"

            def __init__(self, max_words=5):
                self.max_words = max_words

            def evaluate(self, turn):
                n = len(turn.agent_output.split())
                if n > self.max_words:
                    return EvalResult.failed(self.name, f"{n} words, limit {self.max_words}.")
                return EvalResult.passed(self.name, f"{n} words.")
    '''))
    monkeypatch.syspath_prepend(str(tmp_path))
    from wordcount import WordCount  # noqa: E402

    v = make(evaluators=["repetition", WordCount(max_words=3)]).evaluate(agent_output="one two three four.", response_id="r1")
    assert v["WordCount"].passed is False and v["Repetition"].passed is True
    assert v["WordCount"].calls == () and v["WordCount"].ended_at >= v["WordCount"].started_at


# --- judge dimensions -----------------------------------------------------------------------


DIMENSIONS = ["naturalness", "empathy", "attentiveness", "coherence", "context_retention",
              "appropriate_questioning", "adaptability", "non_robotic"]


class NeverCall:
    """A judge client that fails the test if the model is called."""

    @property
    def messages(self):
        class M:
            @staticmethod
            def create(**kwargs):
                pytest.fail("the judge must not be called for this input")
        return M


def test_all_dimensions_are_registered_and_share_the_judge_base():
    from evals.evaluators import BUILTIN, LLMJudgeEvaluator, build

    assert set(DIMENSIONS) <= set(BUILTIN)
    for key in DIMENSIONS + ["relevance"]:
        e = build(key, judge_provider=StubClient("judge"))
        assert isinstance(e, LLMJudgeEvaluator)
        prompt = type(e).system_prompt()
        assert e.name in prompt and "Identical inputs must yield identical verdicts" in prompt
        assert type(e).fail_criteria and type(e).definition


def test_every_dimension_runs_through_the_shared_path_with_the_stub():
    v = make(evaluators=["relevance"] + DIMENSIONS).evaluate(
        user_input="I've had a fever since yesterday, I'm worried.",
        agent_output="Is the fever above 38.5?", history=[("user", "hi"), ("assistant", "hello")])
    assert len(v) == 9 and v.ok
    assert all(len(x.calls) == 1 for x in v)
    assert {x.calls[0].prompt_version for x in v} >= {"relevance/v2"}


def test_prechecks_answer_without_calling_the_judge():
    v = make(evaluators=["appropriate_questioning", "context_retention"], judge_provider=NeverCall()).evaluate(
        user_input="fever", agent_output="Book a same-day appointment.")   # no '?', no history
    assert v["AppropriateQuestioning"].passed is True
    assert "no follow-up question" in v["AppropriateQuestioning"].explanation
    assert v["ContextRetention"].passed is True
    assert "no information to retain" in v["ContextRetention"].explanation
    assert all(x.calls == () for x in v)   # no judge call was made


def test_judge_fail_verdict_maps_to_score_zero():
    from stub_client import _Message, _TextBlock

    class FailingJudge:
        class messages:
            @staticmethod
            def create(**kwargs):
                return _Message(content=[_TextBlock(json.dumps({"verdict": "fail", "explanation": "Canned redirect."}))])

    v = make(evaluators=["non_robotic"], judge_provider=FailingJudge()).evaluate(user_input="fever", agent_output="Please book an appointment.")
    assert v[0].passed is False and v[0].score == 0.0 and v[0].explanation == "Canned redirect."


def test_judge_explanations_with_escaped_hebrew_are_normalised():
    from evals.evaluators._judge import _unescape
    from stub_client import _Message, _TextBlock

    assert _unescape("quoted \\u05d0\\u05e0\\u05d9 here") == "quoted אני here"
    assert _unescape("plain text") == "plain text"

    class EscapingJudge:
        class messages:
            @staticmethod
            def create(**kwargs):
                body = json.dumps({"verdict": "pass", "explanation": "says \\u05e9\\u05dc\\u05d5\\u05dd"})
                return _Message(content=[_TextBlock(body)])

    v = make(evaluators=["empathy"], judge_provider=EscapingJudge()).evaluate(user_input="q", agent_output="a.")
    assert v[0].explanation == "says שלום"
