"""Unit tests for evals.types: pure data, no OpenTelemetry or Anthropic involved."""

import pytest

from evals.types import Message, Turn, Verdict, Verdicts, as_message


class TestAsMessage:
    def test_message_passes_through(self):
        m = Message("user", "hi")
        assert as_message(m) is m

    @pytest.mark.parametrize("item", [("user", "hi"), ["user", "hi"]])
    def test_pair(self, item):
        assert as_message(item) == Message("user", "hi")

    @pytest.mark.parametrize("item", [{"role": "assistant", "content": "x"}, {"role": "assistant", "text": "x"}])
    def test_mapping_content_or_text(self, item):
        assert as_message(item) == Message("assistant", "x")

    def test_mapping_prefers_content_over_text(self):
        assert as_message({"role": "user", "content": "a", "text": "b"}).text == "a"

    def test_mapping_without_role(self):
        with pytest.raises(ValueError, match="no 'role'"):
            as_message({"content": "x"})

    def test_mapping_without_text(self):
        with pytest.raises(ValueError, match="neither"):
            as_message({"role": "user"})

    def test_content_blocks_rejected_with_hint(self):
        with pytest.raises(TypeError, match="flatten content-block lists"):
            as_message({"role": "user", "content": [{"type": "text", "text": "x"}]})

    def test_bare_string_rejected(self):
        with pytest.raises(TypeError, match="not a single string"):
            as_message("hello")

    def test_unknown_shape_rejected(self):
        with pytest.raises(TypeError, match="cannot interpret"):
            as_message(42)


class NAType:  # the class NAME is what Turn checks, so this mimics pandas.NA without importing pandas
    pass


class TestTurn:
    def test_nan_and_na_read_as_missing(self):
        t = Turn(agent_output="x", user_input=float("nan"), response_id=NAType(), model=None)
        assert t.user_input is None and t.response_id is None and t.model is None

    def test_missing_agent_output_becomes_empty(self):
        assert Turn(agent_output=None).agent_output == ""
        assert Turn(agent_output=float("nan")).agent_output == ""

    def test_non_text_field_rejected(self):
        with pytest.raises(TypeError, match="response_id must be plain text"):
            Turn(agent_output="x", response_id=123)

    def test_content_block_list_rejected_with_hint(self):
        with pytest.raises(TypeError, match="flatten content blocks"):
            Turn(agent_output=[{"type": "text", "text": "x"}])

    def test_history_string_rejected(self):
        with pytest.raises(TypeError, match="not a single string"):
            Turn(agent_output="x", history="user: hi")

    def test_history_is_coerced_and_nan_history_is_empty(self):
        t = Turn(agent_output="x", history=[("user", "a"), {"role": "assistant", "content": "b"}])
        assert t.history == [Message("user", "a"), Message("assistant", "b")]
        assert Turn(agent_output="x", history=float("nan")).history == []

    def test_sampled_is_bool(self):
        assert Turn(agent_output="x", sampled=0).sampled is False
        assert Turn(agent_output="x").sampled is True

    def test_text_alias(self):
        assert Turn(agent_output="reply").text == "reply"

    def test_coerce_row_ignores_unknown_keys_and_overrides_win(self):
        row = {"agent_output": "a", "user_input": "u", "score_from_export": 1, "response_id": "r1"}
        t = Turn.coerce(row, response_id="override")
        assert (t.agent_output, t.user_input, t.response_id) == ("a", "u", "override")

    def test_coerce_row_without_agent_output(self):
        assert Turn.coerce({"user_input": "u"}).agent_output == ""

    def test_coerce_turn_passthrough_and_copy_on_override(self):
        t = Turn(agent_output="a", response_id="r")
        assert Turn.coerce(t) is t
        t2 = Turn.coerce(t, response_id="r2")
        assert t2 is not t and t2.response_id == "r2" and t.response_id == "r"

    def test_coerce_rejects_other_types(self):
        with pytest.raises(TypeError, match="expected a Turn or a mapping"):
            Turn.coerce("agent output")


def _v(name, label=None, error=None, score=None):
    if error:
        return Verdict(name, None, None, "detail", error)
    return Verdict(name, label, 1.0 if label == "pass" else 0.0, f"{name} says {label}", None)


class TestVerdict:
    def test_passed_is_derived_from_label(self):
        assert _v("A", "pass").passed is True
        assert _v("A", "fail").passed is False
        assert _v("A", error="timeout").passed is None

    def test_status_strings(self):
        assert _v("A", "pass").status == "pass"
        assert _v("A", error="timeout").status == "error:timeout"

    def test_to_dict_keys(self):
        d = _v("A", "fail").to_dict()
        assert d == {"name": "A", "label": "fail", "score": 0.0, "passed": False,
                     "explanation": "A says fail", "error_type": None, "attributes": {},
                     "started_at": None, "ended_at": None, "calls": []}


class TestVerdicts:
    @pytest.fixture
    def vs(self):
        turn = Turn(agent_output="x", response_id="r1", conversation_id="c1", trace_id="a" * 32, span_id="b" * 16)
        return Verdicts([_v("Repetition", "pass"), _v("Relevance", "fail"), _v("Empathy", error="timeout")], turn=turn)

    def test_index_by_position_and_case_insensitive_name(self, vs):
        assert vs[0].name == "Repetition"
        assert vs["relevance"].name == "Relevance"
        with pytest.raises(KeyError, match="no verdict named"):
            vs["Missing"]

    def test_ok_failed_errors(self, vs):
        assert vs.ok is False
        assert [v.name for v in vs.failed] == ["Relevance"]
        assert [v.name for v in vs.errors] == ["Empathy"]

    def test_ok_requires_every_pass_and_no_errors(self):
        t = Turn(agent_output="x")
        assert Verdicts([_v("A", "pass")], turn=t).ok is True
        assert Verdicts([_v("A", "pass"), _v("B", error="timeout")], turn=t).ok is False

    def test_to_dict_carries_ids(self, vs):
        d = vs.to_dict()
        assert (d["response_id"], d["conversation_id"], d["trace_id"], d["span_id"]) == ("r1", "c1", "a" * 32, "b" * 16)
        assert [v["name"] for v in d["verdicts"]] == ["Repetition", "Relevance", "Empathy"]

    def test_repr_lists_every_verdict(self, vs):
        text = repr(vs)
        assert "Repetition" in text and "error:timeout" in text and "Relevance says fail" in text
        assert "<table>" in vs._repr_html_()

    def test_empty_repr(self):
        assert repr(Verdicts([], turn=Turn(agent_output="x"))) == "Verdicts()"
