"""The three clinical evaluators: QuestionsBeforeRouting, RedFlagsRuledOut (protocol lists) and FocusedQuestioning."""

import pytest

from evals import Evals, vocab
from evals.evaluators import (FocusedQuestioningEvaluator, LLMJudgeEvaluator, QuestionsBeforeRoutingEvaluator,
                              RedFlagsRuledOutEvaluator, build)
from evals.types import Turn
from helpers import FixedJudge

ROUTED = Turn(agent_output="קבע תור לרופא המשפחה.", user_input="יש לי כאב ראש")


class NoCall:
    """A client that fails the test if the judge is called."""

    @property
    def messages(self):
        class M:
            @staticmethod
            def create(**kwargs):
                pytest.fail("the judge must not be called for this input")
        return M


class Capture:
    """A client that keeps the prompts it was sent and answers pass."""

    def __init__(self):
        self.seen = []
        self._inner = FixedJudge("pass")

    @property
    def messages(self):
        outer = self

        class M:
            @staticmethod
            def create(**kwargs):
                outer.seen.append(kwargs)
                return outer._inner.messages.create(**kwargs)
        return M


class TestDefaults:
    @pytest.mark.parametrize("key, cls", [("questions_before_routing", QuestionsBeforeRoutingEvaluator),
                                          ("red_flags_ruled_out", RedFlagsRuledOutEvaluator),
                                          ("focused_questioning", FocusedQuestioningEvaluator)])
    def test_registered_on_the_shared_judge_base(self, key, cls):
        ev = build(key, judge_provider=FixedJudge("pass"))
        assert isinstance(ev, cls) and isinstance(ev, LLMJudgeEvaluator)
        assert cls.system_prompt() == ev.system_prompt()          # no protocol given: the class defaults

    def test_default_protocols_are_in_the_prompt(self):
        q = QuestionsBeforeRoutingEvaluator.system_prompt()
        assert all(item in q for item in QuestionsBeforeRoutingEvaluator.default_items)
        assert "emergency care passes" in q and "volunteered counts as collected" in q
        r = RedFlagsRuledOutEvaluator.system_prompt()
        assert all(flag in r for flag in RedFlagsRuledOutEvaluator.default_items)
        assert "reported one of the red flags below as present" in r and "advice, not a question" in r

    def test_the_conversation_is_shown_to_the_judge(self):
        j = Capture()
        t = Turn(agent_output="קבע תור.", user_input="לא", history=[("user", "כאב ראש"), ("assistant", "יש חום?")])
        assert RedFlagsRuledOutEvaluator(j).evaluate(t).label == "pass"
        assert "<conversation_history>" in j.seen[0]["messages"][0]["content"]


class TestProtocol:
    def test_a_custom_list_replaces_the_defaults_and_versions_the_prompt(self):
        ev = QuestionsBeforeRoutingEvaluator(FixedJudge("pass"), required=["pain level 1-10", " fever "])
        prompt = ev.system_prompt()
        assert "pain level 1-10; fever." in prompt and "how severe it is" not in prompt
        assert ev.items == ("pain level 1-10", "fever")
        assert ev.prompt_version.startswith("questions_before_routing/v1+") and len(ev.prompt_version.split("+")[1]) == 8
        assert QuestionsBeforeRoutingEvaluator.prompt_version == "questions_before_routing/v1"      # the class is untouched
        assert "how severe it is" in QuestionsBeforeRoutingEvaluator.system_prompt()

    def test_the_version_follows_the_list(self):
        a = RedFlagsRuledOutEvaluator(red_flags=["chest pain"]).prompt_version
        b = RedFlagsRuledOutEvaluator(red_flags=["chest pain"]).prompt_version
        c = RedFlagsRuledOutEvaluator(red_flags=["chest pain", "fainting"]).prompt_version
        d = RedFlagsRuledOutEvaluator(red_flags=["chest pain"], require_all=True).prompt_version
        assert a == b and len({a, c, d}) == 3

    def test_require_all_changes_the_second_criterion(self):
        some = RedFlagsRuledOutEvaluator().fail_criteria[1]
        every = RedFlagsRuledOutEvaluator(require_all=True).fail_criteria[1]
        assert "no question from the assistant about any red flag" in some
        assert "for at least one red flag in the list above" in every and every != some

    def test_the_custom_version_is_on_the_call_record(self):
        ev = RedFlagsRuledOutEvaluator(red_flags=["chest pain"])
        v = Evals(evaluators=[ev], judge_provider=FixedJudge("fail")).evaluate(ROUTED)["RedFlagsRuledOut"]
        assert v.label == "fail" and v.calls[0].prompt_version == ev.prompt_version

    @pytest.mark.parametrize("bad, exc", [([], ValueError), (["", "  "], ValueError), ("chest pain", TypeError)])
    def test_a_bad_list_is_rejected_at_construction(self, bad, exc):
        with pytest.raises(exc):
            RedFlagsRuledOutEvaluator(red_flags=bad)


class TestFocusedQuestioning:
    def test_no_question_passes_without_a_call(self):
        r = FocusedQuestioningEvaluator(NoCall()).evaluate(ROUTED)
        assert r.label == "pass" and r.calls == () and "no question" in r.explanation

    def test_a_question_goes_to_the_judge(self):
        t = Turn(agent_output="האם הברך נפוחה? ואיזה סניף הכי קרוב אליך?", user_input="כואבת לי הברך")
        r = FocusedQuestioningEvaluator(FixedJudge("fail")).evaluate(t)
        assert r.label == "fail" and len(r.calls) == 1 and r.calls[0].prompt_version == "focused_questioning/v1"


class TestInputs:
    @pytest.mark.parametrize("cls", [QuestionsBeforeRoutingEvaluator, RedFlagsRuledOutEvaluator, FocusedQuestioningEvaluator])
    def test_missing_reply_is_an_error_not_a_call(self, cls):
        r = cls(NoCall()).evaluate(Turn(agent_output="", user_input="יש לי כאב ראש"))
        assert r.error_type == vocab.ERR_MISSING_INPUT and r.score is None


class TestInstanceAwarePrompt:
    def test_an_instance_override_reaches_the_prompt_and_the_class_keeps_its_own(self):
        class J(LLMJudgeEvaluator):
            name = "J"; prompt_version = "j/v1"; definition = "d"; fail_criteria = ("class criterion",)

        ev = J(FixedJudge("pass"))
        ev.fail_criteria = ("instance criterion",)
        assert "- instance criterion" in ev.system_prompt() and "- class criterion" in J.system_prompt()

    def test_a_classmethod_override_still_works(self):
        class J(LLMJudgeEvaluator):
            name = "J"; prompt_version = "j/v1"

            @classmethod
            def system_prompt(cls):
                return "my own rubric"

        j = Capture()
        J(j).evaluate(ROUTED)
        assert j.seen[0]["system"] == "my own rubric" and J.system_prompt() == "my own rubric"
