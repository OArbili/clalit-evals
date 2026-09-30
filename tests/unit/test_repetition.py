"""Unit tests for the deterministic Repetition evaluator."""

import pytest

from evals import vocab
from evals.evaluators.repetition import RepetitionEvaluator, normalize, sentences
from evals.types import Turn


@pytest.mark.parametrize("raw, expected", [
    ("Bring your card.", "bring your card"),
    ("- Bring your card", "bring your card"),
    ("* Bring your card", "bring your card"),
    ("• כרטיס מגנטי ", "כרטיס מגנטי"),
    ("1) Bring  your   card!!", "bring your card"),
    ("2. Bring your card;", "bring your card"),
    ("  Hello, world?  ", "hello, world"),
])
def test_normalize(raw, expected):
    assert normalize(raw) == expected


def test_sentences_split_on_terminal_punctuation_and_newlines():
    text = "First one. Second one!\nThird one\n\n- fourth\n- fourth"
    assert sentences(text) == ["first one", "second one", "third one", "fourth", "fourth"]


def test_sentences_drop_empty_fragments():
    assert sentences("...\n\n  \n") == []


def test_sentences_hebrew():
    assert sentences("קבע תור היום. אם יש קושי בנשימה, פנה למיון.") == ["קבע תור היום", "אם יש קושי בנשימה, פנה למיון"]


class TestEvaluate:
    ev = RepetitionEvaluator()

    def test_name(self):
        assert self.ev.name == "Repetition"

    def test_intra_response_repeat_fails(self):
        r = self.ev.evaluate(Turn(agent_output="Book today. Rest well. Book today."))
        assert r.label == vocab.LABEL_FAIL and r.score == 0.0
        assert 'The sentence "book today" occurs twice in this response.' == r.explanation

    def test_repeated_bullet_fails(self):
        r = self.ev.evaluate(Turn(agent_output="- כרטיס מגנטי\n- הפניה\n- כרטיס מגנטי"))
        assert r.label == vocab.LABEL_FAIL and "כרטיס מגנטי" in r.explanation

    def test_cross_turn_repeat_fails_and_names_the_earlier_turn(self):
        t = Turn(agent_output="Please book an appointment. Thanks.",
                 history=[("user", "hi"), ("assistant", "Please book an appointment."), ("user", "ok")])
        r = self.ev.evaluate(t)
        assert r.label == vocab.LABEL_FAIL and "in an earlier agent turn" in r.explanation

    def test_user_echo_is_not_repetition(self):
        t = Turn(agent_output="I need a doctor.", history=[("user", "I need a doctor.")])
        r = self.ev.evaluate(t)
        assert r.label == vocab.LABEL_PASS and "cross-turn" in r.explanation

    def test_near_duplicate_passes(self):
        t = Turn(agent_output="The clinic is open on Friday 8:00 to 12:00.",
                 history=[("assistant", "The clinic is open Sunday to Thursday 8:00 to 19:00.")])
        assert self.ev.evaluate(t).label == vocab.LABEL_PASS

    def test_case_and_punctuation_do_not_hide_a_repeat(self):
        assert self.ev.evaluate(Turn(agent_output="Book today!  book today.")).label == vocab.LABEL_FAIL

    def test_pass_explanation_reports_mode_and_count(self):
        r = self.ev.evaluate(Turn(agent_output="One. Two. Three."))
        assert r.label == vocab.LABEL_PASS and r.explanation == "No sentence is repeated (intra-response comparison over 3 sentences)."
        assert "1 sentence)" in self.ev.evaluate(Turn(agent_output="Only one.")).explanation

    def test_missing_output_is_an_error_not_a_verdict(self):
        r = self.ev.evaluate(Turn(agent_output=""))
        assert r.is_error and r.error_type == vocab.ERR_MISSING_INPUT and r.score is None and r.label is None
