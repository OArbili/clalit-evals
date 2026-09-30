"""Helpers shared by unit and e2e tests: an Evals on the stub judge, and judge doubles."""

from __future__ import annotations

import json
import re

from evals import Evals
from stub_client import StubClient, _Message, _TextBlock


def make(**kw) -> Evals:
    """An Evals on the stub judge, strict by default."""
    kw.setdefault("strict", True)
    kw.setdefault("judge_provider", StubClient("judge"))
    return Evals(**kw)


class RaisingJudge:
    """A client whose messages.create raises a given exception."""

    def __init__(self, exc: BaseException) -> None:
        self._exc = exc

    @property
    def messages(self):
        exc = self._exc

        class M:
            @staticmethod
            def create(**kwargs):
                raise exc

        return M


class FixedJudge:
    """A client that always returns one verdict."""

    def __init__(self, verdict: str, explanation: str = "fixed", stop_reason: str = "end_turn") -> None:
        self._body = json.dumps({"verdict": verdict, "explanation": explanation})
        self._stop = stop_reason

    @property
    def messages(self):
        body, stop = self._body, self._stop

        class M:
            @staticmethod
            def create(**kwargs):
                return _Message(content=[_TextBlock(body)] if stop != "refusal" else [], stop_reason=stop)

        return M


class ScriptedJudge:
    """Replays ground truth: verdict = expected[dimension] for the reply in the prompt.

    Parses the dimension from the system prompt and the reply from the rendered
    user message, so it exercises the real prompt layout end to end. Counts calls.
    """

    _DIM = re.compile(r"Dimension under judgement: (\w+)")
    _REPLY = re.compile(r"<assistant_reply>\n(.*?)\n</assistant_reply>", re.S)
    _KEY = {"Repetition": "repetition", "Relevance": "relevance", "Naturalness": "naturalness", "Empathy": "empathy",
            "Attentiveness": "attentiveness", "Coherence": "coherence", "ContextRetention": "context_retention",
            "AppropriateQuestioning": "appropriate_questioning", "Adaptability": "adaptability", "NonRobotic": "non_robotic"}

    def __init__(self, expected_turns: list[dict]) -> None:
        self._by_reply = {t["agent_output"]: t for t in expected_turns}
        self.calls: list[tuple[str, str]] = []

    @property
    def messages(self):
        outer = self

        class M:
            @staticmethod
            def create(**kwargs):
                dim = outer._DIM.search(kwargs["system"]).group(1)
                reply = outer._REPLY.search(kwargs["messages"][0]["content"]).group(1)
                outer.calls.append((dim, reply))
                t = outer._by_reply.get(reply)
                label = (t or {}).get("expected", {}).get(outer._KEY[dim]) or "pass"
                why = (t or {}).get("why", "")
                return _Message(content=[_TextBlock(json.dumps({"verdict": label, "explanation": f"scripted: {why}"}))])

        return M


class ScriptedSummaryJudge:
    """Replays the summary dataset's labelled facts, claims and verdicts from the real prompts.

    Extraction is keyed by the source text; verification by (target text, statements).
    Counts calls per prompt so tests can check the reference cache.
    """

    _TEXT = re.compile(r"<text>\n(.*?)\n</text>", re.S)
    _TARGET = re.compile(r"<target>\n(.*?)\n</target>", re.S)
    _LIST = re.compile(r"<statements>\n(.*?)\n</statements>", re.S)

    def __init__(self, cases: list[dict]) -> None:
        self._facts: dict[str, list[str]] = {}
        self._verdicts: dict[tuple[str, tuple[str, ...]], list[bool]] = {}
        for c in cases:
            if c["reference"]:
                self._facts[c["reference"]] = c["reference_facts"]
            if c["llm_summary"]:
                self._facts[c["llm_summary"]] = c["summary_claims"]
            if c["reference_facts"]:
                self._verdicts[(c["llm_summary"], tuple(c["reference_facts"]))] = [bool(x) for x in c["captured"]]
            if c["summary_claims"]:
                self._verdicts[(c["reference"], tuple(c["summary_claims"]))] = [bool(x) for x in c["supported"]]
        self.extract_calls: list[str] = []
        self.verify_calls: list[tuple[str, tuple[str, ...]]] = []

    @property
    def messages(self):
        outer = self

        class M:
            @staticmethod
            def create(**kwargs):
                content = kwargs["messages"][0]["content"]
                if kwargs["system"].startswith("You extract"):
                    text = outer._TEXT.search(content).group(1)
                    outer.extract_calls.append(text)
                    return _Message(content=[_TextBlock(json.dumps({"statements": outer._facts[text]}))])
                target = outer._TARGET.search(content).group(1)
                statements = tuple(line.split(". ", 1)[1] for line in outer._LIST.search(content).group(1).splitlines())
                outer.verify_calls.append((target, statements))
                stated = outer._verdicts[(target, statements)]
                results = [{"statement": i, "stated": ok, "evidence": s if ok else ""}
                           for i, (s, ok) in enumerate(zip(statements, stated), 1)]
                return _Message(content=[_TextBlock(json.dumps({"results": results}))])

        return M
