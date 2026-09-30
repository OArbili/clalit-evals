"""An Anthropic-SDK-shaped stand-in (`.messages.create`), so the examples run without an API key.

The SDK wraps it in `AnthropicProvider` automatically; any provider adapter works the same way.

It returns fixed responses shaped like the real SDK's Message object. The agent
reply deliberately repeats a sentence, so the example shows one failing and one
passing evaluation. Not part of the library — test scaffolding only.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

TRIAGE_REPLY = (
    "Book a same-day appointment at your Clalit clinic. "
    "If your breathing becomes difficult or the fever passes 39.5, go to the emergency room. "
    "Book a same-day appointment at your Clalit clinic."
)

JUDGE_REPLY = {
    "verdict": "pass",
    "explanation": (
        "The answer recommends a care pathway for the fever and sore throat "
        "the member described."
    ),
}


@dataclass
class _TextBlock:
    text: str
    type: str = "text"


@dataclass
class _Usage:
    input_tokens: int = 412
    output_tokens: int = 38


@dataclass
class _Message:
    content: list[_TextBlock]
    id: str = "msg_01Ab3Cd5Ef"
    model: str = "claude-opus-4-8"
    stop_reason: str = "end_turn"
    usage: _Usage = field(default_factory=_Usage)


class _Messages:
    def __init__(self, kind: str) -> None:
        self._kind = kind

    def create(self, **kwargs) -> _Message:
        if self._kind == "judge":
            return _Message(content=[_TextBlock(json.dumps(JUDGE_REPLY))])
        if self._kind == "summary_judge":
            return _Message(content=[_TextBlock(json.dumps(_naive_summary_judge(kwargs)))])
        return _Message(content=[_TextBlock(TRIAGE_REPLY)])


# Four LLM summaries of one reference, as the stub judge replays them (summary_example.py --offline).
# The third is the worked example of docs/SUMMARY_FACT_CAPTURE.md §8.
SUMMARY_REFERENCE = (
    "Female, 34. Fever 38.5 since yesterday and sore throat. No breathing difficulty. "
    "Not pregnant. Advised same-day appointment with family doctor; ER if breathing worsens."
)
_REFERENCE_FACTS = [
    "The member is female", "The member is 34 years old", "The member has a fever of 38.5",
    "The fever started yesterday", "The member has a sore throat", "The member has no breathing difficulty",
    "The member is not pregnant", "The member was advised to book a same-day appointment with the family doctor",
    "The member was advised to go to the ER if breathing worsens",
]
SUMMARY_CASES = [
    {
        "title": "facts missed, nothing invented",
        "summary": "34-year-old woman with fever 38.5 since yesterday and throat pain, breathing normally. "
                   "Advised a same-day GP appointment.",
        "captured": [1, 1, 1, 1, 1, 1, 0, 1, 0],
        "claims": ["The member is 34 years old", "The member is a woman", "The member has a fever of 38.5",
                   "The fever started yesterday", "The member has throat pain", "The member is breathing normally",
                   "The member was advised a same-day GP appointment"],
        "supported": [1, 1, 1, 1, 1, 1, 1],
    },
    {
        "title": "nothing missed, a claim invented",
        "summary": "Woman, 34. Fever 38.5 since yesterday with a sore throat; no breathing difficulty; not pregnant. "
                   "Known diabetic. Same-day family-doctor appointment advised, ER if breathing worsens.",
        "captured": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "claims": ["The member is a woman", "The member is 34 years old", "The member has a fever of 38.5",
                   "The fever started yesterday", "The member has a sore throat", "The member has no breathing difficulty",
                   "The member is not pregnant", "The member is a known diabetic",
                   "A same-day family-doctor appointment was advised", "The member was advised to go to the ER if breathing worsens"],
        "supported": [1, 1, 1, 1, 1, 1, 1, 0, 1, 1],
    },
    {
        "title": "facts missed and a claim invented (the spec's worked example)",
        "summary": "34-year-old woman with fever 38.5 since yesterday and throat pain, breathing normally. "
                   "Known diabetic. Advised a same-day GP appointment.",
        "captured": [1, 1, 1, 1, 1, 1, 0, 1, 0],
        "claims": ["The member is 34 years old", "The member is a woman", "The member has a fever of 38.5",
                   "The fever started yesterday", "The member has throat pain", "The member is breathing normally",
                   "The member is a known diabetic", "The member was advised a same-day GP appointment"],
        "supported": [1, 1, 1, 1, 1, 1, 0, 1],
    },
    {
        "title": "faithful and complete",
        "summary": "Woman, 34. Fever 38.5 since yesterday with a sore throat; no breathing difficulty; not pregnant. "
                   "Same-day family-doctor appointment advised, ER if breathing worsens.",
        "captured": [1, 1, 1, 1, 1, 1, 1, 1, 1],
        "claims": ["The member is a woman", "The member is 34 years old", "The member has a fever of 38.5",
                   "The fever started yesterday", "The member has a sore throat", "The member has no breathing difficulty",
                   "The member is not pregnant", "A same-day family-doctor appointment was advised",
                   "The member was advised to go to the ER if breathing worsens"],
        "supported": [1, 1, 1, 1, 1, 1, 1, 1, 1],
    },
]
SUMMARY_LLM = SUMMARY_CASES[2]["summary"]   # the spec's worked example

_EXTRACTED = {SUMMARY_REFERENCE: _REFERENCE_FACTS, **{c["summary"]: c["claims"] for c in SUMMARY_CASES}}
_STATED = {}   # (target text, statement) -> stated
for _c in SUMMARY_CASES:
    _STATED.update({(_c["summary"], f): bool(ok) for f, ok in zip(_REFERENCE_FACTS, _c["captured"])})
    _STATED.update({(SUMMARY_REFERENCE, s): bool(ok) for s, ok in zip(_c["claims"], _c["supported"])})


def _naive_summary_judge(kwargs: dict) -> dict:
    """A deterministic stand-in for the summary fact judge (summary_example.py --offline).

    For the texts above it replays the reviewer's facts, claims and verdicts.
    For any other text it splits sentences and matches them verbatim, which is
    crude: the real judge recognises paraphrases.
    """
    import re
    text = kwargs["messages"][0]["content"]
    if kwargs["system"].startswith("You extract"):
        body = re.search(r"<text>\n(.*?)\n</text>", text, re.S).group(1)
        scripted = _EXTRACTED.get(body)
        if scripted is not None:
            return {"statements": list(scripted)}
        parts = [p.strip() for p in re.split(r"(?<=[.!?])\s+|\n+", body) if p.strip()]
        return {"statements": [p.rstrip(".") for p in parts]}
    target = re.search(r"<target>\n(.*?)\n</target>", text, re.S).group(1)
    listing = re.search(r"<statements>\n(.*?)\n</statements>", text, re.S).group(1)
    results = []
    for line in listing.splitlines():
        n, s = line.split(". ", 1)
        stated = _STATED.get((target, s), s in target)
        results.append({"statement": int(n), "stated": stated, "evidence": s if stated else ""})
    return {"results": results}


class StubClient:
    def __init__(self, kind: str = "triage") -> None:
        self.messages = _Messages(kind)
