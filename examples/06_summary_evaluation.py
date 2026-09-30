"""An LLM-written summary against a human-written reference, fact by fact: the two summary evaluators.

    pip install "clalit-evals @ git+https://github.com/OArbili/clalit-evals.git@v0.2.0"
    python3 06_summary_evaluation.py --offline     # a function stands in for the judge, no key
    python3 06_summary_evaluation.py               # the judge from the environment (4 judge calls per summary)

SummaryFactCapture asks "did the LLM summary miss anything the reference says?" -- recall, passes at >= 0.8.
SummaryFactPrecision asks "did it say anything the reference does not support?" -- precision, passes only at 1.0.
Each is two judge calls: extract the statements from one text, then verify each against the other. The
reference's facts are extracted once and cached, so every summary of one conversation is measured against
the same list. The LLM summary is `agent_output`; the human one is `reference`.
"""

import sys

from evals import Evals
from evals.evaluators import SummaryFactCaptureEvaluator, SummaryFactPrecisionEvaluator

REFERENCE = ("Female, 34. Fever 38.5 since yesterday and sore throat. No breathing difficulty. Not pregnant. "
             "Advised same-day appointment with family doctor; ER if breathing worsens.")

SUMMARIES = {
    "facts missed":      "34-year-old woman with fever 38.5 since yesterday and throat pain, breathing normally. "
                         "Advised a same-day GP appointment.",
    "a claim invented":  "Woman, 34. Fever 38.5 since yesterday with a sore throat; no breathing difficulty; not pregnant. "
                         "Known diabetic. Same-day family-doctor appointment advised, ER if breathing worsens.",
    "faithful, complete": "Woman, 34. Fever 38.5 since yesterday with a sore throat; no breathing difficulty; not pregnant. "
                         "Same-day family-doctor appointment advised, ER if breathing worsens.",
}


# --- an offline judge: a reviewer's facts and answers for these exact texts -------------------------------
# A real judge reads the texts. This one replays a table, in the two shapes the evaluators ask for:
# extraction -> {"statements": [...]}, verification -> {"results": [{"statement": n, "stated": bool, "evidence": ""}]}.

REFERENCE_FACTS = [
    "The member is female", "The member is 34 years old", "The member has a fever of 38.5",
    "The fever started yesterday", "The member has a sore throat", "The member has no breathing difficulty",
    "The member is not pregnant", "The member was advised a same-day appointment with the family doctor",
    "The member was advised to go to the ER if breathing worsens",
]
CLAIMS = {   # per summary: the claims a reviewer extracts from it, and which of them the reference supports
    "facts missed": (["The member is 34 years old", "The member is a woman", "The member has a fever of 38.5",
                      "The fever started yesterday", "The member has throat pain", "The member is breathing normally",
                      "The member was advised a same-day GP appointment"], [1, 1, 1, 1, 1, 1, 1]),
    "a claim invented": (["The member is a woman", "The member is 34 years old", "The member has a fever of 38.5",
                          "The fever started yesterday", "The member has a sore throat", "The member has no breathing difficulty",
                          "The member is not pregnant", "The member is a known diabetic",
                          "A same-day family-doctor appointment was advised", "The member was advised to go to the ER if breathing worsens"],
                         [1, 1, 1, 1, 1, 1, 1, 0, 1, 1]),
    "faithful, complete": (["The member is a woman", "The member is 34 years old", "The member has a fever of 38.5",
                            "The fever started yesterday", "The member has a sore throat", "The member has no breathing difficulty",
                            "The member is not pregnant", "A same-day family-doctor appointment was advised",
                            "The member was advised to go to the ER if breathing worsens"], [1] * 9),
}
CAPTURED = {   # per summary: which reference facts it states
    "facts missed": [1, 1, 1, 1, 1, 1, 0, 1, 0],
    "a claim invented": [1] * 9,
    "faithful, complete": [1] * 9,
}
_EXTRACTED = {REFERENCE: REFERENCE_FACTS, **{SUMMARIES[k]: c for k, (c, _) in CLAIMS.items()}}
_STATED = {}
for _k, (_claims, _ok) in CLAIMS.items():
    _STATED.update({(REFERENCE, c): bool(o) for c, o in zip(_claims, _ok)})           # claim vs the reference
    _STATED.update({(SUMMARIES[_k], f): bool(o) for f, o in zip(REFERENCE_FACTS, CAPTURED[_k])})   # fact vs the summary


def fake_judge(*, system, messages, schema, **_):
    text = messages[0].text
    if "statements" in schema["properties"]:                                    # extraction
        body = text.split("<text>")[1].split("</text>")[0].strip()
        return {"statements": _EXTRACTED[body]}
    target = text.split("<target>")[1].split("</target>")[0].strip()           # verification
    listing = text.split("<statements>")[1].split("</statements>")[0].strip().splitlines()
    return {"results": [{"statement": int(line.split(". ", 1)[0]), "stated": _STATED[(target, line.split(". ", 1)[1])],
                         "evidence": ""} for line in listing]}


judge = {"judge_provider": fake_judge, "judge_model": "fake-judge"} if "--offline" in sys.argv else {}

# capture_content=True makes the evaluators list the missing facts / unsupported claims themselves (member
# text, so off by default); threshold= moves the pass line. Both are per instance, so build the instances.
ev = Evals(evaluators=[SummaryFactCaptureEvaluator(threshold=0.8, capture_content=True),
                       SummaryFactPrecisionEvaluator(capture_content=True)], **judge)

print(f"Reference (human):\n  {REFERENCE}\n")
for i, (title, summary) in enumerate(SUMMARIES.items(), 1):
    verdicts = ev.evaluate(agent_output=summary, reference=REFERENCE,
                           response_id=f"msg_summary_{i:02d}", conversation_id="conv_8b21f4")
    print(f"--- {i}. {title} ---\nLLM summary:\n  {summary}")
    for v in verdicts:
        print(f"  {v.name:<21} {v.label:<5} {v.score:.2f}   {v.explanation}")
    print(f"  both pass: {'yes' if verdicts.ok else 'no'}")

# The record on the last one: the counts as attributes, and the two judge calls per evaluator (the
# reference extraction is cached, so the capture evaluator made only one call this time).
print("\nThe record on case 3:")
for v in verdicts:
    print(f"  {v.name}: attributes={dict(v.attributes)}")
    print(f"    judge calls: {[c.prompt_version for c in v.calls]}")
