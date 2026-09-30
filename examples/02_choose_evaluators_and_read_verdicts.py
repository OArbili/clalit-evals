"""Choose evaluators, read every field of a verdict, see how errors come back.

    pip install "clalit-evals @ git+ssh://git@github.com/OArbili/clalit-evals.git@v0.2.0"
    python3 02_choose_evaluators_and_read_verdicts.py            # the judge from the environment
    python3 02_choose_evaluators_and_read_verdicts.py --offline
"""

import json
import sys

from evals import Evals


def fake_judge(*, system, messages, schema, **_):
    return {"verdict": "pass", "explanation": "Nothing in the reply matches a fail criterion."}


judge = {"judge_provider": fake_judge, "judge_model": "fake-judge"} if "--offline" in sys.argv else {}
ev = Evals(evaluators=["repetition", "relevance", "empathy", "context_retention"], **judge)

verdicts = ev.evaluate(
    user_input="I'm worried, the fever is not going down.",
    agent_output="Book a same-day appointment. Book a same-day appointment.",   # a repeated sentence
    history=[("user", "Hi, I have had a fever since yesterday."), ("assistant", "How high is it?")],
    response_id="msg_2", conversation_id="conv_1",
    only=["repetition", "empathy"],                               # a subset of the configured evaluators
)

print(verdicts, "\n")
print("ok:", verdicts.ok, "| failed:", [v.name for v in verdicts.failed], "| errors:", [v.name for v in verdicts.errors])

v = verdicts["empathy"]                                           # by name, case-insensitive
print("\nEmpathy:", v.label, v.score, v.passed, "|", v.explanation)
for call in v.calls:                                              # the record: one JudgeCall per model call
    print("  judge call:", call.provider, call.model, call.prompt_version, f"{call.duration_s:.2f}s",
          "tokens", call.input_tokens, call.output_tokens)

# An empty reply is not an exception: it is a verdict with error_type set and no score.
bad = ev.evaluate(user_input="Should I come in?", agent_output="")
print("\nempty reply ->", {x.name: x.status for x in bad})

print("\nto_dict():")
print(json.dumps(verdicts.to_dict(), indent=2, ensure_ascii=False)[:600], "...")
