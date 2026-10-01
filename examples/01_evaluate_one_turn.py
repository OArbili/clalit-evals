"""Evaluate one agent turn and print the verdicts. Needs only the package:

    pip install "clalit-evals @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"
    export ANTHROPIC_API_KEY=...      # or OPENAI_API_KEY (+ OPENAI_BASE_URL for a gateway) and EVALS_JUDGE_MODEL
    python3 01_evaluate_one_turn.py

    python3 01_evaluate_one_turn.py --offline     # no key: a function stands in for the judge
"""

import sys

from evals import Evals


def fake_judge(*, system, messages, schema, **_):
    """A judge for offline runs: answers 'pass' in the shape every evaluator asks for."""
    return {"verdict": "pass", "explanation": "Nothing in the reply matches a fail criterion."}


judge = {"judge_provider": fake_judge, "judge_model": "fake-judge"} if "--offline" in sys.argv else {}
ev = Evals(**judge)                    # default evaluators: repetition + relevance; the judge from the environment

verdicts = ev.evaluate(
    user_input="I've had a fever of 38.5 and a sore throat since yesterday. Should I come in?",
    agent_output="Book a same-day appointment at your clinic. If breathing becomes difficult, go to the ER.",
    response_id="msg_01Ab3Cd5Ef",      # optional ids pass through to the verdicts
)
print(verdicts)                        # one line per evaluator: name, pass / fail / error, explanation
print("ok:", verdicts.ok)
