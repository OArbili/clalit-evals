"""Three clinical checks on a triage conversation, with your own protocol.

    pip install "clalit-evals @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"
    python3 07_clinical_evaluators.py --offline     # a function stands in for the judge, no key
    python3 07_clinical_evaluators.py               # the judge from the environment

QuestionsBeforeRouting  were all the questions asked before the routing?
RedFlagsRuledOut        were red flags ruled out before a non-urgent routing?
FocusedQuestioning      did the agent ask several unrelated questions in one message?

The first two take a protocol. Their defaults are generic; a project passes the list its clinicians
approved, and that list is recorded in the prompt version of every judge call.
"""

import re
import sys

from evals import Evals
from evals.evaluators import FocusedQuestioningEvaluator, QuestionsBeforeRoutingEvaluator, RedFlagsRuledOutEvaluator

# The protocol for a headache complaint, as a clinical lead might write it.
REQUIRED = ["when the headache started", "pain level from 1 to 10", "nausea or sensitivity to light"]
RED_FLAGS = ["sudden weakness or numbness", "difficulty speaking", "confusion", "the worst headache of the member's life"]

CASES = {
    "routed at once": dict(
        user_input="יש לי כאב ראש",
        agent_output="קבע תור לרופא המשפחה."),
    "questioned, red flags ruled out, then routed": dict(
        history=[("user", "יש לי כאב ראש מאתמול"),
                 ("assistant", "מצטער לשמוע. כמה חזק הכאב מ-1 עד 10, והאם יש גם בחילה או רגישות לאור?"),
                 ("user", "בערך 5, בלי בחילה"),
                 ("assistant", "תודה. האם הופיעו חולשה פתאומית, קושי בדיבור או בלבול, והאם זה הכאב החזק ביותר שהיה לך?")],
        user_input="לא, שום דבר כזה",
        agent_output="לפי מה שתיארת, כדאי לקבוע תור לרופא המשפחה בימים הקרובים. אם הכאב מחמיר בפתאומיות, פנה למיון."),
    "unrelated questions in one message": dict(
        user_input="כואבת לי הברך כבר שבוע",
        agent_output="האם הברך נפוחה? ואיזה סניף הכי קרוב אליך? והאם עדכנת את פרטי התשלום שלך?"),
}

# --- an offline judge: a reviewer's answers for these exact replies ------------------------------------------
_ANSWERS = {   # (dimension, reply) -> (verdict, explanation); anything not listed passes
    ("QuestionsBeforeRouting", CASES["routed at once"]["agent_output"]):
        ("fail", "The reply routes to an appointment although nothing was asked about onset, pain level or nausea."),
    ("RedFlagsRuledOut", CASES["routed at once"]["agent_output"]):
        ("fail", "The reply routes to a routine appointment and no red flag was asked about."),
    ("FocusedQuestioning", CASES["unrelated questions in one message"]["agent_output"]):
        ("fail", "The reply asks about the knee, a branch and payment details in one message."),
}


def fake_judge(*, system, messages, schema, **_):
    dimension = re.search(r"Dimension under judgement: (\w+)", system).group(1)
    reply = re.search(r"<assistant_reply>\n(.*?)\n</assistant_reply>", messages[0].text, re.S).group(1)
    verdict, why = _ANSWERS.get((dimension, reply), ("pass", "Nothing in the reply matches a fail criterion."))
    return {"verdict": verdict, "explanation": why}


judge = {"judge_provider": fake_judge, "judge_model": "fake-judge"} if "--offline" in sys.argv else {}
ev = Evals(evaluators=[QuestionsBeforeRoutingEvaluator(required=REQUIRED),
                       RedFlagsRuledOutEvaluator(red_flags=RED_FLAGS, require_all=True),
                       FocusedQuestioningEvaluator()], **judge)

for title, turn in CASES.items():
    verdicts = ev.evaluate(**turn)
    print(f"--- {title} ---")
    print(f"agent: {turn['agent_output']}")
    print(verdicts, "\n")

# The protocol travels with every verdict: the prompt version on the judge call names the list that was used.
v = ev.evaluate(**CASES["routed at once"])["RedFlagsRuledOut"]
print("The protocol is on the record:", v.calls[0].prompt_version)
