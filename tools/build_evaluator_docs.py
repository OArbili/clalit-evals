"""Generate docs/evaluators/*.md: one README per evaluator, rules from the code, Hebrew examples by hand.

    python3 tools/build_evaluator_docs.py

The definition, fail criteria, pass note, pre-check and prompt version are read
from the evaluator classes, so the READMEs cannot drift from the prompts that
run. The examples below are hand-written; tests/unit/test_evaluator_docs.py
fails when the files on disk differ from what this script renders.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run from a clone without pip install -e .

from pathlib import Path

from evals import vocab
from evals.evaluators import BUILTIN, LLMJudgeEvaluator, build
from evals.evaluators.summary_facts import _SummaryFactsEvaluator

OUT = Path(__file__).resolve().parents[1] / "docs" / "evaluators"
LINK = "https://example.org/book"
REDIRECT = ("בנושא הזה אני עדיין לא יכול לטפל ישירות, אבל רופא/ת המשפחה יוכלו לעזור לך להמשיך. "
            "אפשר להמשיך לקביעת תור כאן: " + LINK)

# One fail and one pass example per evaluator, Hebrew, text only. `history` lists
# earlier turns as (role, text); `explanation` is what the evaluator writes.
EXAMPLES: dict[str, dict[str, dict]] = {
    "repetition": {
        "fail": dict(user="יש לי חום 38.5 וכאב גרון מאתמול",
                     agent="קבע תור לרופא המשפחה עוד היום. אם יש קושי בנשימה, פנה למיון. קבע תור לרופא המשפחה עוד היום.",
                     explanation='The sentence "קבע תור לרופא המשפחה עוד היום" occurs twice in this response.'),
        "pass": dict(history=[("user", "מה שעות הפעילות של המרפאה?"), ("assistant", "המרפאה פתוחה בימים א׳–ה׳ בין 8:00 ל-19:00.")],
                     user="ובשישי?", agent="המרפאה פתוחה בימי שישי בין 8:00 ל-12:00.",
                     explanation="No sentence is repeated (cross-turn comparison over 1 sentence).",
                     note="A near-duplicate of the earlier sentence is not a repeat: only the exact same normalised sentence fails."),
    },
    "relevance": {
        "fail": dict(user="איבדתי את כרטיס החבר שלי, מה עושים?",
                     agent="בעיות רפואיות תלויות בגיל ובמין, ולכן לא ניתן לתאר בעיות של מטופל אחר דרך המשתמש שלך.",
                     explanation="No sentence of the reply addresses the lost membership card ('איבדתי את כרטיס החבר שלי'); it is about describing another patient's problems."),
        "pass": dict(user="יש לי כאבים ברגליים כבר שבוע",
                     agent="כאבי רגליים שנמשכים שבוע כדאי לבדוק. אני ממליץ לקבוע תור לרופא המשפחה השבוע.",
                     explanation="The reply addresses the leg pain the user described ('כאבי רגליים שנמשכים שבוע') with a care recommendation."),
    },
    "naturalness": {
        "fail": dict(user="תודה",
                     agent="בהתאם לנהלי הקופה, לאחר סיום הפנייה יש לוודא כי כל הפרטים הרלוונטיים תועדו כנדרש, וכי המבוטח קיבל את מלוא המידע הנוגע לזכויותיו, לרבות אפשרות הגשת ערר על פי הנוהל התקף. לפרטים נוספים ניתן לפנות למוקד השירות בשעות הפעילות.",
                     explanation="The reply is a form letter of bureaucratic boilerplate ('בהתאם לנהלי הקופה', 'לרבות אפשרות הגשת ערר') with no wording specific to the user's 'תודה'."),
        "pass": dict(user="אני מפחדת, יש לי דימום חזק ואני בהריון",
                     agent="אני שומע שאת מפחדת, ודימום חזק בהריון הוא סיבה טובה לא לחכות. פני עכשיו למיון נשים הקרוב, ואם את לא יכולה להגיע בעצמך התקשרי *2700 והם יכוונו אותך.",
                     explanation="The wording responds to this specific message ('אני שומע שאת מפחדת, ודימום חזק בהריון') rather than a stock formula."),
    },
    "empathy": {
        "fail": dict(user="אני מפחדת, יש לי דימום חזק ואני בהריון",
                     agent="לקביעת תור לרופא נשים יש להיכנס לאתר.",
                     explanation="The user said she is afraid ('אני מפחדת') and the reply does not mention or respond to it at all."),
        "pass": dict(user="אני מפחדת, יש לי דימום חזק ואני בהריון",
                     agent="אני שומע שאת מפחדת, ודימום חזק בהריון הוא סיבה טובה לא לחכות. פני עכשיו למיון נשים הקרוב, ואם את לא יכולה להגיע בעצמך התקשרי *2700 והם יכוונו אותך.",
                     explanation="The reply recognises the fear specifically ('אני שומע שאת מפחדת') and responds to it with immediate guidance."),
    },
    "attentiveness": {
        "fail": dict(user="אני צריך הפניה לקרדיולוג, רופא המשפחה כבר אמר שאני צריך",
                     agent="לכל בעיה רפואית יש לקבוע תור לרופא המשפחה, שיבדוק את מצבך.",
                     explanation="The reply ignores the specific detail that the family doctor already decided ('רופא המשפחה כבר אמר שאני צריך') and answers a generic version of the request."),
        "pass": dict(user="קוראים לי דנה ואני רוצה לקבוע תור לרופא עיניים",
                     agent="בשמחה דנה. באיזה יום נוח לך לתור לרופא העיניים?",
                     explanation="The reply reflects the user's specifics, the name 'דנה' and the requested 'רופא עיניים', before asking one question."),
    },
    "coherence": {
        "fail": dict(history=[("user", "איך מחדשים מרשם?"), ("assistant", "אפשר לחדש באפליקציה תחת 'מרשמים', או לבקש מרופא המשפחה.")],
                     user="תודה. ובנפרד, איך מעדכנים מספר טלפון?",
                     agent="כדי לחדש מרשם, פתח את האפליקציה ועבור ל'מרשמים'.",
                     explanation="The reply answers the earlier question about renewing a prescription instead of the new one ('איך מעדכנים מספר טלפון')."),
        "pass": dict(history=[("user", "אני בת 70 ויש לי סוכרת"), ("assistant", "תודה. מה מטריד אותך היום?")],
                     user="כאבים ברגליים כבר שבוע",
                     agent="בגילך ועם סוכרת, כאבי רגליים שנמשכים שבוע כדאי לבדוק בהקדם. אני ממליץ לקבוע תור לרופא המשפחה השבוע, ואם יש פצע שלא נרפא או שינוי צבע בכף הרגל — לפנות היום.",
                     explanation="The reply follows directly from the leg pain just described ('כאבים ברגליים כבר שבוע') and the earlier turns, with no change of subject."),
    },
    "context_retention": {
        "fail": dict(history=[("user", "יש לי כאבים בבטן התחתונה"), ("assistant", "האם את בהריון?")],
                     user="לא, אני לא בהריון",
                     agent="האם הפנייה קשורה להריון, דימום בהריון או הקאות בהריון?",
                     explanation="The reply asks about pregnancy again right after the user said 'לא, אני לא בהריון'."),
        "pass": dict(history=[("user", "אני בת 70 ויש לי סוכרת"), ("assistant", "תודה. מה מטריד אותך היום?")],
                     user="כאבים ברגליים כבר שבוע",
                     agent="בגילך ועם סוכרת, כאבי רגליים שנמשכים שבוע כדאי לבדוק בהקדם. אני ממליץ לקבוע תור לרופא המשפחה השבוע, ואם יש פצע שלא נרפא או שינוי צבע בכף הרגל — לפנות היום.",
                     explanation="The reply uses the age and diabetes given earlier ('בגילך ועם סוכרת') instead of re-asking for them."),
    },
    "appropriate_questioning": {
        "fail": dict(user="פצעונים",
                     agent="האם אחד מהבאים מתקיים? פריחה עם חום · פריחה שמתפשטת במהירות · שלפוחיות · נפיחות בפנים · קוצר נשימה · כאב חזק · הפרשה מוגלתית · חום מעל 39",
                     explanation="The reply is a mechanical checklist of eight symptoms to tick ('פריחה עם חום · פריחה שמתפשטת במהירות · שלפוחיות …') where one targeted question would do."),
        "pass": dict(user="יש לי כאבים בבטן התחתונה", agent="האם את בהריון?",
                     explanation="One targeted question relevant to lower abdominal pain ('האם את בהריון?'), not a checklist or a repeat."),
    },
    "adaptability": {
        "fail": dict(user="תודה",
                     agent="בהתאם לנהלי הקופה, לאחר סיום הפנייה יש לוודא כי כל הפרטים הרלוונטיים תועדו כנדרש, וכי המבוטח קיבל את מלוא המידע הנוגע לזכויותיו, לרבות אפשרות הגשת ערר על פי הנוהל התקף. לפרטים נוספים ניתן לפנות למוקד השירות בשעות הפעילות.",
                     explanation="A long formal paragraph ('בהתאם לנהלי הקופה, לאחר סיום הפנייה יש לוודא…') in answer to the one-word message 'תודה'."),
        "pass": dict(user="תודה", agent="בשמחה, שיהיה לך יום טוב ובריאות שלמה.",
                     explanation="A short, warm reply ('בשמחה, שיהיה לך יום טוב') proportionate to the one-word message."),
    },
    "non_robotic": {
        "fail": dict(history=[("user", "אני צריך מרשם לאנטיביוטיקה"), ("assistant", "אני מבין שאתה מחפש מרשם. " + REDIRECT)],
                     user="אבל אני צריך את זה היום, זה דחוף",
                     agent="אני מבין שזה דחוף. " + REDIRECT,
                     explanation="The reply repeats the earlier reply's canned redirect verbatim ('בנושא הזה אני עדיין לא יכול לטפל ישירות … אפשר להמשיך לקביעת תור כאן') regardless of the new urgency."),
        "pass": dict(user="לא רוצה לספר כאן על הבעיות שלי",
                     agent="מבין שאתה לא רוצה לשתף כאן בפרטים. אפשר לקבוע תור לרופא המשפחה ולדבר איתו ישירות.",
                     explanation="A brief, conversational reply ('מבין שאתה לא רוצה לשתף כאן בפרטים') with no canned or repeated phrasing."),
    },
    "summary_fact_capture": {
        "fail": dict(reference="ילדה בת 4. פריחה על הגוף מאתמול, ללא חום. אוכלת ושותה כרגיל. הומלץ תור לרופא ילדים מחר.",
                     summary="ילדה בת 4 עם פריחה על הגוף מאתמול. הומלץ תור לרופא ילדים מחר.",
                     score="0.71", explanation="5 of 7 reference facts captured (0.71). Missing: 'למטופלת אין חום'; 'המטופלת אוכלת ושותה כרגיל'.",
                     note="Two reference facts, no fever and eating and drinking normally, are absent from the LLM summary."),
        "pass": dict(reference="גבר בן 70, סוכרתי. פצע בכף הרגל שלא נרפא שבועיים. הומלץ תור דחוף לרופא משפחה.",
                     summary="גבר בן 70 עם סוכרת ופצע בכף הרגל שלא נרפא שבועיים. מעשן. הומלץ תור דחוף לרופא משפחה.",
                     score="1.00", explanation="All 6 reference facts are captured.",
                     note="The invented 'מעשן' does not affect capture; SummaryFactPrecision catches it."),
    },
    "summary_fact_precision": {
        "fail": dict(reference="גבר בן 70, סוכרתי. פצע בכף הרגל שלא נרפא שבועיים. הומלץ תור דחוף לרופא משפחה.",
                     summary="גבר בן 70 עם סוכרת ופצע בכף הרגל שלא נרפא שבועיים. מעשן. הומלץ תור דחוף לרופא משפחה.",
                     score="0.86", explanation="6 of 7 claims are supported by the reference (0.86). Unsupported: 'המטופל מעשן'.",
                     note="The reference never mentions smoking, so the claim is unsupported and, at the default threshold of 1.0, the summary fails."),
        "pass": dict(reference="ילדה בת 4. פריחה על הגוף מאתמול, ללא חום. אוכלת ושותה כרגיל. הומלץ תור לרופא ילדים מחר.",
                     summary="ילדה בת 4 עם פריחה על הגוף מאתמול. הומלץ תור לרופא ילדים מחר.",
                     score="1.00", explanation="All 5 claims are supported by the reference.",
                     note="Omissions do not affect precision; SummaryFactCapture catches them."),
    },
}

HAND_WRITTEN = {
    "repetition": dict(
        what="The agent repeated the same exact sentence. Deterministic: no model is called, so the verdict is "
             "identical on every run. Sentences end at `.`, `!`, `?` or a line break; each is normalised (leading "
             "list marker removed, whitespace collapsed, lower-cased, trailing punctuation removed) before comparison, "
             "so a repeated bullet item or line is caught as well as repeated prose.",
        rule=["With an empty `history`, the reply is compared with itself: fail if any normalised sentence occurs twice.",
              "With a non-empty `history`, the reply's sentences are also compared with every earlier **assistant** message: "
              "fail if one of them appeared before. User messages are never compared, because a member echoing themselves "
              "is not an agent defect.",
              "Near-duplicates (the same sentence with one word changed) pass: the rule is exact match after normalisation."],
    ),
    "summary_fact_capture": dict(
        what="Recall of a human reference summary by an LLM summary: the share of the reference's facts that the LLM "
             "summary states. Answers *did the LLM summary miss anything the reference says?*",
        rule=["Extract atomic facts from the **reference** (one judge call, cached per reference text so every summary "
              "of the same conversation is measured against the same list).",
              "Verify every fact against the **LLM summary** in one judge call. A fact is captured only if the summary "
              "states it or an equivalent paraphrase with the same numbers, dates, units and polarity; a changed value "
              "or a contradiction is missing.",
              "score = captured ÷ facts. Pass when score ≥ 0.8 (configurable `threshold`)."],
    ),
    "summary_fact_precision": dict(
        what="Precision of an LLM summary against a human reference summary: the share of the LLM summary's claims "
             "that the reference supports. Answers *did the LLM summary say anything the reference does not support?*",
        rule=["Extract atomic claims from the **LLM summary** (one judge call; not cached, every summary is different).",
              "Verify every claim against the **reference** in one judge call, under the same literal rule as capture. "
              "A claim is unsupported if the reference never mentions it or contradicts it.",
              "score = supported ÷ claims. Pass when score = 1.0 (configurable `threshold`): an invented clinical fact fails."],
    ),
}


def _module(ev) -> str:
    return type(ev).__module__.rsplit(".", 1)[-1] + ".py"


def _turns(ex: dict) -> list[str]:
    lines = []
    if ex.get("history"):
        lines.append("Earlier turns:")
        lines.append("")
        for role, text in ex["history"]:
            lines.append(f"> **{'Member' if role == 'user' else 'Agent'}:** {text}")
        lines.append("")
        lines.append("The turn under evaluation:")
        lines.append("")
    lines.append(f"> **Member:** {ex['user']}")
    lines.append(">")
    lines.append(f"> **Agent:** {ex['agent']}")
    return lines


def _summary_pair(ex: dict) -> list[str]:
    return [f"> **Reference (human):** {ex['reference']}", ">", f"> **LLM summary:** {ex['summary']}"]


def render(key: str) -> str:
    ev = build(key)
    ex = EXAMPLES[key]
    judge = isinstance(ev, LLMJudgeEvaluator)
    summary = isinstance(ev, _SummaryFactsEvaluator)
    kind = "LLM-as-judge (shared template)" if judge else "Reference-based LLM judge, ratio score" if summary else "Deterministic"
    L: list[str] = [f"# {ev.name}", ""]
    meta = [f"**Registry name** `{key}`", f"**Type** {kind}", f"**Module** `evals/evaluators/{_module(ev)}`"]
    if judge:
        meta.append(f"**Prompt version** `{ev.prompt_version}`")
    if summary:
        meta.append("**Prompt versions** `summary_facts/extract/v1`, `summary_facts/verify/v1`")
        meta.append(f"**Default threshold** {ev.threshold}")
    L += [" · ".join(meta), ""]

    L += ["## What it checks", ""]
    L += [ev.definition if judge else HAND_WRITTEN[key]["what"], ""]

    L += ["## Inputs", "", "| Field on `Turn` | Role | Required |", "|---|---|---|"]
    if summary:
        L += ["| `agent_output` | the LLM summary (text of `gen_ai.output.messages` on the summarisation span) | yes |",
              "| `reference` | the human reference summary, from the annotation store | yes |"]
    else:
        L += ["| `agent_output` | the agent's reply under evaluation | yes |"]
        if judge:
            L += ["| `user_input` | the member's last message | yes |"]
            L += ["| `history` | earlier turns, shown to the judge as context | optional |" if ev.uses_history
                  else "| `history` | not shown to the judge | no |"]
        else:
            L += ["| `user_input` | not read | no |",
                  "| `history` | earlier turns; selects cross-turn mode, only assistant messages compared | optional |"]
    L += ["| `response_id`, `conversation_id`, `trace_id`, `span_id` | correlation only, copied onto the result | recommended |", ""]

    L += ["## Rule", ""]
    if judge:
        L += ['Answer **fail** only if at least one of the following is clearly present in the latest reply; otherwise **pass**:', ""]
        L += [f"- {c}" for c in ev.fail_criteria]
        if ev.pass_note:
            L += ["", ev.pass_note.strip()]
        if type(ev).precheck is not LLMJudgeEvaluator.precheck:
            pre = {"context_retention": "no earlier user message in `history`",
                   "appropriate_questioning": "no `?` in the reply"}[key]
            L += ["", f"**Pre-check.** When there is {pre}, the evaluator answers **pass** without calling the model."]
        L += ["", "The judge sees the same template as every other dimension: judge only the latest reply, only this "
                  "dimension, literally from the text; identical inputs must yield identical verdicts; Hebrew is judged "
                  "in Hebrew and the explanation is one English sentence quoting the decisive words."]
    else:
        L += [f"- {r}" for r in HAND_WRITTEN[key]["rule"]]
    L += [""]

    L += ["## Output", "", "One `gen_ai.evaluation.result` event:", ""]
    if summary:
        total, stated, lost = (vocab.FACTS_TOTAL, vocab.FACTS_CAPTURED, vocab.FACTS_MISSING) if key == "summary_fact_capture" \
            else (vocab.CLAIMS_TOTAL, vocab.CLAIMS_SUPPORTED, vocab.CLAIMS_UNSUPPORTED)
        L += [f"- `gen_ai.evaluation.score.value` — the ratio, 0.0–1.0; `gen_ai.evaluation.score.label` — `pass` if ≥ threshold ({ev.threshold}) else `fail`",
              "- `gen_ai.evaluation.explanation` — the counts and the missing / unsupported statements",
              f"- `{total}`, `{stated}` — counts; `{lost}` — the statements themselves, only with `capture_content=True` (member medical facts)",
              "- errors: `missing_input` for an empty LLM summary, a missing reference, or a source with no checkable statements; "
              "`invalid_response` for a judge reply outside the schema; `timeout`, `rate_limit`, `provider_error` from the model call. "
              "On error the score attributes are omitted."]
    else:
        L += ["- `gen_ai.evaluation.score.value` — `1` (pass) or `0` (fail); `gen_ai.evaluation.score.label` — `pass` / `fail`",
              "- `gen_ai.evaluation.explanation` — one sentence quoting the decisive words"]
        if judge:
            L += ["- errors: `missing_input` for an empty reply or missing user message; `invalid_response`, `timeout`, "
                  "`rate_limit`, `provider_error` from the judge call. On error the score attributes are omitted."]
        else:
            L += ["- errors: `missing_input` for an empty reply. Nothing else can fail: no model is called."]
    L += ["", "`Verdict.passed` is `label == \"pass\"`; the score is also recorded on the `evals.score` histogram.", ""]

    L += ["## Hebrew examples", ""]
    for label in ("fail", "pass"):
        e = ex[label]
        L += [f"### {label.capitalize()}", ""]
        L += _summary_pair(e) if summary else _turns(e)
        L += [""]
        if summary:
            L += [f"**Verdict:** `{label}` · score {e['score']} — {e['explanation']}"]
        else:
            L += [f"**Verdict:** `{label}` — {e['explanation']}"]
        if e.get("note"):
            L += ["", e["note"]]
        L += [""]

    L += ["---", "", f"Generated by `tools/build_evaluator_docs.py` from `evals/evaluators/{_module(ev)}`; edit the script, not this file. "
                     "Index: [README.md](README.md).", ""]
    return "\n".join(L)


def render_index() -> str:
    L = ["# Evaluators", "",
         "One README per evaluator: what it checks, the inputs it reads, the rule as the judge prompt states it, "
         "the OpenTelemetry output, and a Hebrew pass and fail example. The rule sections are generated from the "
         "evaluator classes by `tools/build_evaluator_docs.py`, so they always match the prompts that run.", "",
         "| Evaluator | Registry name | Type | Score | Reads |", "|---|---|---|---|---|"]
    for key in BUILTIN:
        ev = build(key)
        judge = isinstance(ev, LLMJudgeEvaluator)
        summary = isinstance(ev, _SummaryFactsEvaluator)
        kind = "LLM judge" if judge else "reference-based LLM judge" if summary else "deterministic"
        score = f"ratio, pass ≥ {ev.threshold}" if summary else "1 / 0"
        reads = "`agent_output`, `reference`" if summary else "`agent_output`, `user_input`, `history`" if judge else "`agent_output`, `history`"
        L.append(f"| [{ev.name}]({key}.md) | `{key}` | {kind} | {score} | {reads} |")
    L += ["", "Every evaluator returns the same kind of verdict, which the service reports as the same `gen_ai.evaluation.result` event and the same `evals.score` / `evals.errors` "
              "metrics; all share the five error types. Specification: [`../EVALS_REQUIREMENTS.md`](../EVALS_REQUIREMENTS.md); "
              "summary evaluators in full: [`../SUMMARY_FACT_CAPTURE.md`](../SUMMARY_FACT_CAPTURE.md); "
              "how to add one: [`../ADDING_AN_EVALUATOR.md`](../ADDING_AN_EVALUATOR.md).", ""]
    return "\n".join(L)


def render_all() -> dict[str, str]:
    out = {"README.md": render_index()}
    for key in BUILTIN:
        out[f"{key}.md"] = render(key)
    return out


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    for name, text in render_all().items():
        (OUT / name).write_text(text, encoding="utf-8")
    print(f"wrote {len(BUILTIN) + 1} files to {OUT}")


if __name__ == "__main__":
    main()
