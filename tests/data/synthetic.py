"""Loader for the synthetic dataset and converters to the shapes the pipeline consumes."""

from __future__ import annotations

import json
from pathlib import Path

DIMENSIONS = ["repetition", "relevance", "naturalness", "empathy", "attentiveness", "coherence",
              "context_retention", "appropriate_questioning", "adaptability", "non_robotic"]
INACTIVITY = "לא זוהתה פעילות"
PATH = Path(__file__).with_name("synthetic_conversations.jsonl")


def load() -> list[dict]:
    return [json.loads(line) for line in PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def expected_turns(convs: list[dict] | None = None) -> list[dict]:
    """One record per evaluated assistant message, in pipeline order, with the export's response_id."""
    out = []
    for c in convs or load():
        for i, m in enumerate(c["messages"]):
            if m["role"] != "assistant" or m.get("skip"):
                continue
            out.append({"conversation_id": c["id"], "response_id": f"{c['id']}_{i + 1}", "position": i + 1,
                        "agent_output": m["text"], "expected": m["expected"], "why": m.get("why", "")})
    return out


def evals_rows(convs: list[dict] | None = None) -> list[dict]:
    """One row per turn, as Evals.evaluate() takes them: one turn per assistant message, history = everything before it."""
    rows = []
    for c in convs or load():
        msgs = c["messages"]; last_user = None
        for i, m in enumerate(msgs):
            if m["role"] == "user":
                last_user = m["text"]; continue
            if m.get("skip"):
                continue
            rows.append({"conversation_id": c["id"], "response_id": f"{c['id']}_{i + 1}",
                         "user_input": last_user, "agent_output": m["text"],
                         "history": [(x["role"], x["text"]) for x in msgs[:i]]})
    return rows


def export_rows(convs: list[dict] | None = None) -> list[dict]:
    """The Elastic export shape. Flags are conversation-level ground truth (TRUE = a problem)."""
    rows = []
    for c in convs or load():
        text = "\n".join(f"{m['role']}: {m['text']}" for m in c["messages"])
        judged = [m for m in c["messages"] if m["role"] == "assistant" and not m.get("skip")]
        rep = any(m["expected"]["repetition"] == "fail" for m in judged)
        rel = any(m["expected"]["relevance"] == "fail" for m in judged)
        meta = {"user_id": "synthetic", "language": c["lang"], "agent_id": "synthetic_triage_agent",
                "message_count": len(c["messages"]) + 1, "last_message_id": len(c["messages"]),
                "initial_message_uid": f"{c['id']}_0"}
        rows.append({"conversation": text, "metadata": repr(meta), "session_id": c["id"],
                     "repetition_flag": "TRUE" if rep else "FALSE", "repetition_explanation": "synthetic ground truth",
                     "relevancy_flag": "TRUE" if rel else "FALSE", "relevancy_explanation": "synthetic ground truth"})
    return rows


def write_export_csv(path: Path, convs: list[dict] | None = None) -> Path:
    import csv
    rows = export_rows(convs)
    with open(path, "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader(); w.writerows(rows)
    return path


# ---- summaries ---------------------------------------------------------------------------

SUMMARIES_PATH = Path(__file__).with_name("synthetic_summaries.jsonl")
SUMMARY_EVALUATORS = ["summary_fact_capture", "summary_fact_precision"]


def load_summaries() -> list[dict]:
    """One record per (reference, LLM summary) pair with labelled facts, claims and expected scores."""
    return [json.loads(line) for line in SUMMARIES_PATH.read_text(encoding="utf-8").splitlines() if line.strip()]


def summary_rows(cases: list[dict] | None = None) -> list[dict]:
    """One row per case, as Evals.evaluate() takes them: the LLM summary is agent_output, the human one is reference."""
    return [{"conversation_id": c["conversation_id"], "response_id": c["response_id"],
             "agent_output": c["llm_summary"], "reference": c["reference"]} for c in cases or load_summaries()]
