"""Many turns: the package evaluates one turn per call; a thread pool is the batch.

    pip install "clalit-evals @ git+ssh://git@github.com/OArbili/clalit-evals.git@v0.2.0"
    python3 05_many_turns.py --offline
    python3 05_many_turns.py                       # the judge from the environment (50 judge calls)
"""

import sys
from concurrent.futures import ThreadPoolExecutor

from evals import Evals


def fake_judge(*, system, messages, schema, **_):
    if "weather" in messages[0].text.split("<assistant_reply>")[-1]:      # the off-topic rows
        return {"verdict": "fail", "explanation": "The reply talks about the weather, not the member's question."}
    return {"verdict": "pass", "explanation": "The reply addresses the member's question."}


judge = {"judge_provider": fake_judge, "judge_model": "fake-judge"} if "--offline" in sys.argv else {}
ev = Evals(**judge)                                 # Evals is safe to share between threads

rows = [{"response_id": f"msg_{i:02d}", "user_input": "Should I come in with this fever?",
         "agent_output": "Book a same-day appointment at your clinic." if i % 10 else "The weather is lovely today."}
        for i in range(50)]

with ThreadPoolExecutor(max_workers=8) as pool:
    report = list(pool.map(ev.evaluate, rows))      # Verdicts per row, in input order

by_name: dict[str, list] = {}
for vs in report:
    for v in vs:
        by_name.setdefault(v.name, []).append(v.passed)
pass_rate = {n: sum(p is True for p in ps) / max(1, sum(p is not None for p in ps)) for n, ps in by_name.items()}

print("turns:", len(report))
print("pass rate:", {n: f"{r:.0%}" for n, r in pass_rate.items()})
print("failed rows:", [vs.turn.response_id for vs in report if vs.failed])
print("errors:", [vs.turn.response_id for vs in report if vs.errors])
