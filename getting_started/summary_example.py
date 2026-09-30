"""The two summary fact evaluators on four LLM summaries of one human reference.

    python3 getting_started/summary_example.py --offline   # deterministic stub judge, no API key
    python3 getting_started/summary_example.py             # the real judge (ANTHROPIC_API_KEY)

SummaryFactCapture asks "did the LLM summary miss anything the reference says?"
(recall, passes at >= 0.8). SummaryFactPrecision asks "did it say anything the
reference does not support?" (precision, passes only at 1.0). The four summaries
cover every combination, which is why they are two scores and not one. Offline,
the stub judge replays a reviewer's facts and verdicts for these exact texts.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run from a clone without pip install -e .

import json
import sys

from evals import Evals
from stub_client import SUMMARY_CASES, SUMMARY_REFERENCE

kwargs = {}
if "--offline" in sys.argv:
    from stub_client import StubClient
    kwargs = {"judge_provider": StubClient("summary_judge")}

from evals.evaluators import SummaryFactCaptureEvaluator, SummaryFactPrecisionEvaluator

# capture_content=True makes the evaluators list the missing facts / unsupported claims themselves (member text);
# it is an option of these two evaluators, so build the instances rather than naming them.
ev = Evals(evaluators=[SummaryFactCaptureEvaluator(capture_content=True), SummaryFactPrecisionEvaluator(capture_content=True)], **kwargs)
results = []
if True:
    print(f"Reference (human):\n  {SUMMARY_REFERENCE}\n")
    for i, case in enumerate(SUMMARY_CASES, 1):
        # The LLM summary is agent_output, the human one is reference. In a
        # pipeline these come from the summarisation span and the annotation store.
        verdicts = ev.evaluate(
            agent_output=case["summary"], reference=SUMMARY_REFERENCE,
            response_id=f"msg_summary_{i:02d}", conversation_id="conv_8b21f4", tenant_id="clalit",
        )
        print(f"--- {i}. {case['title']} ---")
        print(f"LLM summary:\n  {case['summary']}")
        for v in verdicts:
            print(f"  {v.name:<21} {v.label:<5} {v.score:.2f}   {v.explanation}")
        print(f"  both pass: {'yes' if verdicts.ok else 'no'}\n")
        results.append(verdicts)

# The package emitted nothing; the counts (and, with capture_content=True, the lists) travel on the
# verdicts as attributes, for whoever reports or stores them.
print("Case 3, the record on each verdict -- attributes and judge calls:")
for v in results[2]:
    print(f"  {v.name}: {json.dumps(dict(v.attributes), ensure_ascii=False)}")
    print(f"    calls: {[(c.prompt_version, c.model) for c in v.calls]}")
