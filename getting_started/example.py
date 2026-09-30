"""One Clalit triage turn, evaluated, with everything the package gives back.

    python3 getting_started/example.py --offline                 # a stub judge, no key
    ANTHROPIC_API_KEY=... python3 getting_started/example.py     # the real judge

The package evaluates a turn and returns Verdicts. Each Verdict carries the
record of how it was produced (timing, the judge calls with model and tokens);
the package itself emits no telemetry and stores nothing. Turn on logging to
watch it work.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run from a clone without pip install -e .

import logging
import os
import sys

from evals import Evals

MEMBER_MESSAGE = "I've had a fever of 38.5 and a sore throat since yesterday. Should I come in?"
AGENT_REPLY = ("Book a same-day appointment at your Clalit clinic. If breathing becomes difficult, go to the ER. "
               "Book a same-day appointment at your Clalit clinic.")


def main(offline: bool) -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)-7s %(name)s: %(message)s")

    if offline:
        from stub_client import StubClient
        ev = Evals(judge_provider=StubClient("judge"))
    else:
        ev = Evals()                                     # the judge from the environment

    # The turn as your agent produced it (or as it was read back from the trace store).
    verdicts = ev.evaluate(
        user_input=MEMBER_MESSAGE,
        agent_output=AGENT_REPLY,
        history=[("user", "Hi, I'm not feeling well."),
                 ("assistant", "I'm sorry to hear that. What symptoms are you having?")],
        response_id="msg_01Ab3Cd5Ef",                    # the agent's response id: the link back to the trace
        conversation_id="conv_8b21f4",
        tenant_id="clalit",                              # reporting dimensions travel on the turn
        group="clalit-innovation",
    )

    print("\n=== verdicts ===")
    print(verdicts)
    print(f"\nok={verdicts.ok}  relevance passed={verdicts['Relevance'].passed}")

    print("\n=== the record behind each verdict (what a host reports as spans and metrics) ===")
    for v in verdicts:
        print(f"{v.name:<11} {v.status:<6} {1000 * (v.ended_at - v.started_at):6.1f} ms  calls={len(v.calls)}")
        for c in v.calls:
            print(f"{'':<11} judge {c.provider}/{c.model} prompt={c.prompt_version} tokens={c.input_tokens}/{c.output_tokens} "
                  f"finish={c.finish_reason} {1000 * c.duration_s:.1f} ms")


if __name__ == "__main__":
    offline = "--offline" in sys.argv
    if not offline and not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("OPENAI_API_KEY")):
        sys.exit("no judge configured (ANTHROPIC_API_KEY or OPENAI_API_KEY). Use --offline to run with a stub.")
    main(offline)
