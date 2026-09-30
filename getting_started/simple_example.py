"""The simplest use of the package: evaluate one turn, print the verdicts. Nothing is exported or stored.

    python3 getting_started/simple_example.py            # offline: a stub judge, no key, no network

Only a judge provider is needed. For a real judge remove the judge_provider
kwarg; the SDK then takes the provider from the environment (ANTHROPIC_API_KEY,
or OPENAI_API_KEY / OPENAI_BASE_URL). The package emits no telemetry: it returns
Verdicts and, if you enable logging (logging.basicConfig), logs what it did.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))   # run from a clone without pip install -e .

from evals import Evals
from stub_client import StubClient

ev = Evals(judge_provider=StubClient("judge"))

USER = "I've had a fever of 38.5 and a sore throat since yesterday. Should I come in?"
REPLY = "Book a same-day appointment at your Clalit clinic. If breathing becomes difficult, go to the ER."

# 1. The minimum: the member's message and the agent's reply.
print("--- 1. the minimum ---")
verdicts = ev.evaluate(user_input=USER, agent_output=REPLY)
print(verdicts)                  # one line per evaluator: name, pass / fail / error, explanation
print("ok:", verdicts.ok)

# 2. The same turn with every identifier. All are optional and pass through unchanged
#    to whatever reports or stores the verdicts; the SDK holds none of them.
print("\n--- 2. with every identifier ---")
verdicts = ev.evaluate(
    user_input=USER,
    agent_output=REPLY,
    history=[("user", "Hi, I'm not feeling well."),          # earlier messages, oldest first (reference= for summaries)
             ("assistant", "I'm sorry to hear that. What symptoms are you having?")],
    response_id="msg_01Ab3Cd5Ef",                   # gen_ai.response.id: the agent's reply; the link when span ids are missing
    conversation_id="conv_8b21f4",                  # gen_ai.conversation.id: the whole conversation
    trace_id="4bf92f3577b34da6a3ce929d0e0e4736",    # the agent's span in the trace store: the verdict event is parented to it
    span_id="00f067aa0ba902b7",
    model="agent-model-1",                          # gen_ai.request.model: the agent's model, a metric dimension only
    tenant_id="clalit",                             # tenant.id and service.namespace: reporting dimensions on the turn
    group="clalit-innovation",
)
print(verdicts)
print("ok:", verdicts.ok)
