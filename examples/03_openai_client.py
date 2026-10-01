"""The judge through an openai SDK client, or any OpenAI-compatible endpoint.

    pip install "clalit-evals[openai] @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"
    export OPENAI_API_KEY=...
    python3 03_openai_client.py                    # openai.OpenAI() as the judge, model gpt-5-mini

    pip install "clalit-evals @ git+https://github.com/OArbili/clalit-evals.git@v0.3.0"     # no SDK: the endpoint over the standard library
    OPENAI_BASE_URL=http://vllm.internal:8000/v1 EVALS_JUDGE_MODEL=my-judge python3 03_openai_client.py --http

    python3 03_openai_client.py --offline          # no key: a client-shaped test double
"""

import sys

from evals import Evals
from evals.providers import OpenAICompatibleProvider

TURN = dict(user_input="Can I take ibuprofen with my blood pressure pills?",
            agent_output="Please check with your pharmacist or doctor before combining them; I can book a call today.")

if "--offline" in sys.argv:
    class _Completion:                                   # what client.chat.completions.create returns, reduced
        def model_dump(self):
            return {"id": "chatcmpl-1", "model": "gpt-test", "usage": {"prompt_tokens": 50, "completion_tokens": 12},
                    "choices": [{"finish_reason": "stop", "message": {"content":
                        '{"verdict": "pass", "explanation": "The reply addresses the drug interaction the member asked about."}'}}]}

    class FakeOpenAIClient:                              # the shape as_provider() recognises: .chat.completions.create
        class chat:
            class completions:
                @staticmethod
                def create(**kwargs):
                    return _Completion()

    ev = Evals(judge_provider=FakeOpenAIClient(), judge_model="gpt-test")
elif "--http" in sys.argv:
    ev = Evals(judge_provider=OpenAICompatibleProvider())   # OPENAI_BASE_URL / OPENAI_API_KEY; model from EVALS_JUDGE_MODEL
else:
    import openai
    ev = Evals(judge_provider=openai.OpenAI(), judge_model="gpt-5-mini")   # AzureOpenAI(...) works the same way

verdicts = ev.evaluate(**TURN)
print(verdicts)
call = verdicts["relevance"].calls[0]
print("judge:", call.provider, call.model, "->", call.response_model, "| tokens", call.input_tokens, call.output_tokens)
