"""The provider contract and its four adapters; the judge through a non-Anthropic provider."""

import json
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer

import anthropic
import httpx2 as httpx
import openai
import pytest

from evals import Evals, vocab
from evals.evaluators.relevance import RelevanceEvaluator
from evals.llm import LLMError, LLMProviderError, LLMRateLimited, LLMRefusal, LLMResponse, LLMTimeout, Provider
from evals.providers import (AnthropicProvider, CallableProvider, OpenAIClientProvider, OpenAICompatibleProvider,
                             as_provider, default_provider)
from evals.providers.anthropic import STOP_REASONS
from evals.types import Message, Turn
from helpers import FixedJudge, RaisingJudge, make
from stub_client import StubClient, _Message, _TextBlock

REQ = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
TURN = Turn(agent_output="Rest and drink.", user_input="I have a sore throat.")


class TestAsProvider:
    def test_none_and_provider_pass_through(self):
        p = CallableProvider(lambda **kw: "x")
        assert as_provider(None) is None and as_provider(p) is p

    def test_anthropic_shaped_client_is_wrapped(self):
        stub = StubClient("judge")
        p = as_provider(stub)
        assert isinstance(p, AnthropicProvider) and p.client is stub and p.name == "anthropic"

    def test_callable_is_wrapped(self):
        assert isinstance(as_provider(lambda **kw: "x"), CallableProvider)

    def test_openai_shaped_client_is_wrapped(self):
        client = OpenAISpy()
        p = as_provider(client)
        assert isinstance(p, OpenAIClientProvider) and p.client is client and p.name == "openai"

    def test_anything_else_is_rejected(self):
        with pytest.raises(TypeError, match="expected a Provider"):
            as_provider(object())


class TestDefaultProvider:
    def test_explicit_choice_and_key_detection(self, monkeypatch):
        for k in ("EVALS_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        assert isinstance(default_provider(), OpenAICompatibleProvider)
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")
        assert isinstance(default_provider(), AnthropicProvider)         # Anthropic wins when both keys are set
        monkeypatch.setenv("EVALS_PROVIDER", "openai")
        assert isinstance(default_provider(), OpenAICompatibleProvider)   # unless chosen explicitly
        monkeypatch.setenv("EVALS_PROVIDER", "nope")
        with pytest.raises(ValueError, match="EVALS_PROVIDER"):
            default_provider()

    def test_nothing_configured_is_explained(self, monkeypatch):
        for k in ("EVALS_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        with pytest.raises(RuntimeError, match="no LLM provider configured"):
            default_provider()


class TestAnthropicProvider:
    def test_request_shape_and_response_normalisation(self):
        seen = {}

        class Spy:
            class messages:
                @staticmethod
                def create(**kw):
                    seen.update(kw)
                    return _Message(content=[_TextBlock("hello")], stop_reason="max_tokens")

        r = AnthropicProvider(Spy()).complete(model="m", messages=[Message("user", "hi")], system="sys",
                                              schema={"type": "object"}, max_tokens=5, timeout_s=3.0, effort="low")
        assert seen == {"model": "m", "max_tokens": 5, "messages": [{"role": "user", "content": "hi"}], "system": "sys",
                        "output_config": {"format": {"type": "json_schema", "schema": {"type": "object"}}, "effort": "low"},
                        "timeout": 3.0}
        assert r == LLMResponse(text="hello", model="claude-opus-4-8", id="msg_01Ab3Cd5Ef", input_tokens=412,
                                output_tokens=38, finish_reason="length")

    def test_no_system_no_schema_no_effort_sends_nothing_extra(self):
        seen = {}

        class Spy:
            class messages:
                @staticmethod
                def create(**kw):
                    seen.update(kw); return _Message(content=[_TextBlock("x")])

        AnthropicProvider(Spy()).complete(model="m", messages=[Message("user", "hi")])
        assert set(seen) == {"model", "max_tokens", "messages"}

    @pytest.mark.parametrize("exc, cls", [
        (anthropic.APITimeoutError(request=REQ), LLMTimeout),
        (anthropic.RateLimitError("slow", response=httpx.Response(429, request=REQ), body=None), LLMRateLimited),
        (anthropic.InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None), LLMProviderError),
    ])
    def test_sdk_exceptions_become_llm_errors(self, exc, cls):
        with pytest.raises(cls):
            AnthropicProvider(RaisingJudge(exc)).complete(model="m", messages=[Message("user", "hi")])

    def test_other_exceptions_propagate_untouched(self):
        with pytest.raises(RuntimeError):
            AnthropicProvider(RaisingJudge(RuntimeError("bug"))).complete(model="m", messages=[Message("user", "hi")])

    def test_stop_reasons_cover_the_sdk_vocabulary(self):
        assert STOP_REASONS["refusal"] == "content_filter" and set(STOP_REASONS.values()) <= set(vocab.FINISH_REASONS)


class FakeOpenAI(BaseHTTPRequestHandler):
    """A chat-completions endpoint whose behaviour is chosen by the model name."""
    requests: list = []

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        FakeOpenAI.requests.append((self.path, dict(self.headers), body))
        model = body["model"]
        if model == "rate-limited":
            self.send_response(429); self.end_headers(); self.wfile.write(b'{"error": "slow down"}'); return
        if model == "broken":
            self.send_response(500); self.end_headers(); self.wfile.write(b"oops"); return
        if model == "refuses":
            msg = {"role": "assistant", "content": None, "refusal": "I can't help with that"}
            fin = "stop"
        else:
            content = json.dumps({"verdict": "fail", "explanation": "off topic"}) if body.get("response_format") else "plain text"
            msg = {"role": "assistant", "content": content}
            fin = "length" if model == "truncates" else "stop"
        out = {"id": "chatcmpl-1", "model": model + "-2026", "choices": [{"message": msg, "finish_reason": fin}],
               "usage": {"prompt_tokens": 10, "completion_tokens": 4}}
        self.send_response(200); self.send_header("Content-Type", "application/json"); self.end_headers()
        self.wfile.write(json.dumps(out).encode())

    def log_message(self, *a):
        pass


@pytest.fixture(scope="module")
def openai_url():
    srv = HTTPServer(("127.0.0.1", 0), FakeOpenAI)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


class TestOpenAICompatibleProvider:
    def test_request_shape_auth_and_response(self, openai_url):
        FakeOpenAI.requests.clear()
        p = OpenAICompatibleProvider(openai_url, "sk-test", default_model="gpt-x")
        r = p.complete(model="gpt-x", messages=[Message("user", "hi")], system="sys", schema={"type": "object"}, max_tokens=7)
        path, headers, body = FakeOpenAI.requests[-1]
        assert path == "/v1/chat/completions" and headers["Authorization"] == "Bearer sk-test"
        assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert body["max_tokens"] == 7 and body["response_format"]["json_schema"]["strict"] is True
        assert body["response_format"]["json_schema"]["schema"] == {"type": "object"}
        assert r == LLMResponse(text='{"verdict": "fail", "explanation": "off topic"}', model="gpt-x-2026", id="chatcmpl-1",
                                input_tokens=10, output_tokens=4, finish_reason="stop")

    def test_azure_style_options(self, openai_url):
        FakeOpenAI.requests.clear()
        p = OpenAICompatibleProvider(openai_url, "az-key", name="azure.ai.openai", api_key_header="api-key",
                                     query={"api-version": "2024-10-21"}, token_param="max_completion_tokens",
                                     reasoning_effort=True, request_overrides={"temperature": 0})
        assert p.name == "azure.ai.openai"
        p.complete(model="gpt-x", messages=[Message("user", "hi")], effort="low")
        path, headers, body = FakeOpenAI.requests[-1]
        headers = {k.lower(): v for k, v in headers.items()}   # urllib normalises header-name case
        assert path == "/v1/chat/completions?api-version=2024-10-21" and headers["api-key"] == "az-key"
        assert body["max_completion_tokens"] == 1024 and body["reasoning_effort"] == "low" and body["temperature"] == 0
        assert "max_tokens" not in body and "response_format" not in body

    def test_finish_reasons_and_refusal(self, openai_url):
        p = OpenAICompatibleProvider(openai_url, "k")
        assert p.complete(model="truncates", messages=[Message("user", "hi")]).finish_reason == "length"
        r = p.complete(model="refuses", messages=[Message("user", "hi")])
        assert r.finish_reason == "content_filter" and r.text == ""

    def test_http_failures_become_llm_errors(self, openai_url):
        p = OpenAICompatibleProvider(openai_url, "k")
        with pytest.raises(LLMRateLimited, match="429"):
            p.complete(model="rate-limited", messages=[Message("user", "hi")])
        with pytest.raises(LLMProviderError, match="500"):
            p.complete(model="broken", messages=[Message("user", "hi")])
        with pytest.raises(LLMProviderError):
            OpenAICompatibleProvider("http://127.0.0.1:9/v1", "k").complete(model="m", messages=[Message("user", "hi")], timeout_s=1)

    def test_env_configuration(self, monkeypatch):
        monkeypatch.setenv("OPENAI_BASE_URL", "http://gw:8000/v1/"); monkeypatch.setenv("OPENAI_API_KEY", "k")
        monkeypatch.setenv("OPENAI_MODEL", "local-model")
        p = OpenAICompatibleProvider()
        assert p.url == "http://gw:8000/v1/chat/completions" and p.default_model == "local-model"

    def test_judge_through_an_openai_endpoint(self, openai_url):
        """The whole judge path -- prompt, schema, span, verdict -- with no Anthropic code involved."""
        ev = RelevanceEvaluator(OpenAICompatibleProvider(openai_url, "k"), model="gpt-x")
        r = ev.evaluate(TURN)
        assert (r.label, r.explanation) == ("fail", "off topic")
        assert RelevanceEvaluator(OpenAICompatibleProvider(openai_url, "k"), model="refuses").evaluate(TURN).error_type == vocab.ERR_PROVIDER
        assert RelevanceEvaluator(OpenAICompatibleProvider(openai_url, "k"), model="rate-limited").evaluate(TURN).error_type == vocab.ERR_RATE_LIMIT

    def test_core_with_an_openai_endpoint(self, openai_url):
        p = OpenAICompatibleProvider(openai_url, "k", default_model="gpt-x")
        vs = Evals(judge_provider=p).evaluate(TURN)
        assert vs["Relevance"].label == "fail"
        (call,) = vs["Relevance"].calls
        assert call.provider == "openai" and call.model == "gpt-x" and call.input_tokens == 10


def _completion(content='{"verdict": "pass", "explanation": "ok"}', finish="stop", refusal=None):
    """What openai's ChatCompletion.model_dump() returns, reduced to the fields the adapter reads."""
    return {"id": "chatcmpl-9x", "model": "gpt-x-2026", "choices": [{"finish_reason": finish,
            "message": {"role": "assistant", "content": content, "refusal": refusal}}],
            "usage": {"prompt_tokens": 40, "completion_tokens": 9}}


class _Dumpable:
    def __init__(self, data): self._data = data
    def model_dump(self): return self._data


class OpenAISpy:
    """The shape of an openai.OpenAI() client: .chat.completions.create(**kwargs) -> ChatCompletion."""

    def __init__(self, result=None, raises=None):
        self.seen = {}
        spy = self

        class _Completions:
            @staticmethod
            def create(**kw):
                spy.seen.update(kw)
                if raises is not None:
                    raise raises
                return _Dumpable(result if result is not None else _completion())

        class _Chat:
            completions = _Completions()

        self.chat = _Chat()


class AzureOpenAI(OpenAISpy):
    """Named like the SDK's Azure client, so the provider name can be derived from it."""


class TestOpenAIClientProvider:
    def test_request_is_the_sdk_keyword_arguments_and_response_is_normalised(self):
        client = OpenAISpy()
        r = OpenAIClientProvider(client).complete(model="gpt-x", messages=[Message("user", "hi")], system="sys",
                                                   schema={"type": "object"}, max_tokens=5, timeout_s=3.0, effort="low")
        assert client.seen == {"model": "gpt-x", "max_tokens": 5, "timeout": 3.0,
                               "messages": [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}],
                               "response_format": {"type": "json_schema", "json_schema": {
                                   "name": "evals_output", "schema": {"type": "object"}, "strict": True}}}
        assert r == LLMResponse(text='{"verdict": "pass", "explanation": "ok"}', model="gpt-x-2026", id="chatcmpl-9x",
                                input_tokens=40, output_tokens=9, finish_reason="stop")

    def test_options_mean_the_same_as_for_the_http_provider(self):
        client = OpenAISpy()
        OpenAIClientProvider(client, token_param="max_completion_tokens", reasoning_effort=True,
                             request_overrides={"temperature": 0}).complete(model="m", messages=[Message("user", "hi")],
                                                                            max_tokens=7, effort="high")
        assert client.seen["max_completion_tokens"] == 7 and client.seen["reasoning_effort"] == "high"
        assert client.seen["temperature"] == 0 and "max_tokens" not in client.seen and "timeout" not in client.seen

    def test_azure_client_name_and_default_model(self, monkeypatch):
        monkeypatch.setenv("OPENAI_MODEL", "my-deployment")
        p = OpenAIClientProvider(AzureOpenAI())
        assert p.name == "azure.ai.openai" and p.default_model == "my-deployment"
        assert OpenAIClientProvider(OpenAISpy(), name="litellm", default_model="d").name == "litellm"

    def test_refusal_and_finish_reasons(self):
        r = OpenAIClientProvider(OpenAISpy(_completion(content=None, refusal="I cannot"))).complete(
            model="m", messages=[Message("user", "hi")])
        assert r.finish_reason == "content_filter" and r.text == ""
        r = OpenAIClientProvider(OpenAISpy(_completion(finish="length"))).complete(model="m", messages=[Message("user", "hi")])
        assert r.finish_reason == "length"

    @pytest.mark.parametrize("exc, cls", [
        (openai.APITimeoutError(request=REQ), LLMTimeout),
        (openai.RateLimitError("slow", response=httpx.Response(429, request=REQ), body=None), LLMRateLimited),
        (openai.InternalServerError("boom", response=httpx.Response(500, request=REQ), body=None), LLMProviderError),
    ])
    def test_sdk_exceptions_become_llm_errors(self, exc, cls):
        with pytest.raises(cls):
            OpenAIClientProvider(OpenAISpy(raises=exc)).complete(model="m", messages=[Message("user", "hi")])

    def test_other_exceptions_propagate_untouched(self):
        with pytest.raises(RuntimeError):
            OpenAIClientProvider(OpenAISpy(raises=RuntimeError("bug"))).complete(model="m", messages=[Message("user", "hi")])

    def test_evals_accepts_the_client_directly(self):
        client = OpenAISpy()
        verdicts = Evals(judge_provider=client, judge_model="gpt-x").evaluate(TURN)
        v = verdicts["Relevance"]
        assert v.passed is True and v.calls[0].provider == "openai" and v.calls[0].model == "gpt-x"
        assert v.calls[0].response_model == "gpt-x-2026" and v.calls[0].input_tokens == 40


class TestCallableProvider:
    def test_mapping_string_and_response_returns(self):
        seen = {}

        def fn(**kw):
            seen.update(kw)
            return {"verdict": "pass", "explanation": "fine"}

        p = CallableProvider(fn, name="gateway", default_model="gw-1")
        r = p.complete(model="gw-1", messages=[Message("user", "hi")], system="s", schema={"type": "object"})
        assert json.loads(r.text) == {"verdict": "pass", "explanation": "fine"} and r.finish_reason == "stop"
        assert seen["model"] == "gw-1" and seen["system"] == "s" and seen["messages"] == [Message("user", "hi")]
        assert CallableProvider(lambda **kw: "text").complete(model="m", messages=[]).text == "text"
        full = LLMResponse(text="t", model="x", finish_reason="length")
        assert CallableProvider(lambda **kw: full).complete(model="m", messages=[]) is full

    def test_llm_errors_from_the_function_map_to_error_types(self):
        def fn(**kw):
            raise LLMTimeout("slow gateway")

        r = RelevanceEvaluator(CallableProvider(fn), model="m").evaluate(TURN)
        assert r.error_type == vocab.ERR_TIMEOUT and "slow gateway" in r.explanation

    def test_judge_with_a_callable(self):
        def fn(**kw):
            return {"verdict": "pass", "explanation": "ok"}

        vs = Evals(judge_provider=CallableProvider(fn, name="my-gateway", default_model="m1")).evaluate(TURN)
        assert vs["Relevance"].label == "pass" and vs["Relevance"].calls[0].provider == "my-gateway"


class TestConfiguration:
    def test_judge_model_is_required_when_the_provider_has_no_default(self):
        with pytest.raises(ValueError, match="judge_model= is required"):
            Evals(judge_provider=CallableProvider(lambda **kw: "x"))

    def test_judge_model_from_the_environment(self, monkeypatch):
        monkeypatch.setenv("EVALS_JUDGE_MODEL", "m-judge")
        p = CallableProvider(lambda **kw: {"verdict": "pass", "explanation": "ok"})
        ev = Evals(judge_provider=p)
        assert ev.judge_provider is p and ev.judge_model == "m-judge"
        assert ev.evaluate(TURN)["Relevance"].calls[0].model == "m-judge"

    def test_evaluators_need_no_credentials_to_build(self, monkeypatch):
        for k in ("EVALS_PROVIDER", "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "OPENAI_BASE_URL"):
            monkeypatch.delenv(k, raising=False)
        ev = RelevanceEvaluator()   # no provider, no key: fine until the first call
        with pytest.raises(RuntimeError, match="no LLM provider configured"):
            ev.provider

    def test_refusal_is_an_llm_error(self):
        r = RelevanceEvaluator(FixedJudge("pass", stop_reason="refusal")).evaluate(TURN)
        assert r.error_type == vocab.ERR_PROVIDER and isinstance(LLMRefusal("x"), LLMError)
