"""OpenAI-style chat completions, two ways.

`OpenAICompatibleProvider` speaks `POST <base_url>/chat/completions` over the
standard library: OpenAI, Azure OpenAI (api-key header + api-version query),
vLLM, Ollama, LiteLLM and other gateways, with no SDK installed.
`OpenAIClientProvider` wraps a client from the `openai` SDK (`OpenAI()`,
`AzureOpenAI()`, or one you configured with a token provider or a proxy);
`as_provider` picks it whenever an object has `.chat.completions.create`.
Both build the same request: structured output through `response_format:
json_schema` with `strict: true`, which the evaluators' schemas satisfy
(closed objects, every property required).
"""

from __future__ import annotations

import json
import os
import socket
import urllib.error
import urllib.request
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm import LLMProviderError, LLMRateLimited, LLMResponse, LLMTimeout, Provider
from ..types import Message

FINISH_REASONS = {
    "stop": "stop",
    "length": "length",
    "tool_calls": "tool_calls",
    "function_call": "tool_calls",
    "content_filter": "content_filter",
}


class OpenAICompatibleProvider(Provider):
    """`OpenAICompatibleProvider()` reads OPENAI_BASE_URL (default https://api.openai.com/v1) and OPENAI_API_KEY.

        OpenAICompatibleProvider(base_url="https://<res>.openai.azure.com/openai/deployments/<dep>",
                                 api_key=..., api_key_header="api-key", query={"api-version": "2024-10-21"},
                                 name="azure.ai.openai", default_model="<deployment>")
        OpenAICompatibleProvider(base_url="http://vllm:8000/v1", name="vllm", default_model="...")

    `token_param` is "max_tokens" (widest compatibility) or "max_completion_tokens"
    (OpenAI reasoning models). `reasoning_effort=True` forwards `effort` as
    `reasoning_effort`. `request_overrides` is merged into every request body.
    """

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        *,
        name: str = "openai",
        default_model: str | None = None,
        api_key_header: str = "Authorization",
        headers: Mapping[str, str] | None = None,
        query: Mapping[str, str] | None = None,
        token_param: str = "max_tokens",
        reasoning_effort: bool = False,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        env = os.environ.get
        self.base_url = (base_url or env("OPENAI_BASE_URL") or "https://api.openai.com/v1").rstrip("/")
        self.name = name
        self.default_model = default_model or env("OPENAI_MODEL")
        self._headers = {"Content-Type": "application/json", **(headers or {})}
        key = api_key or env("OPENAI_API_KEY")
        if key:
            self._headers[api_key_header] = f"Bearer {key}" if api_key_header.lower() == "authorization" else key
        self._query = "&".join(f"{k}={v}" for k, v in (query or {}).items())
        self._token_param = token_param
        self._reasoning_effort = reasoning_effort
        self._overrides = dict(request_overrides or {})

    @property
    def url(self) -> str:
        return f"{self.base_url}/chat/completions" + (f"?{self._query}" if self._query else "")

    def request_body(self, *, model, messages, system, schema, max_tokens, effort) -> dict:
        return request_body(model=model, messages=messages, system=system, schema=schema, max_tokens=max_tokens,
                            effort=effort if self._reasoning_effort else None, token_param=self._token_param,
                            overrides=self._overrides)

    def complete(self, *, model, messages: Sequence[Message], system=None, schema=None, max_tokens=1024,
                 timeout_s=None, effort=None) -> LLMResponse:
        body = self.request_body(model=model, messages=messages, system=system, schema=schema,
                                 max_tokens=max_tokens, effort=effort)
        req = urllib.request.Request(self.url, data=json.dumps(body).encode(), headers=self._headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as resp:
                data = json.loads(resp.read())
        except urllib.error.HTTPError as exc:
            detail = exc.read()[:300].decode("utf-8", "replace")
            if exc.code == 429:
                raise LLMRateLimited(f"HTTP 429: {detail}") from None
            if exc.code in (408, 504):
                raise LLMTimeout(f"HTTP {exc.code}: {detail}") from None
            raise LLMProviderError(f"HTTP {exc.code}: {detail}") from None
        except urllib.error.URLError as exc:
            if isinstance(exc.reason, (socket.timeout, TimeoutError)):
                raise LLMTimeout(str(exc.reason)) from None
            raise LLMProviderError(str(exc.reason)) from None
        except (socket.timeout, TimeoutError) as exc:
            raise LLMTimeout(str(exc)) from None
        except json.JSONDecodeError as exc:
            raise LLMProviderError(f"response is not JSON: {exc}") from None

        return parse_completion(data)


class OpenAIClientProvider(Provider):
    """A client from the `openai` SDK: `OpenAI()`, `AzureOpenAI()`, or any object with `.chat.completions.create`.

        OpenAIClientProvider(openai.OpenAI())                       # or just Evals(judge_provider=openai.OpenAI())
        OpenAIClientProvider(openai.AzureOpenAI(...), default_model="<deployment>")

    Same request as `OpenAICompatibleProvider` (`token_param`, `reasoning_effort`,
    `request_overrides` mean the same); the SDK's exceptions are mapped to the
    spec error types. `default_model` falls back to OPENAI_MODEL.
    """

    def __init__(
        self,
        client: Any,
        *,
        name: str | None = None,
        default_model: str | None = None,
        token_param: str = "max_tokens",
        reasoning_effort: bool = False,
        request_overrides: Mapping[str, Any] | None = None,
    ) -> None:
        self.client = client
        self.name = name or ("azure.ai.openai" if "azure" in type(client).__name__.lower() else "openai")
        self.default_model = default_model or os.environ.get("OPENAI_MODEL")
        self._token_param = token_param
        self._reasoning_effort = reasoning_effort
        self._overrides = dict(request_overrides or {})

    def complete(self, *, model, messages: Sequence[Message], system=None, schema=None, max_tokens=1024,
                 timeout_s=None, effort=None) -> LLMResponse:
        kwargs = request_body(model=model, messages=messages, system=system, schema=schema, max_tokens=max_tokens,
                              effort=effort if self._reasoning_effort else None, token_param=self._token_param,
                              overrides=self._overrides)
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        try:
            response = self.client.chat.completions.create(**kwargs)
        except Exception as exc:
            translated = _translate(exc)
            if translated is None:
                raise
            raise translated from exc
        data = response.model_dump() if hasattr(response, "model_dump") else response
        if not isinstance(data, Mapping):
            raise LLMProviderError(f"unexpected response type {type(response).__name__}")
        return parse_completion(data)


def request_body(*, model, messages, system, schema, max_tokens, effort, token_param="max_tokens",
                 overrides=None) -> dict:
    """The chat-completions request both providers send (also the SDK's keyword arguments)."""
    body: dict[str, Any] = {"model": model, "messages": [], token_param: max_tokens}
    if system is not None:
        body["messages"].append({"role": "system", "content": system})
    body["messages"] += [{"role": m.role, "content": m.text} for m in messages]
    if schema is not None:
        body["response_format"] = {"type": "json_schema",
                                   "json_schema": {"name": "evals_output", "schema": schema, "strict": True}}
    if effort:
        body["reasoning_effort"] = effort
    body.update(overrides or {})
    return body


def parse_completion(data: Mapping) -> LLMResponse:
    """A chat-completions response (JSON, or the SDK object dumped to a dict) -> LLMResponse."""
    try:
        choice = data["choices"][0]
        message = choice.get("message") or {}
    except (KeyError, IndexError, TypeError) as exc:
        raise LLMProviderError(f"response has no choices: {str(data)[:200]}") from exc
    finish = choice.get("finish_reason")
    text = message.get("content") or ""
    if message.get("refusal"):
        finish, text = "content_filter", ""
    usage = data.get("usage") or {}
    return LLMResponse(
        text=text if isinstance(text, str) else "".join(p.get("text", "") for p in text),
        model=data.get("model"),
        id=data.get("id"),
        input_tokens=usage.get("prompt_tokens"),
        output_tokens=usage.get("completion_tokens"),
        finish_reason=FINISH_REASONS.get(finish, finish),
    )


def _translate(exc: BaseException):
    """openai exception -> LLMError, or None when it is not one (a bug in a test double, say)."""
    try:
        import openai
    except ImportError:  # pragma: no cover - a fake client raised something without the SDK installed
        return None
    if isinstance(exc, openai.APITimeoutError):
        return LLMTimeout(str(exc))
    if isinstance(exc, openai.RateLimitError):
        return LLMRateLimited(str(exc))
    if isinstance(exc, openai.APIError):
        return LLMProviderError(str(exc))
    return None
