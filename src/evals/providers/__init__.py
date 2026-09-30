"""Provider adapters. The core imports only `as_provider` and `default_provider` from here.

    anthropic.py           the Anthropic SDK (also Bedrock / Vertex clients from that SDK)
    openai_compatible.py   any OpenAI-compatible chat-completions endpoint: OpenAI, Azure OpenAI,
                           vLLM, Ollama, LiteLLM and other gateways -- standard library HTTP;
                           and OpenAIClientProvider, a client from the openai SDK
    callable_provider.py   your own function, for gateways with no standard shape
"""

from __future__ import annotations

import os
from typing import Any

from ..llm import LLMError, LLMProviderError, LLMRateLimited, LLMRefusal, LLMResponse, LLMTimeout, Provider
from .anthropic import AnthropicProvider
from .callable_provider import CallableProvider
from .openai_compatible import OpenAIClientProvider, OpenAICompatibleProvider

__all__ = [
    "Provider", "LLMResponse", "LLMError", "LLMTimeout", "LLMRateLimited", "LLMProviderError", "LLMRefusal",
    "AnthropicProvider", "OpenAICompatibleProvider", "OpenAIClientProvider", "CallableProvider", "as_provider",
    "default_provider",
]


def as_provider(obj: Any) -> Provider | None:
    """Accept a Provider, an Anthropic client (`.messages.create`), an OpenAI client (`.chat.completions.create`), or a callable.

    None stays None (the caller resolves a default later). Anything else is a
    TypeError: better than a confusing failure on the first model call.
    """
    if obj is None or isinstance(obj, Provider):
        return obj
    messages = getattr(obj, "messages", None)
    if messages is not None and callable(getattr(messages, "create", None)):
        return AnthropicProvider(obj)
    completions = getattr(getattr(obj, "chat", None), "completions", None)
    if completions is not None and callable(getattr(completions, "create", None)):
        return OpenAIClientProvider(obj)
    if callable(obj):
        return CallableProvider(obj)
    raise TypeError(f"expected a Provider, an Anthropic or OpenAI client, or a callable, got {type(obj).__name__}")


def default_provider() -> Provider:
    """The provider the environment describes.

    EVALS_PROVIDER=anthropic | openai chooses explicitly; otherwise the first of
    ANTHROPIC_API_KEY, OPENAI_API_KEY that is set decides.
    """
    env = os.environ.get
    choice = (env("EVALS_PROVIDER") or "").lower()
    if choice == "anthropic" or (not choice and env("ANTHROPIC_API_KEY")):
        return AnthropicProvider()
    if choice == "openai" or (not choice and (env("OPENAI_API_KEY") or env("OPENAI_BASE_URL"))):
        return OpenAICompatibleProvider()
    if choice:
        raise ValueError(f"EVALS_PROVIDER={choice!r}: expected 'anthropic' or 'openai'")
    raise RuntimeError(
        "no LLM provider configured: set ANTHROPIC_API_KEY or OPENAI_API_KEY (EVALS_PROVIDER picks between them), "
        "or pass provider= / judge_provider= to Evals"
    )
