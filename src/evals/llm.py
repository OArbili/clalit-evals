"""The LLM provider contract. The only thing the evaluators and Evals know about a model API.

A provider turns (model, messages, optional system prompt, optional JSON
schema) into an `LLMResponse`, and raises one of the `LLMError` subclasses on
failure. Everything provider-specific -- SDKs, wire formats, stop-reason
vocabularies, exception classes -- lives in `evals/providers/`.
"""

from __future__ import annotations

import abc
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

from . import vocab
from .types import Message


@dataclass(frozen=True)
class LLMResponse:
    """A completed model call, normalised.

    `finish_reason` uses the GenAI vocabulary: stop | length | tool_calls |
    content_filter | error (a provider value outside it passes through).
    """

    text: str
    model: str | None = None
    id: str | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    finish_reason: str | None = None


class LLMError(Exception):
    """Base of the provider failures the evaluators understand; `error_type` is a spec error type."""

    error_type = vocab.ERR_PROVIDER


class LLMTimeout(LLMError):
    error_type = vocab.ERR_TIMEOUT


class LLMRateLimited(LLMError):
    error_type = vocab.ERR_RATE_LIMIT


class LLMProviderError(LLMError):
    error_type = vocab.ERR_PROVIDER


class LLMRefusal(LLMError):
    """The model declined to answer; a provider-side outcome, not a failing turn."""

    error_type = vocab.ERR_PROVIDER


class Provider(abc.ABC):
    """One model API.

    `name` is the `gen_ai.provider.name` value ("anthropic", "openai",
    "azure.ai.openai", ...). `default_model` is used when the caller gives no
    model; None means the caller must name one.
    """

    name: str = "unknown"
    default_model: str | None = None

    @abc.abstractmethod
    def complete(
        self,
        *,
        model: str,
        messages: Sequence[Message],
        system: str | None = None,
        schema: Mapping | None = None,
        max_tokens: int = 1024,
        timeout_s: float | None = None,
        effort: str | None = None,
    ) -> LLMResponse:
        """One completion. With `schema`, the text must be JSON matching it (structured output)."""
