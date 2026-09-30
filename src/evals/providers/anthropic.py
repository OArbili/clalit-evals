"""Anthropic SDK adapter. Imports `anthropic` lazily, so the package works without it."""

from __future__ import annotations

import os
from collections.abc import Mapping, Sequence
from typing import Any

from ..llm import LLMProviderError, LLMRateLimited, LLMResponse, LLMTimeout, Provider
from ..types import Message

# Anthropic's stop reasons -> the GenAI finish-reason vocabulary. Covers every value
# anthropic 1.3.0 can return; a future value passes through unchanged.
STOP_REASONS = {
    "end_turn": "stop",
    "stop_sequence": "stop",
    "pause_turn": "stop",
    "max_tokens": "length",
    "model_context_window_exceeded": "length",
    "tool_use": "tool_calls",
    "refusal": "content_filter",
}


class AnthropicProvider(Provider):
    """`anthropic.Anthropic()` by default; pass any client from that SDK (Bedrock, Vertex) or a test double.

    With no client, reads ANTHROPIC_API_KEY and, for identity-linked keys,
    ANTHROPIC_WORKSPACE_ID (sent as the anthropic-workspace-id header).
    """

    name = "anthropic"
    default_model = "claude-opus-4-8"

    def __init__(self, client: Any = None, *, default_model: str | None = None, workspace_id: str | None = None) -> None:
        if client is None:
            import anthropic  # lazy: the package is optional

            workspace = workspace_id or os.environ.get("ANTHROPIC_WORKSPACE_ID")
            client = anthropic.Anthropic(default_headers={"anthropic-workspace-id": workspace} if workspace else None)
        self.client = client
        if default_model:
            self.default_model = default_model

    def complete(self, *, model, messages: Sequence[Message], system=None, schema: Mapping | None = None,
                 max_tokens=1024, timeout_s=None, effort=None) -> LLMResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": max_tokens,
            "messages": [{"role": m.role, "content": m.text} for m in messages],
        }
        if system is not None:
            kwargs["system"] = system
        output_config: dict[str, Any] = {}
        if schema is not None:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if effort:
            output_config["effort"] = effort
        if output_config:
            kwargs["output_config"] = output_config
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s

        try:
            response = self.client.messages.create(**kwargs)
        except Exception as exc:
            translated = _translate(exc)
            if translated is None:
                raise
            raise translated from exc

        text = "".join(b.text for b in response.content if getattr(b, "type", None) == "text")
        usage = getattr(response, "usage", None)
        stop = getattr(response, "stop_reason", None)
        return LLMResponse(
            text=text,
            model=getattr(response, "model", None),
            id=getattr(response, "id", None),
            input_tokens=getattr(usage, "input_tokens", None),
            output_tokens=getattr(usage, "output_tokens", None),
            finish_reason=STOP_REASONS.get(stop, stop),
        )


def _translate(exc: BaseException):
    """Anthropic exception -> LLMError, or None when it is not one (a bug in a test double, say)."""
    try:
        import anthropic
    except ImportError:  # pragma: no cover - a fake client raised something without the SDK installed
        return None
    if isinstance(exc, anthropic.APITimeoutError):
        return LLMTimeout(str(exc))
    if isinstance(exc, anthropic.RateLimitError):
        return LLMRateLimited(str(exc))
    if isinstance(exc, anthropic.APIError):
        return LLMProviderError(str(exc))
    return None
