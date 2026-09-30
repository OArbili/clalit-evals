"""Wrap a function as a provider, for gateways with no standard shape."""

from __future__ import annotations

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from ..llm import LLMResponse, Provider
from ..types import Message


class CallableProvider(Provider):
    """`fn(*, model, system, messages, schema, max_tokens, timeout_s, effort)` -> str | Mapping | LLMResponse.

    Return the model's text (JSON text when a schema was given), a mapping that
    is serialised for you, or a full `LLMResponse`. Raise an `LLMError`
    subclass for failures you want mapped to a spec error type; any other
    exception is reported as provider_error.
    """

    def __init__(self, fn: Callable[..., Any], *, name: str = "custom", default_model: str | None = None) -> None:
        self._fn = fn
        self.name = name
        self.default_model = default_model

    def complete(self, *, model, messages: Sequence[Message], system=None, schema: Mapping | None = None,
                 max_tokens=1024, timeout_s=None, effort=None) -> LLMResponse:
        out = self._fn(model=model, system=system, messages=list(messages), schema=schema,
                       max_tokens=max_tokens, timeout_s=timeout_s, effort=effort)
        if isinstance(out, LLMResponse):
            return out
        if isinstance(out, Mapping):
            return LLMResponse(text=json.dumps(out, ensure_ascii=False), model=model, finish_reason="stop")
        return LLMResponse(text=str(out), model=model, finish_reason="stop")
