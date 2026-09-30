"""Shared plumbing for LLM-as-judge evaluators.

`JudgeBacked` is how an evaluator holds its judge (provider, model, bind, one
recorded call). `LLMJudgeEvaluator` on top of it is the one-prompt judge: ONE
prompt template, ONE output schema, ONE call record and ONE error mapping, so
verdicts across dimensions are produced the same way and the same or similar
inputs are judged by the same rules. A dimension contributes only its
definition, its literal fail criteria, and optionally a deterministic pre-check
that answers without calling the model.
"""

from __future__ import annotations

import json
import logging
import re
import time
from dataclasses import replace

from .. import vocab
from ..llm import LLMError, LLMRefusal, Provider
from ..providers import as_provider, default_provider
from ..types import JudgeCall, Message
from .base import EvalResult, Evaluator, Turn

log = logging.getLogger("evals.judge")

SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "fail"],
            "description": "fail only if a fail criterion is clearly present; otherwise pass.",
        },
        "explanation": {
            "type": "string",
            "description": "One English sentence stating the reason and quoting the decisive words.",
        },
    },
    "required": ["verdict", "explanation"],
    "additionalProperties": False,
}

TEMPLATE = """\
You are a strict, consistent evaluator of ONE reply from a customer-service assistant \
(a health fund's triage assistant).

Dimension under judgement: {name}
Definition: {definition}

Decision rule. Answer "fail" only if at least one of the following is clearly present \
in the assistant's LATEST reply:
{criteria}
Otherwise answer "pass".{pass_note}

Rules that keep verdicts consistent:
1. Judge only the assistant's latest reply. Earlier turns are context, not the thing being judged.
2. Judge only this dimension. Do not reward or penalise anything the criteria above do not \
name (accuracy, safety, helpfulness, length, relevance).
3. Apply the criteria literally to the text. Do not infer intent, tone or facts that are not \
in the text. Identical inputs must yield identical verdicts.
4. Ignore markdown, links and formatting.
5. The text may be Hebrew or English; judge it in its own language. Write the explanation in \
English, in one sentence, quoting the decisive words as plain text, never as escape sequences.
6. If the situation the criteria describe does not arise in this reply, answer "pass" and say so.
"""


class JudgeBacked(Evaluator):
    """An evaluator that consults the judge model. Holds the judge, nothing else.

    `provider` is a Provider, an Anthropic-style client, or a callable (see
    evals.providers.as_provider); None means the environment's default provider,
    resolved on first use so that building an evaluator needs no credentials.
    `model` None means the provider's default model. `bind()` fills in whatever
    was not given explicitly with the Evals instance's shared judge. Subclasses
    make every model call through `ask_judge()`, so each one is recorded as a
    JudgeCall the same way (LLMJudgeEvaluator below is the one-prompt case;
    summary_facts.py the two-prompt case).
    """

    def __init__(
        self,
        provider: Provider | object | None = None,
        model: str | None = None,
        effort: str = "low",
        timeout: float = 30.0,
    ) -> None:
        self._provider = as_provider(provider)
        self._model = model
        self._effort = effort
        self._timeout = timeout

    def bind(self, *, judge_provider=None, judge_model=None) -> None:
        if self._provider is None and judge_provider is not None:
            self._provider = as_provider(judge_provider)
        if self._model is None and judge_model:
            self._model = judge_model

    @property
    def provider(self) -> Provider:
        if self._provider is None:
            self._provider = default_provider()
        return self._provider

    @property
    def model(self) -> str:
        model = self._model or self.provider.default_model
        if not model:
            raise ValueError(f"{self.name}: no judge model configured and provider {self.provider.name!r} has no default")
        return model

    def ask_judge(self, *, system: str, user: str, schema: dict, prompt_version: str,
                  calls: list[JudgeCall], max_tokens: int = 1024) -> dict:
        """One structured-output call to this evaluator's judge, recorded on `calls` (see call_judge)."""
        return call_judge(
            self.provider, self.model, self._effort, self._timeout,
            system=system, user=user, schema=schema, prompt_version=prompt_version, calls=calls, max_tokens=max_tokens,
        )


class LLMJudgeEvaluator(JudgeBacked):
    """Base for the one-prompt judge evaluators. Subclasses set the class attributes below.

    definition     one sentence: what a passing reply does
    fail_criteria  literal, observable conditions; any one clearly present = fail
    pass_note      optional sentence appended after the decision rule
    uses_history   whether the judge is shown the conversation before the turn
    prompt_version recorded on every JudgeCall (and by the host as eval.judge.prompt_version)
    """

    definition: str = ""
    fail_criteria: tuple[str, ...] = ()
    pass_note: str = ""
    uses_history: bool = True
    prompt_version: str = "judge/v1"

    # -- what a dimension may override -----------------------------------------------
    # A standard dimension sets the class attributes and, at most, precheck(). An
    # evaluator that wants its own wording overrides system_prompt() and/or
    # user_prompt(); the answer must still be SCHEMA (verdict pass|fail, explanation).

    def precheck(self, turn: Turn) -> EvalResult | None:
        """Return a verdict without calling the model, or None to ask the judge."""
        return None

    @classmethod
    def system_prompt(cls) -> str:
        """The rubric: TEMPLATE with this dimension's name, definition, criteria and note filled in."""
        criteria = "\n".join(f"- {c}" for c in cls.fail_criteria)
        note = f" {cls.pass_note.strip()}" if cls.pass_note else ""
        return TEMPLATE.format(name=cls.name, definition=cls.definition, criteria=criteria, pass_note=note)

    def user_prompt(self, turn: Turn) -> str:
        """The turn as the judge sees it: history (if used), the user message, the reply."""
        parts = []
        if self.uses_history and turn.history:
            history = "\n".join(f"{m.role}: {m.text}" for m in turn.history)
            parts.append(f"<conversation_history>\n{history}\n</conversation_history>")
        parts.append(f"<user_message>\n{turn.user_input}\n</user_message>")
        parts.append(f"<assistant_reply>\n{turn.agent_output}\n</assistant_reply>")
        return "\n\n".join(parts)

    # -- the shared path -------------------------------------------------------------

    def evaluate(self, turn: Turn) -> EvalResult:
        if not turn.agent_output or not turn.user_input:
            missing = "gen_ai.output.messages" if not turn.agent_output else "the last user message"
            return EvalResult.errored(self.name, vocab.ERR_MISSING_INPUT, f"{missing} is absent or empty.")

        pre = self.precheck(turn)
        if pre is not None:
            return pre

        calls: list[JudgeCall] = []
        try:
            verdict = self._ask_judge(turn, calls)
            label = verdict["verdict"]
            explanation = _unescape(verdict["explanation"])
            if label not in (vocab.LABEL_PASS, vocab.LABEL_FAIL):
                raise ValueError(f"judge returned verdict {label!r}")
        except Exception as exc:
            err = judge_error(self.name, exc)
            if err is None:
                raise
            return err.with_calls(calls)

        if label == vocab.LABEL_PASS:
            return EvalResult.passed(self.name, explanation).with_calls(calls)
        return EvalResult.failed(self.name, explanation).with_calls(calls)

    def _ask_judge(self, turn: Turn, calls: list[JudgeCall]) -> dict:
        return self.ask_judge(system=self.system_prompt(), user=self.user_prompt(turn), schema=SCHEMA,
                              prompt_version=self.prompt_version, calls=calls)


def call_judge(provider: Provider, model: str, effort: str, timeout: float, *,
               system: str, user: str, schema: dict, prompt_version: str, calls: list[JudgeCall],
               max_tokens: int = 1024) -> dict:
    """One structured-output judge call, recorded as a `JudgeCall` appended to `calls`.

    Shared by every evaluator that consults a model, so the record, the refusal
    handling and the JSON decoding are identical everywhere and independent of
    the provider. Raises an `LLMError` or a JSON/shape error (the record is
    appended first, with the error on it); callers map those with `judge_error`.
    """
    started = time.time()
    try:
        response = provider.complete(
            model=model, messages=[Message("user", user)], system=system, schema=schema,
            max_tokens=max_tokens, timeout_s=timeout, effort=effort,
        )
    except LLMError as exc:
        calls.append(JudgeCall(provider=provider.name, model=model, prompt_version=prompt_version,
                               started_at=started, ended_at=time.time(), error_type=exc.error_type, error=str(exc)))
        log.warning("judge call failed: model=%s error_type=%s (%s)", model, exc.error_type, exc)
        raise

    call = JudgeCall(
        provider=provider.name, model=model, prompt_version=prompt_version, started_at=started, ended_at=time.time(),
        response_model=response.model, response_id=response.id,
        input_tokens=response.input_tokens, output_tokens=response.output_tokens, finish_reason=response.finish_reason,
    )
    # A judge that declines has no verdict; that is a provider-side failure,
    # not a failing turn. Checked before the text is parsed -- it is empty.
    if response.finish_reason == "content_filter":
        calls.append(replace(call, error_type=vocab.ERR_PROVIDER, error="the judge model declined to answer"))
        log.warning("judge call refused: model=%s", model)
        raise LLMRefusal("the judge model declined to answer")
    try:
        data = json.loads(response.text)
    except json.JSONDecodeError as exc:
        calls.append(replace(call, error_type=vocab.ERR_INVALID_RESPONSE, error=f"reply is not JSON: {exc}"))
        raise
    calls.append(call)
    log.debug("judge call: model=%s prompt=%s tokens=%s/%s %.2fs", model, prompt_version,
              response.input_tokens, response.output_tokens, call.duration_s)
    return data


def judge_error(name: str, exc: BaseException) -> EvalResult | None:
    """Map a judge-call exception to one of the five spec error types, or None if it is not one.

    Provider failures arrive as `LLMError` subclasses carrying their type. A
    reply that is not the schema (bad JSON, missing keys, a value outside the
    enum, a wrong type) is the judge's failure, not the turn's: invalid_response.
    """
    if isinstance(exc, LLMError):
        return EvalResult.errored(name, exc.error_type, str(exc))
    if isinstance(exc, (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError)):
        return EvalResult.errored(name, vocab.ERR_INVALID_RESPONSE, str(exc))
    return None


_ESCAPED = re.compile(r"\\u([0-9a-fA-F]{4})")


def _unescape(text: str) -> str:
    """Turn stray JSON-style \\uXXXX sequences a judge may emit inside a string back into characters."""
    return _ESCAPED.sub(lambda m: chr(int(m.group(1), 16)), text)

