"""Turn-level evaluators for an LLM agent. Evaluation logic only.

Repetition (deterministic), nine LLM-as-judge dimensions (Relevance,
Naturalness, Empathy, Attentiveness, Coherence, ContextRetention,
AppropriateQuestioning, Adaptability, NonRobotic), three clinical checks on
the same rubric (QuestionsBeforeRouting, RedFlagsRuledOut, FocusedQuestioning)
and two reference-based summary evaluators (SummaryFactCapture,
SummaryFactPrecision).

Headline name: `Evals`. Everything else is a return type (Turn, Verdict,
Verdicts, JudgeCall) or a Provider from evals.providers. No model SDK
appears in user code: the judges speak to a `Provider` (Anthropic, any
OpenAI-compatible endpoint, or your own callable). The package emits no
telemetry and stores nothing; every Verdict carries the record its host needs
to report it, and the REST service (evals_service) is the host that does.

    from evals import Evals

    ev = Evals()                       # the judge from ANTHROPIC_API_KEY or OPENAI_*
    verdicts = ev.evaluate(user_input="...", agent_output="...", response_id="msg_...")
    print(verdicts)

Logging: the "evals" logger (INFO: setup; WARNING: error verdicts and failed
judge calls; DEBUG: every verdict and judge call). Nothing else is emitted.
See EVALS_REQUIREMENTS.md for the specification.
"""

from importlib.metadata import PackageNotFoundError, version as _dist_version

from .core import Evals
from .types import JudgeCall, Message, Turn, Verdict, Verdicts

try:
    __version__ = _dist_version("clalit-evals")   # the distribution name; the import name stays `evals`
except PackageNotFoundError:                       # run from a checkout without an install
    __version__ = "0+unknown"

__all__ = ["Evals", "Turn", "Verdicts", "Verdict", "Message", "JudgeCall", "__version__"]
