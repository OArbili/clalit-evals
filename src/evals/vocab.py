"""The values the package writes into every Verdict. Standard library only.

Score labels, the five error types, the finish-reason set and the attribute
keys of the summary fact evaluators. They follow the OpenTelemetry GenAI
vocabulary so a host can copy them onto a `gen_ai.evaluation.result` event
without re-mapping; the event, span and metric NAMES are not here -- the
package emits no telemetry, so they live with the host that does
(evals_service/semconv.py).
"""

# --- Score labels (gen_ai.evaluation.score.label) -----------------------------
LABEL_PASS = "pass"
LABEL_FAIL = "fail"

# --- Error types (error.type on a verdict that has no score) -------------------
ERR_TIMEOUT = "timeout"
ERR_RATE_LIMIT = "rate_limit"
ERR_INVALID_RESPONSE = "invalid_response"
ERR_PROVIDER = "provider_error"
ERR_MISSING_INPUT = "missing_input"

# The GenAI finish-reason vocabulary. Each provider adapter (evals/providers/)
# normalises its own stop reasons to these; a value outside the set passes
# through unchanged rather than being lost.
FINISH_REASONS = ("stop", "length", "tool_calls", "content_filter", "error")

# --- Verdict.attributes keys of the summary fact evaluators -------------------
# (docs/SUMMARY_FACT_CAPTURE.md section 4). `facts` are statements extracted from
# the human reference, `claims` statements extracted from the LLM summary;
# distinct names so the two never mix.
FACTS_TOTAL = "eval.facts.total"
FACTS_CAPTURED = "eval.facts.captured"
FACTS_MISSING = "eval.facts.missing"            # opt-in: member medical facts
CLAIMS_TOTAL = "eval.claims.total"
CLAIMS_SUPPORTED = "eval.claims.supported"
CLAIMS_UNSUPPORTED = "eval.claims.unsupported"  # opt-in: member medical facts
