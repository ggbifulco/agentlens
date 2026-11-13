"""Recording agent runs as traces (``docs/design.md`` §2, FR-T1..FR-T5).

Typical use::

    import agentlens

    @agentlens.wrap_tool
    def lookup_order(order_id: str) -> dict[str, str]: ...

    llm = agentlens.wrap_llm(MockLLM(["Your order shipped."]))

    with agentlens.run("support-agent", input="Where is order 1234?") as trace:
        trace.output = my_agent(llm, "Where is order 1234?")
"""

from agentlens.tracing.api import current_span, current_trace, run, span, trace
from agentlens.tracing.recorder import Recorder, TraceSink
from agentlens.tracing.redaction import (
    DEFAULT_PATTERNS,
    DEFAULT_REDACTOR,
    DEFAULT_SENSITIVE_KEYS,
    REDACTED,
    PatternRedactor,
    Redactor,
    redact_trace,
)
from agentlens.tracing.wrappers import TOOL_NAME_ATTR, TracedLLM, wrap_llm, wrap_tool

__all__ = [
    "DEFAULT_PATTERNS",
    "DEFAULT_REDACTOR",
    "DEFAULT_SENSITIVE_KEYS",
    "REDACTED",
    "TOOL_NAME_ATTR",
    "PatternRedactor",
    "Recorder",
    "Redactor",
    "TraceSink",
    "TracedLLM",
    "current_span",
    "current_trace",
    "redact_trace",
    "run",
    "span",
    "trace",
    "wrap_llm",
    "wrap_tool",
]
