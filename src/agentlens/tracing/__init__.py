"""Recording agent runs as traces (``docs/design.md`` §2, FR-T1/FR-T5).

Typical use::

    import agentlens

    @agentlens.trace(kind="tool_call")
    def lookup_order(order_id: str) -> dict[str, str]: ...

    with agentlens.run("support-agent", input="Where is order 1234?") as trace:
        trace.output = my_agent("Where is order 1234?")
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

__all__ = [
    "DEFAULT_PATTERNS",
    "DEFAULT_REDACTOR",
    "DEFAULT_SENSITIVE_KEYS",
    "REDACTED",
    "PatternRedactor",
    "Recorder",
    "Redactor",
    "TraceSink",
    "current_span",
    "current_trace",
    "redact_trace",
    "run",
    "span",
    "trace",
]
