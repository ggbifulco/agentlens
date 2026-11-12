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

__all__ = ["Recorder", "TraceSink", "current_span", "current_trace", "run", "span", "trace"]
