"""AgentLens: regression testing and evaluation toolkit for LLM agents.

Record agent runs as traces, replay them deterministically, score them with
deterministic checks and LLM-as-judge, and catch regressions in CI.
"""

__version__ = "0.0.1"

from agentlens.schema import SpanKind, Trace
from agentlens.tracing import current_span, current_trace, run, span, trace

__all__ = [
    "SpanKind",
    "Trace",
    "__version__",
    "current_span",
    "current_trace",
    "run",
    "span",
    "trace",
]
