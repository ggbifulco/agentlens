"""AgentLens: regression testing and evaluation toolkit for LLM agents.

Record agent runs as traces, replay them deterministically, score them with
deterministic checks and LLM-as-judge, and catch regressions in CI.
"""

__version__ = "0.0.1"

from agentlens.schema import SpanKind, Trace
from agentlens.tracing import (
    PatternRedactor,
    Redactor,
    current_span,
    current_trace,
    redact_trace,
    run,
    span,
    trace,
)

__all__ = [
    "PatternRedactor",
    "Redactor",
    "SpanKind",
    "Trace",
    "__version__",
    "current_span",
    "current_trace",
    "redact_trace",
    "run",
    "span",
    "trace",
]
