"""AgentLens: regression testing and evaluation toolkit for LLM agents.

Record agent runs as traces, replay them deterministically, score them with
deterministic checks and LLM-as-judge, and catch regressions in CI.
"""

__version__ = "0.1.0"

from agentlens.evals import Evaluator, evaluator
from agentlens.llm import get_client
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
    wrap_llm,
    wrap_tool,
)

__all__ = [
    "Evaluator",
    "PatternRedactor",
    "Redactor",
    "SpanKind",
    "Trace",
    "__version__",
    "current_span",
    "current_trace",
    "evaluator",
    "get_client",
    "redact_trace",
    "run",
    "span",
    "trace",
    "wrap_llm",
    "wrap_tool",
]
