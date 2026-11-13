"""Helpers for built-in evaluator tests."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import UTC, datetime, timedelta

from pydantic import JsonValue

from agentlens.schema import Case, Span, SpanKind, Trace, Usage


def make_trace(output: JsonValue = None, **fields: object) -> Trace:
    """A finished trace with the given output."""
    return Trace.model_validate({"name": "agent", "input": "q", "output": output, **fields})


def make_case(expected: JsonValue = None) -> Case:
    """A case with the given expectation."""
    return Case(id="c1", input="q", expected=expected)


T0 = datetime(2026, 1, 1, tzinfo=UTC)


def build_trace(
    tools: Sequence[str] = (),
    *,
    llm_calls: int = 0,
    agent_steps: int = 0,
    costs: Sequence[float | None] = (),
    latency_ms: float | None = 1000.0,
) -> Trace:
    """A synthetic run: tool, LLM and step spans one second apart, in that order.

    ``costs`` gives one LLM span with usage per entry (``None`` = unpriced);
    ``latency_ms=None`` leaves the run unfinished.
    """
    trace_id = "t1"
    spans: list[Span] = []

    def add(kind: SpanKind, name: str, usage: Usage | None = None) -> None:
        start = T0 + timedelta(seconds=len(spans))
        spans.append(
            Span(
                id=f"s{len(spans)}",
                trace_id=trace_id,
                kind=kind,
                name=name,
                start_time=start,
                end_time=start + timedelta(milliseconds=500),
                usage=usage,
            )
        )

    for tool in tools:
        add(SpanKind.TOOL_CALL, tool)
    for _ in range(llm_calls):
        add(SpanKind.LLM_CALL, "mock-model")
    for cost in costs:
        add(SpanKind.LLM_CALL, "priced-model", Usage(input_tokens=10, cost_usd=cost))
    for i in range(agent_steps):
        add(SpanKind.AGENT_STEP, f"step{i}")
    end = None if latency_ms is None else T0 + timedelta(milliseconds=latency_ms)
    return Trace(
        id=trace_id, name="agent", input="q", output="a", start_time=T0, end_time=end, spans=spans
    )
