"""Helpers to build traces/experiments with known metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from agentlens.compare import ExperimentMetrics, RunMetrics
from agentlens.schema import EvalResult, Experiment, Span, SpanKind, Trace, Usage

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


def make_trace(
    case_id: str,
    repetition: int = 0,
    *,
    error: str | None = None,
    cost: float | None = 0.01,
    latency_ms: float = 100.0,
    steps: int = 2,
    offset_s: float = 0.0,
) -> Trace:
    """A finished trace with ``steps`` LLM calls sharing ``cost``."""
    start = T0 + timedelta(seconds=offset_s)
    trace = Trace(
        name="suite",
        start_time=start,
        end_time=start + timedelta(milliseconds=latency_ms),
        error=error,
        metadata={"case_id": case_id, "repetition": repetition},
    )
    trace.spans = [
        Span(
            trace_id=trace.id,
            kind=SpanKind.LLM_CALL,
            name="mock",
            start_time=start,
            end_time=start,
            usage=Usage(input_tokens=1, cost_usd=None if cost is None else cost / steps),
        )
        for _ in range(steps)
    ]
    return trace


def result(evaluator: str, score: float, passed: bool | None = None) -> EvalResult:
    return EvalResult(
        evaluator=evaluator, score=score, passed=score >= 0.5 if passed is None else passed
    )


def run(
    case_id: str,
    *,
    passed: bool = True,
    cost: float | None = 0.01,
    latency_ms: float | None = 100.0,
    steps: int = 2,
    scores: dict[str, float] | None = None,
    repetition: int = 0,
) -> RunMetrics:
    return RunMetrics(
        case_id=case_id,
        repetition=repetition,
        passed=passed,
        error=None,
        cost_usd=cost,
        latency_ms=latency_ms,
        steps=steps,
        scores=scores or {},
    )


def metrics(name: str, runs: list[RunMetrics]) -> ExperimentMetrics:
    return ExperimentMetrics.from_runs(Experiment(name=name, suite="suite"), runs)
