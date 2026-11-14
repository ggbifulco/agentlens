"""Builders for compare/gates tests: runs with chosen metrics."""

from __future__ import annotations

import itertools
from datetime import UTC, datetime, timedelta

from agentlens.schema import EvalResult, Span, SpanKind, Trace, Usage
from agentlens.suite import RunResult

T0 = datetime(2026, 1, 1, tzinfo=UTC)
_ids = itertools.count()


def run(
    case_id: str,
    repetition: int = 0,
    *,
    passed: bool | dict[str, bool] = True,
    score: float | dict[str, float] | None = None,
    cost: float | None = None,
    latency_ms: float = 100.0,
    steps: int = 1,
    error: str | None = None,
) -> RunResult:
    """A run of ``case_id`` with the given metrics (one LLM span per step)."""
    trace_id = f"{case_id}-{repetition}-{next(_ids)}"  # unique: the store upserts by id
    spans = [
        Span(
            trace_id=trace_id,
            kind=SpanKind.LLM_CALL,
            name="mock",
            start_time=T0,
            end_time=T0,
            usage=Usage(cost_usd=cost / steps if cost is not None else None),
        )
        for _ in range(steps)
    ]
    trace = Trace(
        id=trace_id,
        name="toy",
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=latency_ms),
        spans=spans,
        error=error,
        metadata={"case_id": case_id, "repetition": repetition},
    )
    verdicts = passed if isinstance(passed, dict) else {"check": passed}
    scores = score if isinstance(score, dict) else {}
    results = [
        EvalResult(
            evaluator=name,
            passed=ok,
            score=scores.get(name, score if isinstance(score, float) else float(ok)),
        )
        for name, ok in verdicts.items()
    ]
    return RunResult(case_id=case_id, repetition=repetition, trace=trace, eval_results=results)


def cases(n: int, **kwargs: object) -> list[RunResult]:
    return [run(f"c{i}", **kwargs) for i in range(n)]  # type: ignore[arg-type]
