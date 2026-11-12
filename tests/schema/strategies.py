"""Hypothesis strategies that generate valid schema objects.

Reusable by later test modules (store, JSONL export, evaluators) that need
realistic, randomised traces.
"""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from hypothesis import strategies as st
from pydantic import JsonValue

from agentlens.schema import (
    Case,
    EvalResult,
    EvaluatorSpec,
    Experiment,
    Span,
    SpanKind,
    Trace,
    Usage,
    new_id,
)

# JSON can't represent NaN/inf, and the store keeps ints in 64-bit columns.
json_scalars = (
    st.none()
    | st.booleans()
    | st.integers(min_value=-(2**63), max_value=2**63 - 1)
    | st.floats(allow_nan=False, allow_infinity=False)
    | st.text(max_size=20)
)
json_values: st.SearchStrategy[JsonValue] = st.recursive(
    json_scalars,
    lambda children: (
        st.lists(children, max_size=4) | st.dictionaries(st.text(max_size=8), children, max_size=4)
    ),
    max_leaves=12,
)
json_dicts = st.dictionaries(st.text(max_size=8), json_values, max_size=4)

utc_datetimes = st.datetimes(
    min_value=datetime(2000, 1, 1), max_value=datetime(2100, 1, 1), timezones=st.just(UTC)
)
durations = st.timedeltas(min_value=timedelta(0), max_value=timedelta(hours=1))
ids = st.builds(new_id)

usages = st.builds(
    Usage,
    input_tokens=st.integers(min_value=0, max_value=10**7),
    output_tokens=st.integers(min_value=0, max_value=10**7),
    cost_usd=st.none() | st.floats(min_value=0, max_value=1e4, allow_nan=False),
)


@st.composite
def traces(draw: st.DrawFn, max_spans: int = 6) -> Trace:
    """A trace whose spans form a valid tree (parents always exist)."""
    trace_id = draw(ids)
    start = draw(utc_datetimes)
    spans: list[Span] = []
    for _ in range(draw(st.integers(min_value=0, max_value=max_spans))):
        parent_id = draw(st.none() | st.sampled_from([s.id for s in spans])) if spans else None
        span_start = start + draw(durations)
        span_end = draw(st.none() | durations.map(lambda d, t=span_start: t + d))
        spans.append(
            Span(
                trace_id=trace_id,
                parent_id=parent_id,
                kind=draw(st.sampled_from(SpanKind)),
                name=draw(st.text(min_size=1, max_size=12)),
                start_time=span_start,
                end_time=span_end,
                input=draw(json_values),
                output=draw(json_values),
                error=draw(st.none() | st.text(max_size=20)),
                usage=draw(st.none() | usages),
                attributes=draw(json_dicts),
            )
        )
    return Trace(
        id=trace_id,
        name=draw(st.text(min_size=1, max_size=12)),
        input=draw(json_values),
        output=draw(json_values),
        error=draw(st.none() | st.text(max_size=20)),
        start_time=start,
        end_time=draw(st.none() | durations.map(lambda d: start + d)),
        spans=spans,
        metadata=draw(json_dicts),
    )


evaluator_specs = st.builds(
    EvaluatorSpec,
    name=st.text(min_size=1, max_size=12).filter(lambda n: n not in {"name", "params"}),
    params=json_dicts,
)

cases = st.builds(
    Case,
    id=st.text(min_size=1, max_size=12),
    input=json_values,
    expected=json_values,
    evaluators=st.lists(evaluator_specs, max_size=3),
    tags=st.lists(st.text(max_size=8), max_size=3),
)

eval_results = st.builds(
    EvalResult,
    evaluator=st.text(min_size=1, max_size=12),
    score=st.floats(min_value=0, max_value=1),
    passed=st.booleans(),
    reason=st.none() | st.text(max_size=20),
    details=json_dicts,
)

experiments = st.builds(
    Experiment,
    id=ids,
    name=st.text(min_size=1, max_size=12),
    suite=st.text(min_size=1, max_size=12),
    created_at=utc_datetimes,
    config=json_dicts,
    is_baseline=st.booleans(),
)
