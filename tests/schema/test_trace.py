"""Tests for Span/Trace validation and derived metrics."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta, timezone

import pytest
from pydantic import ValidationError

from agentlens.schema import Span, SpanKind, Trace, Usage

T0 = datetime(2026, 11, 12, 9, 0, 0, tzinfo=UTC)


def _span(trace_id: str, kind: SpanKind, name: str, offset_ms: int, **kw: object) -> Span:
    start = T0 + timedelta(milliseconds=offset_ms)
    return Span(
        trace_id=trace_id,
        kind=kind,
        name=name,
        start_time=start,
        end_time=start + timedelta(milliseconds=10),
        **kw,  # type: ignore[arg-type]
    )


@pytest.fixture
def support_trace() -> Trace:
    """A small refund conversation: 2 LLM calls around 2 tool calls."""
    tid = "trace-1"
    step = _span(tid, SpanKind.AGENT_STEP, "step", 0)
    spans = [
        step,
        _span(
            tid,
            SpanKind.LLM_CALL,
            "gpt-4o-mini",
            1,
            parent_id=step.id,
            usage=Usage(input_tokens=100, output_tokens=20, cost_usd=0.001),
        ),
        # Deliberately out of order in the list: derived metrics sort by start time.
        _span(tid, SpanKind.TOOL_CALL, "issue_refund", 30, parent_id=step.id),
        _span(tid, SpanKind.TOOL_CALL, "lookup_order", 20, parent_id=step.id),
        _span(
            tid,
            SpanKind.LLM_CALL,
            "gpt-4o-mini",
            40,
            parent_id=step.id,
            usage=Usage(input_tokens=150, output_tokens=30, cost_usd=0.002),
        ),
    ]
    return Trace(
        id=tid,
        name="support",
        input="refund 1234",
        output="done",
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=250),
        spans=spans,
    )


class TestUsage:
    def test_total_tokens(self) -> None:
        assert Usage(input_tokens=3, output_tokens=4).total_tokens == 7

    def test_rejects_negative_values(self) -> None:
        with pytest.raises(ValidationError):
            Usage(input_tokens=-1)
        with pytest.raises(ValidationError):
            Usage(cost_usd=-0.5)


class TestSpan:
    def test_defaults(self) -> None:
        span = Span(trace_id="t", kind=SpanKind.CUSTOM, name="x")
        assert span.id
        assert span.start_time.tzinfo is UTC
        assert span.end_time is None
        assert span.latency_ms is None
        assert span.attributes == {}

    def test_kind_accepts_string_value(self) -> None:
        span = Span.model_validate({"trace_id": "t", "kind": "tool_call", "name": "x"})
        assert span.kind is SpanKind.TOOL_CALL

    def test_latency_ms(self) -> None:
        span = _span("t", SpanKind.CUSTOM, "x", 0)
        assert span.latency_ms == pytest.approx(10.0)

    def test_naive_datetime_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="timezone"):
            Span(trace_id="t", kind=SpanKind.CUSTOM, name="x", start_time=datetime(2026, 1, 1))

    def test_datetimes_are_normalised_to_utc(self) -> None:
        cet = timezone(timedelta(hours=1))
        span = Span(
            trace_id="t",
            kind=SpanKind.CUSTOM,
            name="x",
            start_time=datetime(2026, 1, 1, 10, tzinfo=cet),
        )
        assert span.start_time == datetime(2026, 1, 1, 9, tzinfo=UTC)
        assert span.start_time.tzinfo is UTC

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="before start_time"):
            Span(
                trace_id="t",
                kind=SpanKind.CUSTOM,
                name="x",
                start_time=T0,
                end_time=T0 - timedelta(seconds=1),
            )

    def test_invariants_hold_on_assignment(self) -> None:
        span = Span(trace_id="t", kind=SpanKind.CUSTOM, name="x", start_time=T0)
        span.end_time = T0 + timedelta(seconds=1)
        with pytest.raises(ValidationError, match="before start_time"):
            span.end_time = T0 - timedelta(seconds=1)

    def test_self_parent_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="own parent"):
            Span(id="a", parent_id="a", trace_id="t", kind=SpanKind.CUSTOM, name="x")

    def test_unknown_fields_are_rejected(self) -> None:
        with pytest.raises(ValidationError, match="extra"):
            Span.model_validate({"trace_id": "t", "kind": "custom", "name": "x", "nmae": "y"})


class TestTraceValidation:
    def test_span_with_foreign_trace_id_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="expected t1"):
            Trace(id="t1", name="x", spans=[Span(trace_id="t2", kind=SpanKind.CUSTOM, name="s")])

    def test_duplicate_span_ids_are_rejected(self) -> None:
        a = Span(id="a", trace_id="t", kind=SpanKind.CUSTOM, name="s")
        with pytest.raises(ValidationError, match="duplicate"):
            Trace(id="t", name="x", spans=[a, a.model_copy()])

    def test_unknown_parent_is_rejected(self) -> None:
        orphan = Span(trace_id="t", parent_id="ghost", kind=SpanKind.CUSTOM, name="s")
        with pytest.raises(ValidationError, match="unknown parent"):
            Trace(id="t", name="x", spans=[orphan])

    def test_end_before_start_is_rejected(self) -> None:
        with pytest.raises(ValidationError, match="before start_time"):
            Trace(name="x", start_time=T0, end_time=T0 - timedelta(seconds=1))


class TestTraceDerivedMetrics:
    def test_totals(self, support_trace: Trace) -> None:
        assert support_trace.total_tokens == 300
        assert support_trace.total_cost == pytest.approx(0.003)

    def test_latency_ms(self, support_trace: Trace) -> None:
        assert support_trace.latency_ms == pytest.approx(250.0)

    def test_tool_calls_are_in_call_order(self, support_trace: Trace) -> None:
        assert support_trace.tool_calls == ["lookup_order", "issue_refund"]

    def test_n_steps_counts_agent_steps(self, support_trace: Trace) -> None:
        assert support_trace.n_steps == 1

    def test_n_steps_falls_back_to_llm_calls(self) -> None:
        spans = [_span("t", SpanKind.LLM_CALL, "m", i) for i in range(3)]
        assert Trace(id="t", name="x", spans=spans).n_steps == 3

    def test_empty_trace(self) -> None:
        trace = Trace(name="empty")
        assert trace.total_tokens == 0
        assert trace.total_cost is None
        assert trace.latency_ms is None
        assert trace.tool_calls == []
        assert trace.n_steps == 0

    def test_cost_is_none_when_unknown_but_tokens_are_counted(self) -> None:
        span = _span("t", SpanKind.LLM_CALL, "m", 0, usage=Usage(input_tokens=5))
        trace = Trace(id="t", name="x", spans=[span])
        assert trace.total_cost is None
        assert trace.total_tokens == 5

    def test_derived_metrics_are_not_serialised(self, support_trace: Trace) -> None:
        dumped = support_trace.model_dump()
        for derived in ("total_cost", "total_tokens", "latency_ms", "tool_calls", "n_steps"):
            assert derived not in dumped


class TestTraceNavigation:
    def test_roots_and_children(self, support_trace: Trace) -> None:
        (root,) = support_trace.roots()
        assert root.kind is SpanKind.AGENT_STEP
        children = support_trace.children(root.id)
        assert [c.name for c in children] == [
            "gpt-4o-mini",
            "lookup_order",
            "issue_refund",
            "gpt-4o-mini",
        ]
        assert support_trace.children(children[0].id) == []

    def test_spans_of(self, support_trace: Trace) -> None:
        assert len(support_trace.spans_of(SpanKind.LLM_CALL)) == 2
        assert support_trace.spans_of(SpanKind.RETRIEVAL) == []
