"""OpenTelemetry GenAI exporter (FR-T6), checked against the SDK's in-memory exporter."""

from __future__ import annotations

import json
import logging
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from opentelemetry.sdk.trace import ReadableSpan, TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter
from opentelemetry.trace import SpanKind as OTelKind
from opentelemetry.trace import StatusCode

import agentlens
from agentlens.integrations.otel import (
    INSTRUMENTATION_NAME,
    OTelExporter,
    export_trace,
    fan_out,
    span_attributes,
    span_name,
)
from agentlens.llm import LLMRequest, Message, get_client
from agentlens.schema import Span, SpanKind, Trace, Usage

T0 = datetime(2026, 11, 14, 12, 0, tzinfo=UTC)


@pytest.fixture
def otel() -> Iterator[tuple[TracerProvider, InMemorySpanExporter]]:
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    yield provider, exporter
    provider.shutdown()


def by_name(spans: tuple[ReadableSpan, ...]) -> dict[str, ReadableSpan]:
    return {s.name: s for s in spans}


def sample_trace(*, error: str | None = None) -> Trace:
    """invoke_agent support > [step plan > (chat gpt-x, execute_tool lookup_order), retrieval]."""
    trace = Trace(
        name="support",
        input="refund 1004",
        output="escalated",
        error=error,
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=500),
        metadata={"case_id": "refund", "repetition": 1, "tags": ["x"]},
    )
    step = Span(
        trace_id=trace.id,
        kind=SpanKind.AGENT_STEP,
        name="plan",
        start_time=T0 + timedelta(milliseconds=1),
        end_time=T0 + timedelta(milliseconds=400),
    )
    llm = Span(
        trace_id=trace.id,
        parent_id=step.id,
        kind=SpanKind.LLM_CALL,
        name="gpt-x",
        start_time=T0 + timedelta(milliseconds=2),
        end_time=T0 + timedelta(milliseconds=200),
        input={"messages": [{"role": "user", "content": "refund 1004"}], "temperature": 0},
        output={"content": None, "tool_calls": [{"id": "c1", "name": "lookup_order"}]},
        usage=Usage(input_tokens=12, output_tokens=5, cost_usd=0.002),
        attributes={
            "provider": "openai",
            "model": "gpt-x",
            "temperature": 0,
            "max_tokens": 256,
            "response_model": "gpt-x-2026",
        },
    )
    tool = Span(
        trace_id=trace.id,
        parent_id=step.id,
        kind=SpanKind.TOOL_CALL,
        name="lookup_order",
        start_time=T0 + timedelta(milliseconds=210),
        end_time=None,  # unfinished: ends with the trace
        input={"order_id": "1004"},
        output={"total": 899},
        error="KeyError: '1004'",
    )
    retrieval = Span(
        trace_id=trace.id,
        kind=SpanKind.RETRIEVAL,
        name="faq-search",
        start_time=T0 + timedelta(milliseconds=410),
        end_time=T0 + timedelta(milliseconds=420),
        input="refund policy",
        output=["doc-1"],
    )
    trace.spans = [retrieval, tool, llm, step]  # any order: exported parents-first
    return trace


def test_exports_the_span_tree_with_genai_conventions(
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, memory = otel
    trace = sample_trace()
    assert export_trace(trace, tracer_provider=provider) == 5

    spans = by_name(memory.get_finished_spans())
    assert set(spans) == {
        "invoke_agent support",
        "invoke_agent plan",
        "chat gpt-x",
        "execute_tool lookup_order",
        "faq-search",
    }
    root = spans["invoke_agent support"]
    step, chat = spans["invoke_agent plan"], spans["chat gpt-x"]
    tool, retrieval = spans["execute_tool lookup_order"], spans["faq-search"]

    # One OTel trace, same tree as the AgentLens trace.
    assert {s.context.trace_id for s in spans.values()} == {root.context.trace_id}
    assert root.parent is None
    assert step.parent.span_id == root.context.span_id
    assert chat.parent.span_id == step.context.span_id
    assert tool.parent.span_id == step.context.span_id
    assert retrieval.parent.span_id == root.context.span_id
    assert root.instrumentation_scope.name == INSTRUMENTATION_NAME

    # Recorded timings, to the microsecond.
    assert root.start_time == int(T0.timestamp()) * 10**9
    assert root.end_time - root.start_time == 500 * 10**6
    assert chat.end_time - chat.start_time == 198 * 10**6
    assert tool.end_time == root.end_time  # unfinished span ends with the trace

    assert dict(root.attributes) == {
        "gen_ai.operation.name": "invoke_agent",
        "gen_ai.agent.name": "support",
        "agentlens.trace_id": trace.id,
        "agentlens.metadata.case_id": "refund",
        "agentlens.metadata.repetition": 1,
        "agentlens.usage.cost_usd": 0.002,
        "agentlens.usage.total_tokens": 17,
    }
    assert chat.kind is OTelKind.CLIENT
    assert {k: v for k, v in chat.attributes.items() if k.startswith("gen_ai.")} == {
        "gen_ai.operation.name": "chat",
        "gen_ai.provider.name": "openai",
        "gen_ai.request.model": "gpt-x",
        "gen_ai.request.temperature": 0.0,
        "gen_ai.request.max_tokens": 256,
        "gen_ai.response.model": "gpt-x-2026",
        "gen_ai.usage.input_tokens": 12,
        "gen_ai.usage.output_tokens": 5,
    }
    assert chat.attributes["agentlens.usage.cost_usd"] == 0.002
    assert step.attributes["gen_ai.agent.name"] == "plan"
    assert tool.kind is OTelKind.INTERNAL
    assert tool.attributes["gen_ai.operation.name"] == "execute_tool"
    assert tool.attributes["gen_ai.tool.name"] == "lookup_order"
    assert tool.attributes["error.type"] == "KeyError"
    assert tool.status.status_code is StatusCode.ERROR
    assert tool.status.description == "KeyError: '1004'"
    assert chat.status.status_code is StatusCode.UNSET
    assert "gen_ai.operation.name" not in retrieval.attributes
    assert retrieval.attributes["agentlens.span.kind"] == "retrieval"
    assert retrieval.attributes["agentlens.span_id"] == trace.spans[0].id

    # No content unless asked for.
    for span in spans.values():
        assert not any(
            k in span.attributes
            for k in (
                "gen_ai.input.messages",
                "gen_ai.output.messages",
                "gen_ai.tool.call.arguments",
                "agentlens.input",
            )
        )


def test_capture_content(otel: tuple[TracerProvider, InMemorySpanExporter]) -> None:
    provider, memory = otel
    export_trace(sample_trace(), tracer_provider=provider, capture_content=True)
    spans = by_name(memory.get_finished_spans())
    chat = spans["chat gpt-x"].attributes
    assert json.loads(str(chat["gen_ai.input.messages"])) == [
        {"role": "user", "content": "refund 1004"}
    ]
    assert json.loads(str(chat["gen_ai.output.messages"])) == [
        {"role": "assistant", "tool_calls": [{"id": "c1", "name": "lookup_order"}]}
    ]
    tool = spans["execute_tool lookup_order"].attributes
    assert json.loads(str(tool["gen_ai.tool.call.arguments"])) == {"order_id": "1004"}
    assert json.loads(str(tool["gen_ai.tool.call.result"])) == {"total": 899}
    root = spans["invoke_agent support"].attributes
    assert json.loads(str(root["agentlens.input"])) == "refund 1004"
    assert json.loads(str(root["agentlens.output"])) == "escalated"
    retrieval = spans["faq-search"].attributes
    assert json.loads(str(retrieval["agentlens.output"])) == ["doc-1"]


def test_failed_trace_marks_the_root(otel: tuple[TracerProvider, InMemorySpanExporter]) -> None:
    provider, memory = otel
    export_trace(sample_trace(error="RunTimeoutError: took too long"), tracer_provider=provider)
    root = by_name(memory.get_finished_spans())["invoke_agent support"]
    assert root.status.status_code is StatusCode.ERROR
    assert root.attributes["error.type"] == "RunTimeoutError"


def test_unfinished_trace_ends_at_its_last_span(
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, memory = otel
    trace = Trace(name="t", start_time=T0)
    trace.spans = [
        Span(
            trace_id=trace.id,
            kind=SpanKind.CUSTOM,
            name="work",
            start_time=T0,
            end_time=T0 + timedelta(seconds=1),
        )
    ]
    export_trace(trace, tracer_provider=provider)
    root = by_name(memory.get_finished_spans())["invoke_agent t"]
    assert root.end_time - root.start_time == 10**9
    empty = Trace(name="empty", start_time=T0)
    assert export_trace(empty, tracer_provider=provider) == 1
    assert by_name(memory.get_finished_spans())["invoke_agent empty"].end_time == root.start_time


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        ("ValueError: boom", "ValueError"),
        ("agentlens.Error: x", "agentlens.Error"),
        ("something went wrong", "_OTHER"),
        ("bad thing: happened", "_OTHER"),
        (": no type", "_OTHER"),
    ],
)
def test_error_type(error: str, expected: str) -> None:
    span = Span(trace_id="t", kind=SpanKind.CUSTOM, name="x", error=error)
    assert span_attributes(span)["error.type"] == expected


def test_llm_span_without_known_attributes() -> None:
    span = Span(
        trace_id="t",
        kind=SpanKind.LLM_CALL,
        name="MockLLM",
        input="raw prompt",
        output="raw answer",
        attributes={"temperature": True, "max_tokens": "many", "model": ""},
    )
    assert span_name(span) == "chat MockLLM"
    attrs = span_attributes(span, capture_content=True)
    assert attrs["gen_ai.operation.name"] == "chat"
    for key in (
        "gen_ai.provider.name",
        "gen_ai.request.model",
        "gen_ai.request.temperature",
        "gen_ai.request.max_tokens",
        "gen_ai.usage.input_tokens",
    ):
        assert key not in attrs
    assert attrs["gen_ai.input.messages"] == '"raw prompt"'
    assert attrs["gen_ai.output.messages"] == '"raw answer"'
    bare = Span(trace_id="t", kind=SpanKind.TOOL_CALL, name="noop")
    assert "gen_ai.tool.call.arguments" not in span_attributes(bare, capture_content=True)
    usage = Span(trace_id="t", kind=SpanKind.CUSTOM, name="c", usage=Usage(input_tokens=3))
    attrs = span_attributes(usage)
    assert attrs["gen_ai.usage.input_tokens"] == 3
    assert "agentlens.usage.cost_usd" not in attrs


def test_explicit_tracer_and_parent_context(
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    from opentelemetry import trace as otel_trace

    provider, memory = otel
    tracer = provider.get_tracer("custom")
    with tracer.start_as_current_span("ci-job") as job:
        export_trace(
            sample_trace(),
            tracer=tracer,
            context=otel_trace.set_span_in_context(job),
        )
    spans = by_name(memory.get_finished_spans())
    assert spans["invoke_agent support"].parent.span_id == spans["ci-job"].context.span_id
    assert spans["chat gpt-x"].instrumentation_scope.name == "custom"


def test_exporter_as_run_sink_with_wrap_llm(
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, memory = otel
    exporter = OTelExporter(tracer_provider=provider)
    stored: list[Trace] = []
    with agentlens.run("echo", sink=fan_out(stored.append, exporter)):
        llm = agentlens.wrap_llm(get_client("mock"))
        llm.complete(LLMRequest(messages=[Message.user("hi")], temperature=0.2))
        agentlens.wrap_tool(lambda q: q.upper(), name="shout")("hi")
    assert len(stored) == 1
    names = {s.name for s in memory.get_finished_spans()}
    assert "invoke_agent echo" in names
    assert "execute_tool shout" in names
    chat = next(s for s in memory.get_finished_spans() if s.name.startswith("chat "))
    assert chat.attributes["gen_ai.request.temperature"] == pytest.approx(0.2)
    assert chat.attributes["agentlens.trace_id"] == stored[0].id
    assert "OTelExporter" in repr(exporter)


def test_exporter_never_raises(caplog: pytest.LogCaptureFixture) -> None:
    class Broken:
        def get_tracer(self, *args: Any, **kwargs: Any) -> Any:
            raise RuntimeError("collector down")

    exporter = OTelExporter(tracer_provider=Broken())  # type: ignore[arg-type]
    with caplog.at_level(logging.ERROR, logger="agentlens.otel"):
        exporter(sample_trace())
    assert "exporting trace" in caplog.text


def test_fan_out_isolates_failures(caplog: pytest.LogCaptureFixture) -> None:
    seen: list[str] = []

    def boom(trace: Trace) -> None:
        raise ValueError("disk full")

    sink = fan_out(boom, lambda t: seen.append(t.name))
    with caplog.at_level(logging.ERROR, logger="agentlens.otel"):
        sink(Trace(name="t"))
    assert seen == ["t"]
    assert "trace sink" in caplog.text


def test_capture_content_with_partial_values(
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, memory = otel
    trace = Trace(name="partial", start_time=T0, end_time=T0, output="done")
    trace.spans = [
        Span(trace_id=trace.id, kind=SpanKind.LLM_CALL, name="m", start_time=T0),
        Span(trace_id=trace.id, kind=SpanKind.TOOL_CALL, name="t", output=1, start_time=T0),
        Span(trace_id=trace.id, kind=SpanKind.CUSTOM, name="c", input=2, start_time=T0),
    ]
    export_trace(trace, tracer_provider=provider, capture_content=True)
    spans = by_name(memory.get_finished_spans())
    root = spans["invoke_agent partial"].attributes
    assert "agentlens.input" not in root
    assert root["agentlens.output"] == '"done"'
    chat = spans["chat m"].attributes
    assert "gen_ai.input.messages" not in chat
    assert "gen_ai.output.messages" not in chat
    tool = spans["execute_tool t"].attributes
    assert "gen_ai.tool.call.arguments" not in tool
    assert tool["gen_ai.tool.call.result"] == "1"
    custom = spans["c"].attributes
    assert custom["agentlens.input"] == "2"
    assert "agentlens.output" not in custom


def test_capture_content_trace_without_output(
    otel: tuple[TracerProvider, InMemorySpanExporter],
) -> None:
    provider, memory = otel
    export_trace(
        Trace(name="q", input="x", start_time=T0), tracer_provider=provider, capture_content=True
    )
    root = by_name(memory.get_finished_spans())["invoke_agent q"].attributes
    assert root["agentlens.input"] == '"x"'
    assert "agentlens.output" not in root
