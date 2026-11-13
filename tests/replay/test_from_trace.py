"""Tests for :meth:`agentlens.replay.Cassette.from_trace`."""

from __future__ import annotations

import pytest

import agentlens
from agentlens.llm import LLMRequest, LLMResponse, Message, MockLLM
from agentlens.replay import Cassette, CassetteFormatError, tool_key
from agentlens.schema import Span, SpanKind, Trace, Usage


def _req(text: str) -> LLMRequest:
    return LLMRequest(messages=[Message.user(text)])


def test_from_plain_traced_run_skips_failed_nested_and_other_spans() -> None:
    llm = agentlens.wrap_llm(MockLLM(["a", "inner", "b"]), provider="acme")

    @agentlens.wrap_tool
    def research(topic: str) -> str | None:
        return llm.complete(_req(f"inner {topic}")).content  # nested: skipped

    @agentlens.wrap_tool
    def broken() -> None:
        raise RuntimeError("x")

    with agentlens.run("agent") as trace:
        with agentlens.span("plan"):
            llm.complete(_req("q0"))
        research("cats")
        with pytest.raises(RuntimeError):
            broken()
        llm.complete(_req("q1"))

    cassette = Cassette.from_trace(trace, mode="lenient", record_new=True)
    assert (cassette.mode, cassette.record_new) == ("lenient", True)
    assert [(e.kind, e.index, e.name) for e in cassette.entries] == [
        ("llm", 0, None),
        ("tool", 0, "research"),
        ("llm", 1, None),
    ]
    first, tool, _ = cassette.entries
    assert first.key == _req("q0").fingerprint()
    response = LLMResponse.model_validate(first.response)
    assert response.content == "a"
    assert response.usage == trace.spans_of(SpanKind.LLM_CALL)[0].usage
    assert tool.request == {"topic": "cats"}
    assert tool.response == "inner"
    assert tool.key == tool_key("research", {"topic": "cats"})
    assert cassette.metadata == {
        "trace_id": trace.id,
        "trace_name": "agent",
        "model": "mock",
        "provider": "acme",
    }


def _ended(span: Span) -> Span:
    span.end_time = span.start_time
    return span


def test_from_trace_with_invalid_llm_span() -> None:
    trace = Trace(name="t")
    trace.spans.append(
        _ended(Span(trace_id=trace.id, kind=SpanKind.LLM_CALL, name="m", input={"messages": []}))
    )
    with pytest.raises(CassetteFormatError, match="is not a valid call"):
        Cassette.from_trace(trace)


def test_from_trace_with_non_dict_output_and_unknown_parent() -> None:
    trace = Trace(name="t")
    trace.spans.append(
        _ended(
            Span(
                trace_id=trace.id,
                parent_id="missing",
                kind=SpanKind.LLM_CALL,
                name="m",
                input={"messages": "hi"},
                output="not a dict",
                usage=Usage(input_tokens=2),
            )
        )
    )
    trace.spans.append(Span(trace_id=trace.id, kind=SpanKind.TOOL_CALL, name="open"))
    cassette = Cassette.from_trace(trace)
    (entry,) = cassette.entries
    assert entry.response == LLMResponse(usage=Usage(input_tokens=2)).model_dump(mode="json")
    assert cassette.metadata == {"trace_id": trace.id, "trace_name": "t"}
