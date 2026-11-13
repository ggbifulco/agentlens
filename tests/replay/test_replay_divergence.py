"""Divergence detection during cassette playback and between traces (FR-R3)."""

from __future__ import annotations

from pathlib import Path

import pytest

import agentlens
from agentlens.llm import LLMRequest, Message, MockLLM
from agentlens.replay import (
    DIVERGED_ATTR,
    DIVERGENCE_ATTR,
    Cassette,
    CassetteFormatError,
    CassetteMissError,
    ReplayDivergence,
    find_divergence,
)
from agentlens.replay import cassette as cassette_module
from agentlens.schema import Span, SpanKind, Trace
from tests.replay.conftest import SCRIPT, FailingLLM, agent, lookup_order

QUESTION = "Where is order 1234?"
CHANGED = "Where is my order 1234?"


def _record(path: Path) -> Trace:
    cassette = Cassette()
    llm = agentlens.wrap_llm(cassette.wrap_llm(MockLLM(SCRIPT)))
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with agentlens.run("support-agent", input=QUESTION) as trace:
        trace.output = agent(llm, lookup, QUESTION)
    cassette.save(path)
    return trace


def _replay(cassette: Cassette, question: str, llm_client: object = None) -> Trace:
    llm = agentlens.wrap_llm(cassette.wrap_llm(llm_client))  # type: ignore[arg-type]
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with agentlens.run("support-agent", input=question) as trace:
        trace.output = agent(llm, lookup, question)
    return trace


def _req(text: str) -> LLMRequest:
    return LLMRequest(messages=[Message.user(text)])


# -- playback ----------------------------------------------------------------------


def test_faithful_replay_has_no_divergence(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorded = _record(path)
    cassette = Cassette.load(path)
    replayed = _replay(cassette, QUESTION, FailingLLM())
    assert cassette.divergences == []
    assert cassette.first_divergence is None
    assert DIVERGENCE_ATTR not in replayed.metadata
    assert all(DIVERGED_ATTR not in span.attributes for span in replayed.spans)
    assert find_divergence(recorded, replayed) is None
    assert find_divergence(recorded, replayed, kind="tool") is None


def test_strict_error_carries_the_divergence_and_marks_the_trace(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _record(path)
    cassette = Cassette.load(path, mode="strict")
    llm = agentlens.wrap_llm(cassette.wrap_llm())
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with pytest.raises(CassetteMissError) as info, agentlens.run("replay") as trace:
        agent(llm, lookup, CHANGED)
    err = info.value
    divergence = err.divergence
    assert (divergence.kind, divergence.index, divergence.reason) == ("llm", 0, "request_changed")
    assert divergence.mode == "strict"
    assert divergence.expected_request == err.expected_request
    assert divergence.actual_request == err.actual_request
    assert divergence.expected_key == cassette.entries[0].key
    assert divergence.actual_key == err.actual_key
    # the message ends with the unified diff
    message = str(err)
    assert message.startswith("strict cassette miss: llm call #0: request key")
    assert message.endswith(divergence.diff)
    assert f'-      "content": "{QUESTION}",' in message.splitlines()
    assert f'+      "content": "{CHANGED}",' in message.splitlines()
    assert cassette.divergences == [divergence]

    span = trace.spans[0]
    assert span.kind is SpanKind.LLM_CALL
    assert span.error is not None
    assert "CassetteMissError" in span.error
    assert span.attributes[DIVERGED_ATTR] is True
    assert span.attributes[DIVERGENCE_ATTR] == divergence.summary()
    assert divergence.span_id == span.id
    assert trace.metadata[DIVERGENCE_ATTR] == divergence.to_dict(include_requests=False)


def test_lenient_replay_records_divergences_without_failing(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _record(path)
    cassette = Cassette.load(path, mode="lenient")
    replayed = _replay(cassette, CHANGED, FailingLLM())
    assert replayed.output == "Order 1234 shipped yesterday."  # recorded answers played
    divergences = cassette.divergences
    assert [(d.kind, d.index, d.mode) for d in divergences] == [
        ("llm", 0, "lenient"),
        ("llm", 1, "lenient"),
    ]
    assert cassette.first_divergence == divergences[0]
    llm_spans = replayed.spans_of(SpanKind.LLM_CALL)
    assert [d.span_id for d in divergences] == [span.id for span in llm_spans]
    assert all(span.attributes[DIVERGED_ATTR] is True for span in llm_spans)
    tool_span = replayed.spans_of(SpanKind.TOOL_CALL)[0]
    assert DIVERGED_ATTR not in tool_span.attributes  # same tool call: no divergence
    # only the first divergence goes in the trace metadata
    assert replayed.metadata[DIVERGENCE_ATTR] == divergences[0].to_dict(include_requests=False)


def test_fallthrough_records_divergences_and_calls_the_real_client(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    _record(path)
    cassette = Cassette.load(path, mode="fallthrough")
    replayed = _replay(cassette, CHANGED, MockLLM(["I cannot find it."]))
    assert replayed.output == "I cannot find it."
    first = cassette.first_divergence
    assert first is not None
    assert (first.kind, first.index, first.mode, first.reason) == (
        "llm",
        0,
        "fallthrough",
        "request_changed",
    )
    assert len(cassette.divergences) == 1
    assert replayed.metadata[DIVERGENCE_ATTR]["summary"] == first.summary()  # type: ignore[index]


def test_extra_call_beyond_the_recording(tmp_path: Path) -> None:
    cassette = Cassette()
    cassette.wrap_llm(MockLLM(["a"])).complete(_req("q0"))
    cassette.mode = "fallthrough"
    cassette.rewind()
    llm = cassette.wrap_llm(MockLLM(["live"]))
    llm.complete(_req("q0"))
    assert llm.complete(_req("q1")).content == "live"
    (divergence,) = cassette.divergences
    assert (divergence.index, divergence.reason, divergence.expected_key) == (1, "extra_call", None)
    assert divergence.span_id is None  # not traced
    assert divergence.diff.splitlines()[3] == "+{"


def test_strict_miss_beyond_the_end_has_an_extra_call_divergence() -> None:
    cassette = Cassette(mode="strict")
    with pytest.raises(CassetteMissError, match="no recorded llm call #0") as info:
        cassette.wrap_llm().complete(_req("q0"))
    assert info.value.divergence.reason == "extra_call"
    assert "+++ replayed llm call #0" in str(info.value)


def test_record_mode_never_diverges() -> None:
    cassette = Cassette()
    llm = cassette.wrap_llm(MockLLM(["a", "b"]))
    llm.complete(_req("q0"))
    cassette.rewind()
    llm.complete(_req("other"))  # re-recording replaces the entry
    assert cassette.divergences == []


def test_rewind_clears_divergences() -> None:
    cassette = Cassette()
    cassette.wrap_llm(MockLLM(["a"])).complete(_req("q0"))
    cassette.mode = "lenient"
    cassette.rewind()
    cassette.wrap_llm().complete(_req("changed"))
    assert len(cassette.divergences) == 1
    cassette.rewind()
    assert cassette.divergences == []
    assert cassette.first_divergence is None


def test_tool_divergence_marks_the_tool_span() -> None:
    cassette = Cassette()
    cassette.wrap_tool(lookup_order)("1")
    cassette.mode = "lenient"
    cassette.rewind()
    tool = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with agentlens.run("tools") as trace:
        assert tool("2") == {"order_id": "1", "status": "shipped"}
    (divergence,) = cassette.divergences
    assert (divergence.kind, divergence.reason, divergence.actual_name) == (
        "tool",
        "request_changed",
        "lookup_order",
    )
    assert '+    "order_id": "2"' in divergence.diff.splitlines()
    span = trace.spans[0]
    assert span.id == divergence.span_id
    assert span.attributes[DIVERGED_ATTR] is True


def test_cassette_outside_the_tracing_wrapper_only_marks_the_trace() -> None:
    cassette = Cassette()
    cassette.wrap_llm(MockLLM(["a"])).complete(_req("q0"))
    cassette.mode = "lenient"
    cassette.rewind()
    # Wrong order (cassette outside tracing): the current span is not the llm_call.
    llm = cassette.wrap_llm()
    with agentlens.run("outer") as trace, agentlens.span("step"):
        llm.complete(_req("changed"))
    divergence = cassette.first_divergence
    assert divergence is not None
    assert divergence.span_id is None
    assert DIVERGED_ATTR not in trace.spans[0].attributes
    assert trace.metadata[DIVERGENCE_ATTR] == divergence.to_dict(include_requests=False)


def test_marking_the_trace_never_breaks_the_replay(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom() -> Trace:
        raise RuntimeError("tracing bug")

    monkeypatch.setattr(cassette_module, "current_trace", boom)
    cassette = Cassette()
    cassette.wrap_llm(MockLLM(["a"])).complete(_req("q0"))
    cassette.mode = "lenient"
    cassette.rewind()
    assert cassette.wrap_llm().complete(_req("changed")).content == "a"
    assert len(cassette.divergences) == 1
    assert "could not mark the replay divergence" in caplog.text


# -- find_divergence ------------------------------------------------------------------


def _traced_run(question: str, script: list[object], *, extra_tool: bool = False) -> Trace:
    llm = agentlens.wrap_llm(MockLLM(script))  # type: ignore[arg-type]
    lookup = agentlens.wrap_tool(lookup_order)
    with agentlens.run("support-agent", input=question) as trace:
        trace.output = agent(llm, lookup, question)
        if extra_tool:
            lookup("9999")
    return trace


def test_find_divergence_reports_the_first_changed_llm_request(tmp_path: Path) -> None:
    recorded = _record(tmp_path / "run.jsonl")
    replayed = _traced_run(CHANGED, list(SCRIPT))
    divergence = find_divergence(recorded, replayed)
    assert isinstance(divergence, ReplayDivergence)
    assert (divergence.kind, divergence.index, divergence.reason, divergence.mode) == (
        "llm",
        0,
        "request_changed",
        None,
    )
    assert divergence.span_id == replayed.spans_of(SpanKind.LLM_CALL)[0].id
    assert f'+      "content": "{CHANGED}",' in divergence.diff.splitlines()
    assert divergence.summary() == "llm call #0: request differs from the recording"
    # tools were called with the same arguments
    assert find_divergence(recorded, replayed, kind="tool") is None


def test_find_divergence_extra_and_missing_calls(tmp_path: Path) -> None:
    recorded = _record(tmp_path / "run.jsonl")
    longer = _traced_run(QUESTION, list(SCRIPT), extra_tool=True)
    extra = find_divergence(recorded, longer, kind="tool")
    assert extra is not None
    assert (extra.index, extra.reason) == (1, "extra_call")
    assert extra.span_id == longer.spans_of(SpanKind.TOOL_CALL)[1].id
    missing = find_divergence(longer, recorded, kind="tool")
    assert missing is not None
    assert (missing.index, missing.reason, missing.span_id) == (1, "missing_call", None)
    assert missing.actual_request is None


def test_find_divergence_tool_changed() -> None:
    def make(tool_name: str) -> Trace:
        tool = agentlens.wrap_tool(lookup_order, name=tool_name)
        with agentlens.run("t") as trace:
            tool("1")
        return trace

    divergence = find_divergence(make("lookup_order"), make("find_order"), kind="tool")
    assert divergence is not None
    assert divergence.reason == "tool_changed"
    assert (divergence.expected_name, divergence.actual_name) == ("lookup_order", "find_order")


def test_find_divergence_rejects_invalid_llm_spans() -> None:
    trace = Trace(name="bad")
    span = Span(trace_id=trace.id, kind=SpanKind.LLM_CALL, name="x", input="not a request")
    span.end_time = span.start_time
    trace.spans.append(span)
    with pytest.raises(CassetteFormatError):
        find_divergence(trace, trace)
