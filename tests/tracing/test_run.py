"""Tests for the ``run`` context manager."""

from __future__ import annotations

import pytest

import agentlens
from agentlens.schema import Trace


def test_run_returns_finished_trace(sink: list[Trace]) -> None:
    with agentlens.run("agent", input={"q": "hi"}, metadata={"case": "c1"}, sink=sink.append) as t:
        assert agentlens.current_trace() is t
        assert t.end_time is None
        t.output = "hello"

    assert agentlens.current_trace() is None
    assert sink == [t]
    assert t.name == "agent"
    assert t.input == {"q": "hi"}
    assert t.metadata == {"case": "c1"}
    assert t.output == "hello"
    assert t.error is None
    assert t.end_time is not None
    assert t.latency_ms is not None
    assert t.latency_ms >= 0


def test_run_without_sink_still_returns_trace() -> None:
    with agentlens.run("agent") as t:
        pass
    assert t.end_time is not None


def test_run_records_and_reraises_user_exception(sink: list[Trace]) -> None:
    with pytest.raises(KeyError), agentlens.run("agent", sink=sink.append) as t:
        raise KeyError("order")
    assert t.error == "KeyError: 'order'"
    assert t.end_time is not None
    assert sink == [t]


def test_run_converts_non_json_input() -> None:
    class Opaque:
        def __repr__(self) -> str:
            return "<Opaque>"

    with agentlens.run("agent", input={"obj": Opaque(), "t": (1, 2)}) as t:
        pass
    assert t.input == {"obj": "<Opaque>", "t": [1, 2]}


def test_nested_runs_are_independent(sink: list[Trace]) -> None:
    with agentlens.run("outer", sink=sink.append) as outer:
        with agentlens.span("before"):
            pass
        with agentlens.run("inner", sink=sink.append) as inner:
            with agentlens.span("in-inner"):
                pass
            assert agentlens.current_trace() is inner
        assert agentlens.current_trace() is outer
        with agentlens.span("after"):
            pass

    assert [s.name for s in outer.spans] == ["before", "after"]
    assert [s.name for s in inner.spans] == ["in-inner"]
    assert all(s.parent_id is None for s in inner.spans)
    assert [t.name for t in sink] == ["inner", "outer"]


def test_finished_trace_round_trips(sink: list[Trace]) -> None:
    @agentlens.trace(kind="tool_call")
    def tool(x: int) -> int:
        return x * 2

    with agentlens.run("agent", input=1, sink=sink.append) as t:
        t.output = tool(1)
    restored = Trace.model_validate_json(t.model_dump_json())
    assert restored == t
    assert restored.tool_calls == ["test_finished_trace_round_trips.<locals>.tool"]
