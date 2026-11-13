"""Tests for :func:`agentlens.wrap_tool` (``tool_call`` spans)."""

from __future__ import annotations

import asyncio
import functools
import inspect
import logging
from collections.abc import Iterator

import pytest

import agentlens
from agentlens.schema import SpanKind
from agentlens.tracing import TOOL_NAME_ATTR, Recorder


@agentlens.wrap_tool
def lookup_order(order_id: str, *, verbose: bool = False) -> dict[str, str]:
    """Look an order up."""
    return {"id": order_id, "status": "shipped"}


@agentlens.wrap_tool(name="refund")
async def issue_refund(order_id: str, amount: float) -> bool:
    await asyncio.sleep(0)
    return amount > 0


@agentlens.wrap_tool
def failing_tool(order_id: str) -> None:
    raise KeyError(order_id)


class Inventory:
    @agentlens.wrap_tool
    def stock(self, sku: str) -> int:
        return len(sku)


def _search(query: str, limit: int = 3) -> list[str]:
    return [f"{query}-{i}" for i in range(limit)]


def test_decorator_records_tool_call_span() -> None:
    with agentlens.run("agent") as t:
        assert lookup_order("1234") == {"id": "1234", "status": "shipped"}
    (s,) = t.spans
    assert s.kind is SpanKind.TOOL_CALL
    assert s.name == "lookup_order"
    assert s.input == {"order_id": "1234"}
    assert s.output == {"id": "1234", "status": "shipped"}
    assert s.end_time is not None
    assert t.tool_calls == ["lookup_order"]


def test_wrapper_keeps_metadata() -> None:
    assert lookup_order.__name__ == "lookup_order"
    assert lookup_order.__doc__ == "Look an order up."
    assert getattr(lookup_order, TOOL_NAME_ATTR) == "lookup_order"
    assert inspect.signature(lookup_order).parameters.keys() == {"order_id", "verbose"}
    assert inspect.iscoroutinefunction(issue_refund)


def test_plain_wrapper_with_name() -> None:
    search = agentlens.wrap_tool(_search, name="search_docs")
    with agentlens.run("agent") as t:
        assert search("refund", limit=2) == ["refund-0", "refund-1"]
    (s,) = t.spans
    assert s.name == "search_docs"
    assert s.input == {"query": "refund", "limit": 2}
    assert getattr(search, TOOL_NAME_ATTR) == "search_docs"
    assert search.__wrapped__ is _search  # type: ignore[attr-defined]


def test_async_tool_with_name() -> None:
    async def main() -> agentlens.Trace:
        with agentlens.run("agent") as t:
            assert await issue_refund("1234", 9.5) is True
        return t

    (s,) = asyncio.run(main()).spans
    assert s.kind is SpanKind.TOOL_CALL
    assert s.name == "refund"
    assert s.input == {"order_id": "1234", "amount": 9.5}
    assert s.output is True


def test_method_tool_drops_self() -> None:
    with agentlens.run("agent") as t:
        assert Inventory().stock("ab") == 2
    (s,) = t.spans
    assert s.name == "stock"
    assert s.input == {"sku": "ab"}


def test_tool_spans_nest_under_llm_agent_step() -> None:
    @agentlens.trace(kind="agent_step")
    def step() -> None:
        lookup_order("1")
        lookup_order("2")

    with agentlens.run("agent") as t:
        step()
    outer, *tools = t.spans
    assert [s.parent_id for s in tools] == [outer.id, outer.id]
    assert t.tool_calls == ["lookup_order", "lookup_order"]


def test_errors_are_recorded_and_reraised() -> None:
    with agentlens.run("agent") as t, pytest.raises(KeyError):
        failing_tool("9")
    (s,) = t.spans
    assert s.error == "KeyError: '9'"
    assert s.output is None


def test_async_errors_are_recorded_and_reraised() -> None:
    @agentlens.wrap_tool
    async def broken() -> None:
        raise ValueError("bad")

    async def main() -> agentlens.Trace:
        with agentlens.run("agent") as t, pytest.raises(ValueError, match="bad"):
            await broken()
        return t

    (s,) = asyncio.run(main()).spans
    assert s.error == "ValueError: bad"


def test_passthrough_outside_run() -> None:
    assert lookup_order("1") == {"id": "1", "status": "shipped"}
    assert asyncio.run(issue_refund("1", 0)) is False
    with pytest.raises(KeyError):
        failing_tool("1")
    assert agentlens.current_trace() is None


def test_wrapping_twice_never_nests_spans() -> None:
    assert agentlens.wrap_tool(lookup_order) is lookup_order
    assert agentlens.wrap_tool(lookup_order, name="lookup_order") is lookup_order
    renamed = agentlens.wrap_tool(lookup_order, name="find_order")
    assert renamed is not lookup_order
    with agentlens.run("agent") as t:
        renamed("1")
    assert [s.name for s in t.spans] == ["find_order"]


def test_callables_without_name_or_signature() -> None:
    class Adder:
        def __call__(self, a: int, b: int) -> int:
            return a + b

    partial = agentlens.wrap_tool(functools.partial(_search, limit=1))
    adder = agentlens.wrap_tool(Adder())
    builtin = agentlens.wrap_tool(dict)  # no inspectable signature on some versions
    with agentlens.run("agent") as t:
        assert partial("x") == ["x-0"]
        assert adder(1, 2) == 3
        assert builtin(a=1) == {"a": 1}
    names = [s.name for s in t.spans]
    assert names == ["partial", "Adder", "dict"]
    assert t.spans[0].input == {"query": "x"}  # arguments of the call itself
    assert t.spans[1].input == {"a": 1, "b": 2}


def test_generator_tools_are_rejected() -> None:
    def gen() -> Iterator[int]:
        yield 1

    with pytest.raises(TypeError, match="generator"):
        agentlens.wrap_tool(gen)


def test_recorder_failure_never_breaks_the_tool(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("recorder bug")

    monkeypatch.setattr(Recorder, "end_span", boom)
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent"):
        assert lookup_order("1")["id"] == "1"
        assert agentlens.current_span() is None
    assert "could not end span" in caplog.text
