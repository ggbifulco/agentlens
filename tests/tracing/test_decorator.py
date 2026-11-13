"""Tests for the ``@trace`` decorator and ``span`` context manager."""

from __future__ import annotations

import asyncio

import pytest

import agentlens
from agentlens.schema import SpanKind, Trace


@agentlens.trace(kind=SpanKind.TOOL_CALL, name="lookup_order")
def lookup_order(order_id: str, *, verbose: bool = False) -> dict[str, str]:
    return {"id": order_id, "status": "shipped"}


@agentlens.trace(kind="agent_step")
def step(question: str) -> str:
    order = lookup_order("1234")
    return f"Order {order['id']} is {order['status']}"


@agentlens.trace
def plain(x: int) -> int:
    return x + 1


class Agent:
    @agentlens.trace(kind="agent_step")
    def answer(self, question: str) -> str:
        return question.upper()


def test_untraced_outside_run() -> None:
    assert agentlens.current_trace() is None
    assert step("where?") == "Order 1234 is shipped"


def test_decorator_records_nested_spans() -> None:
    with agentlens.run("agent") as t:
        t.output = step("where is my order?")

    outer, inner = t.spans
    assert outer.kind is SpanKind.AGENT_STEP
    assert outer.name == "step"
    assert outer.parent_id is None
    assert outer.input == {"question": "where is my order?"}
    assert outer.output == "Order 1234 is shipped"
    assert inner.kind is SpanKind.TOOL_CALL
    assert inner.name == "lookup_order"
    assert inner.parent_id == outer.id
    assert inner.input == {"order_id": "1234"}
    assert inner.output == {"id": "1234", "status": "shipped"}
    assert all(s.end_time is not None for s in t.spans)
    assert t.tool_calls == ["lookup_order"]
    assert t.n_steps == 1


def test_bare_decorator_defaults() -> None:
    with agentlens.run("agent") as t:
        assert plain(1) == 2
    (only,) = t.spans
    assert only.kind is SpanKind.CUSTOM
    assert only.name == "plain"
    assert only.input == {"x": 1}
    assert only.output == 2


def test_decorator_preserves_metadata() -> None:
    assert lookup_order.__name__ == "lookup_order"
    assert plain.__wrapped__(1) == 2  # type: ignore[attr-defined]


def test_method_drops_self() -> None:
    with agentlens.run("agent") as t:
        assert Agent().answer("hi") == "HI"
    assert t.spans[0].input == {"question": "hi"}
    assert t.spans[0].name == "Agent.answer"


def test_mismatched_call_records_raw_args_and_raises() -> None:
    with agentlens.run("agent") as t, pytest.raises(TypeError):
        plain(1, 2)  # type: ignore[call-arg]
    (only,) = t.spans
    assert only.input == {"args": [1, 2], "kwargs": {}}
    assert only.error is not None
    assert only.error.startswith("TypeError")


def test_exception_recorded_on_span_and_reraised() -> None:
    @agentlens.trace(kind="tool_call")
    def failing() -> None:
        raise RuntimeError("tool down")

    with pytest.raises(RuntimeError, match="tool down"), agentlens.run("agent") as t:
        failing()
    (only,) = t.spans
    assert only.error == "RuntimeError: tool down"
    assert only.output is None
    assert only.end_time is not None
    assert t.error == "RuntimeError: tool down"
    assert agentlens.current_span() is None


def test_current_span_inside_decorated_function() -> None:
    seen: list[str | None] = []

    @agentlens.trace
    def inspect_span() -> None:
        current = agentlens.current_span()
        seen.append(current.name if current else None)

    with agentlens.run("agent"):
        inspect_span()
        assert agentlens.current_span() is None
    assert seen == ["test_current_span_inside_decorated_function.<locals>.inspect_span"]


def test_invalid_kind_rejected() -> None:
    with pytest.raises(ValueError, match="not_a_kind"):
        agentlens.trace(kind="not_a_kind")


def test_generator_functions_rejected() -> None:
    def gen() -> object:
        yield 1

    async def agen() -> object:
        yield 1

    with pytest.raises(TypeError, match="generator"):
        agentlens.trace(gen)
    with pytest.raises(TypeError, match="generator"):
        agentlens.trace(agen)


# -- async -------------------------------------------------------------------


@agentlens.trace(kind="tool_call")
async def fetch(item: str, delay: float) -> str:
    await asyncio.sleep(delay)
    return item.upper()


@agentlens.trace(kind="agent_step")
async def fan_out() -> list[str]:
    # The slower task starts first, so start and end order differ.
    return list(await asyncio.gather(fetch("a", 0.02), fetch("b", 0.0)))


def test_async_function_untraced_outside_run() -> None:
    assert asyncio.run(fan_out()) == ["A", "B"]


def test_async_gather_nests_under_parent() -> None:
    async def main() -> Trace:
        with agentlens.run("agent") as t:
            t.output = await fan_out()
        return t

    t = asyncio.run(main())
    parent = next(s for s in t.spans if s.kind is SpanKind.AGENT_STEP)
    children = t.children(parent.id)
    assert sorted(c.name for c in children) == ["fetch", "fetch"]
    assert {c.output for c in children} == {"A", "B"}
    assert all(c.parent_id == parent.id for c in children)
    assert parent.output == ["A", "B"]
    assert len(t.spans) == 3


def test_concurrent_runs_do_not_mix() -> None:
    async def one(i: int) -> Trace:
        with agentlens.run(f"run-{i}") as t:
            await fetch(str(i), 0.01 * (3 - i))
        return t

    async def main() -> list[Trace]:
        return list(await asyncio.gather(*(one(i) for i in range(3))))

    traces = asyncio.run(main())
    for i, t in enumerate(traces):
        (only,) = t.spans
        assert only.input == {"item": str(i), "delay": 0.01 * (3 - i)}
        assert only.trace_id == t.id


def test_async_exception_recorded_and_reraised() -> None:
    @agentlens.trace
    async def boom() -> None:
        await asyncio.sleep(0)
        raise ValueError("bad")

    async def main() -> Trace:
        with agentlens.run("agent") as t:
            await boom()
        return t  # pragma: no cover

    with pytest.raises(ValueError, match="bad"):
        asyncio.run(main())


def test_async_cancellation_recorded() -> None:
    recorded: list[Trace] = []

    @agentlens.trace
    async def slow() -> None:
        await asyncio.sleep(10)

    async def main() -> None:
        with agentlens.run("agent", sink=recorded.append):
            task = asyncio.ensure_future(slow())
            await asyncio.sleep(0.01)
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task

    asyncio.run(main())
    (only,) = recorded[0].spans
    assert only.error == "CancelledError"


# -- span() ------------------------------------------------------------------


def test_span_context_manager() -> None:
    with agentlens.run("agent") as t:
        with agentlens.span(
            "search", kind="retrieval", input="refund policy", attributes={"k": 3}
        ) as s:
            assert s is not None
            plain(1)
            s.output = ["doc-1"]
        with pytest.raises(LookupError), agentlens.span("fails"):
            raise LookupError("missing")

    search, child, fails = t.spans
    assert search.kind is SpanKind.RETRIEVAL
    assert search.input == "refund policy"
    assert search.attributes == {"k": 3}
    assert search.output == ["doc-1"]
    assert child.parent_id == search.id
    assert fails.error == "LookupError: missing"


def test_span_outside_run_yields_none() -> None:
    with agentlens.span("noop") as s:
        assert s is None
