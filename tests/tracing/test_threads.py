"""Context propagation into threads."""

from __future__ import annotations

import asyncio
import contextvars
import threading
from concurrent.futures import ThreadPoolExecutor

import agentlens
from agentlens.schema import SpanKind, Trace


@agentlens.trace(kind="tool_call")
def work(i: int) -> int:
    return i * i


def test_copied_context_attaches_thread_spans_to_parent() -> None:
    with agentlens.run("agent") as t, agentlens.span("fan-out") as parent:
        assert parent is not None
        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(contextvars.copy_context().run, work, i) for i in range(20)]
            results = [f.result() for f in futures]

    assert results == [i * i for i in range(20)]
    tools = t.spans_of(SpanKind.TOOL_CALL)
    assert len(tools) == 20
    assert all(s.parent_id == parent.id for s in tools)
    assert sorted(s.output for s in tools) == sorted(results)  # type: ignore[type-var]
    assert len({s.id for s in t.spans}) == 21


def test_asyncio_to_thread_propagates_context() -> None:
    async def main() -> Trace:
        with agentlens.run("agent") as t:
            await asyncio.to_thread(work, 3)
        return t

    t = asyncio.run(main())
    assert [s.output for s in t.spans] == [9]


def test_plain_thread_is_not_recorded() -> None:
    """Threads started without copying the context see no active run."""
    seen: list[Trace | None] = []

    def target() -> None:
        seen.append(agentlens.current_trace())
        work(2)

    with agentlens.run("agent") as t:
        thread = threading.Thread(target=target)
        thread.start()
        thread.join()

    assert seen == [None]
    assert t.spans == []


def test_concurrent_runs_in_threads_do_not_mix() -> None:
    def one(i: int) -> Trace:
        with agentlens.run(f"run-{i}") as t:
            for j in range(10):
                work(j)
        return t

    with ThreadPoolExecutor(max_workers=8) as pool:
        traces = list(pool.map(one, range(8)))
    for t in traces:
        assert len(t.spans) == 10
        assert all(s.trace_id == t.id for s in t.spans)
