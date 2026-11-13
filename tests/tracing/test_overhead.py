"""Benchmark: recording overhead must stay below 5 ms per span (FR-T5)."""

from __future__ import annotations

import time

import agentlens

N = 2_000
BUDGET_MS = 5.0


@agentlens.trace(kind="tool_call")
def tool(order_id: str, payload: dict[str, object]) -> dict[str, object]:
    return {"id": order_id, **payload}


def _undecorated(order_id: str, payload: dict[str, object]) -> dict[str, object]:
    return {"id": order_id, **payload}


def test_overhead_per_span_below_budget() -> None:
    payload: dict[str, object] = {"items": list(range(20)), "note": "x" * 200}

    start = time.perf_counter()
    for _ in range(N):
        _undecorated("1234", payload)
    baseline = time.perf_counter() - start

    with agentlens.run("bench") as t:
        start = time.perf_counter()
        for _ in range(N):
            tool("1234", payload)
        traced = time.perf_counter() - start

    assert len(t.spans) == N
    per_span_ms = (traced - baseline) / N * 1000
    print(f"tracing overhead: {per_span_ms * 1000:.1f} us/span")
    assert per_span_ms < BUDGET_MS
