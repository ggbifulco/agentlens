"""Acceptance test: a recorded run replayed produces an identical trace (ADR 0004)."""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

import agentlens
from agentlens.llm import MockLLM
from agentlens.replay import Cassette, CassetteMode
from agentlens.schema import SpanKind, Trace
from tests.replay.conftest import (
    SCRIPT,
    FailingLLM,
    aagent,
    agent,
    alookup_order,
    lookup_order,
    normalise,
)

QUESTION = "Where is order 1234?"


def _record(path: Path) -> Trace:
    cassette = Cassette()
    llm = agentlens.wrap_llm(cassette.wrap_llm(MockLLM(SCRIPT)))
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with agentlens.run("support-agent", input=QUESTION) as trace:
        trace.output = agent(llm, lookup, QUESTION)
    assert cassette.save(path) == 3
    return trace


@pytest.mark.parametrize("mode", ["strict", "lenient", "fallthrough"])
def test_replayed_run_produces_identical_trace(tmp_path: Path, mode: CassetteMode) -> None:
    path = tmp_path / "run.jsonl"
    recorded = _record(path)

    calls: list[str] = []

    def broken_tool(order_id: str) -> dict[str, str]:
        calls.append(order_id)
        return {"order_id": order_id, "status": "lost"}

    cassette = Cassette.load(path, mode=mode)
    # Real client and tool would fail or answer differently: playback never reaches them.
    llm = agentlens.wrap_llm(cassette.wrap_llm(FailingLLM()))
    lookup = agentlens.wrap_tool(cassette.wrap_tool(broken_tool, name="lookup_order"))
    with agentlens.run("support-agent", input=QUESTION) as replayed:
        replayed.output = agent(llm, lookup, QUESTION)

    assert replayed.output == "Order 1234 shipped yesterday."
    assert calls == []
    assert cassette.unplayed == []
    assert normalise(replayed) == normalise(recorded)
    assert [s.kind for s in replayed.spans] == [
        SpanKind.LLM_CALL,
        SpanKind.TOOL_CALL,
        SpanKind.LLM_CALL,
    ]


def test_strict_playback_without_any_real_client(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"
    recorded = _record(path)
    cassette = Cassette.load(path)
    llm = agentlens.wrap_llm(cassette.wrap_llm())
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with agentlens.run("support-agent", input=QUESTION) as replayed:
        replayed.output = agent(llm, lookup, QUESTION)
    # model name comes from the cassette metadata, so span names/attributes match
    assert normalise(replayed) == normalise(recorded)


def test_replay_from_trace_produces_identical_trace() -> None:
    cassette = Cassette()
    llm = agentlens.wrap_llm(cassette.wrap_llm(MockLLM(SCRIPT)))
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    with agentlens.run("support-agent", input=QUESTION) as recorded:
        recorded.output = agent(llm, lookup, QUESTION)

    rebuilt = Cassette.from_trace(recorded)
    assert [e.model_dump() for e in rebuilt.entries] == [e.model_dump() for e in cassette.entries]
    llm = agentlens.wrap_llm(rebuilt.wrap_llm())
    lookup = agentlens.wrap_tool(rebuilt.wrap_tool(lookup_order))
    with agentlens.run("support-agent", input=QUESTION) as replayed:
        replayed.output = agent(llm, lookup, QUESTION)
    assert normalise(replayed) == normalise(recorded)


def test_async_replayed_run_produces_identical_trace(tmp_path: Path) -> None:
    path = tmp_path / "run.jsonl"

    async def record() -> Trace:
        cassette = Cassette()
        llm = agentlens.wrap_llm(cassette.wrap_llm(MockLLM(SCRIPT)))
        lookup = agentlens.wrap_tool(cassette.wrap_tool(alookup_order))
        with agentlens.run("support-agent", input=QUESTION) as trace:
            trace.output = await aagent(llm, lookup, QUESTION)
        cassette.save(path)
        return trace

    async def replay() -> Trace:
        cassette = Cassette.load(path)

        async def broken(order_id: str) -> dict[str, str]:
            raise AssertionError("tool was called")

        llm = agentlens.wrap_llm(cassette.wrap_llm(FailingLLM()))
        lookup = agentlens.wrap_tool(cassette.wrap_tool(broken, name="alookup_order"))
        with agentlens.run("support-agent", input=QUESTION) as trace:
            trace.output = await aagent(llm, lookup, QUESTION)
        return trace

    recorded = asyncio.run(record())
    replayed = asyncio.run(replay())
    assert normalise(replayed) == normalise(recorded)


def test_already_traced_tool_is_put_behind_the_cassette(tmp_path: Path) -> None:
    """``cassette.wrap_tool(agentlens.wrap_tool(fn))`` still records the tool span on playback."""
    path = tmp_path / "run.jsonl"
    recorded = _record(path)
    cassette = Cassette.load(path)
    traced = agentlens.wrap_tool(lookup_order)
    lookup = cassette.wrap_tool(traced)
    assert getattr(lookup, agentlens.tracing.TOOL_NAME_ATTR) == "lookup_order"
    llm = agentlens.wrap_llm(cassette.wrap_llm())
    with agentlens.run("support-agent", input=QUESTION) as replayed:
        replayed.output = agent(llm, lookup, QUESTION)
    assert normalise(replayed) == normalise(recorded)
