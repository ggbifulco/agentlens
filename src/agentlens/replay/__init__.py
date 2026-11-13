"""Deterministic replay of agent runs with cassettes (``docs/design.md`` §4, ADR 0004).

Record the LLM and tool calls of a run once, then play them back without the
network, so a change to the agent's code can be tested in isolation::

    from agentlens.replay import Cassette

    cassette = Cassette()                                   # mode="record"
    llm = agentlens.wrap_llm(cassette.wrap_llm(real_client))
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    ...
    cassette.save("cassettes/refund.jsonl")

    cassette = Cassette.load("cassettes/refund.jsonl")      # mode="strict"
    llm = agentlens.wrap_llm(cassette.wrap_llm())           # no real client needed

Modes: ``strict`` (index and request key must match, else
:class:`CassetteMissError`), ``lenient`` (index only), ``fallthrough`` (a miss
calls the real client, and with ``record_new=True`` records it).
"""

from agentlens.replay.cassette import (
    CASSETTE_FORMAT,
    CASSETTE_VERSION,
    Cassette,
    CassetteEntry,
    CassetteHeader,
    CassetteLLM,
    CassetteMode,
    EntryKind,
    tool_key,
)
from agentlens.replay.errors import CassetteError, CassetteFormatError, CassetteMissError

__all__ = [
    "CASSETTE_FORMAT",
    "CASSETTE_VERSION",
    "Cassette",
    "CassetteEntry",
    "CassetteError",
    "CassetteFormatError",
    "CassetteHeader",
    "CassetteLLM",
    "CassetteMissError",
    "CassetteMode",
    "EntryKind",
    "tool_key",
]
