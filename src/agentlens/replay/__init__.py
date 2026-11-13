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

Where a replay departs from the recording is reported as a
:class:`ReplayDivergence` (call index + unified request diff): on the strict
error (``exc.divergence``), in ``cassette.divergences`` /
``cassette.first_divergence`` in every mode, on the trace (``replay_diverged``
span attribute, ``replay_divergence`` trace metadata), and by
:func:`find_divergence` for two recorded traces.
"""

from agentlens.replay._types import CassetteMode, EntryKind
from agentlens.replay.cassette import (
    CASSETTE_FORMAT,
    CASSETTE_VERSION,
    DIVERGED_ATTR,
    DIVERGENCE_ATTR,
    Cassette,
    CassetteEntry,
    CassetteHeader,
    CassetteLLM,
    tool_key,
)
from agentlens.replay.compare import find_divergence
from agentlens.replay.divergence import (
    DIFF_MAX_LINE_CHARS,
    DIFF_MAX_LINES,
    DivergenceReason,
    ReplayDivergence,
    request_diff,
)
from agentlens.replay.errors import CassetteError, CassetteFormatError, CassetteMissError

__all__ = [
    "CASSETTE_FORMAT",
    "CASSETTE_VERSION",
    "DIFF_MAX_LINES",
    "DIFF_MAX_LINE_CHARS",
    "DIVERGED_ATTR",
    "DIVERGENCE_ATTR",
    "Cassette",
    "CassetteEntry",
    "CassetteError",
    "CassetteFormatError",
    "CassetteHeader",
    "CassetteLLM",
    "CassetteMissError",
    "CassetteMode",
    "DivergenceReason",
    "EntryKind",
    "ReplayDivergence",
    "find_divergence",
    "request_diff",
    "tool_key",
]
