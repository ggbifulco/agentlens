"""Cassettes: record LLM and tool calls once, play them back deterministically (ADR 0004).

.. code-block:: python

    from agentlens.replay import Cassette

    # 1. Record: calls go to the real client/tool and are stored.
    cassette = Cassette()
    llm = agentlens.wrap_llm(cassette.wrap_llm(real_client))
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    run_agent(llm, lookup)
    cassette.save("tests/cassettes/refund.jsonl")

    # 2. Play back: no network, same responses, same trace.
    cassette = Cassette.load("tests/cassettes/refund.jsonl", mode="strict")
    llm = agentlens.wrap_llm(cassette.wrap_llm())
    lookup = agentlens.wrap_tool(cassette.wrap_tool(lookup_order))
    run_agent(llm, lookup)

Every call is keyed by ``(index, key)``: ``index`` counts the calls of one kind
(``llm`` or ``tool``) through the cassette, starting at 0, and ``key`` is the
sha256 of the normalised request (:meth:`~agentlens.llm.LLMRequest.fingerprint`
for LLM calls, :func:`tool_key` for tool calls). Playback modes:

* ``strict``: the recorded call at ``index`` must have the same key (and tool
  name), otherwise :class:`~agentlens.replay.CassetteMissError`;
* ``lenient``: the recorded call at ``index`` is returned whatever the request;
* ``fallthrough``: a hit is played back, a miss calls the real client/tool
  (and, with ``record_new=True``, stores the new call at that index).

``record`` (the default for a new cassette) always calls through and stores.

Every call whose request differs from the recorded one at its index is a
divergence (FR-R3): it is described by a :class:`~agentlens.replay.ReplayDivergence`
(with a unified diff of the requests), kept in :attr:`Cassette.divergences` and,
inside :func:`agentlens.run`, marked on the trace (span attribute
:data:`DIVERGED_ATTR`, first divergence in the trace metadata under
:data:`DIVERGENCE_ATTR`).
In ``strict`` mode it is also the ``divergence`` of the raised
:class:`~agentlens.replay.CassetteMissError`.
"""

from __future__ import annotations

import functools
import hashlib
import inspect
import json
import logging
import os
import tempfile
import threading
from collections.abc import Awaitable, Callable, Iterable
from contextvars import ContextVar
from pathlib import Path
from typing import Any, Final, Literal, ParamSpec, Self, TypeVar, cast, get_args, overload

from pydantic import Field, JsonValue, NonNegativeInt, ValidationError, model_validator

from agentlens.llm.base import LLMClient
from agentlens.llm.types import LLMRequest, LLMResponse
from agentlens.replay._types import CassetteMode, EntryKind
from agentlens.replay.divergence import ReplayDivergence
from agentlens.replay.errors import CassetteError, CassetteFormatError, CassetteMissError
from agentlens.schema import Span, SpanKind, Trace
from agentlens.schema._base import SchemaModel
from agentlens.tracing._serialize import bind_arguments, to_json_value
from agentlens.tracing.api import _signature, current_span, current_trace
from agentlens.tracing.wrappers import TOOL_NAME_ATTR
from agentlens.tracing.wrappers import wrap_tool as _trace_tool

__all__ = [
    "CASSETTE_FORMAT",
    "CASSETTE_VERSION",
    "DIVERGED_ATTR",
    "DIVERGENCE_ATTR",
    "Cassette",
    "CassetteEntry",
    "CassetteHeader",
    "CassetteLLM",
    "CassetteMode",
    "EntryKind",
    "tool_key",
]

logger = logging.getLogger("agentlens.replay")

P = ParamSpec("P")
R = TypeVar("R")

DIVERGED_ATTR: Final = "replay_diverged"
"""Span attribute set to ``True`` on a replayed call that diverged."""

DIVERGENCE_ATTR: Final = "replay_divergence"
"""Span attribute with the divergence :meth:`~ReplayDivergence.summary`; also the
trace metadata key holding the *first* divergence's
:meth:`~ReplayDivergence.to_dict` (without the full requests)."""

_SPAN_KINDS: Final[dict[EntryKind, SpanKind]] = {
    "llm": SpanKind.LLM_CALL,
    "tool": SpanKind.TOOL_CALL,
}

CASSETTE_FORMAT: Final = "agentlens-cassette"
"""Value of ``format`` in the header line of a cassette file."""

CASSETTE_VERSION: Final = 1
"""Current cassette file format version."""

_MODES: tuple[str, ...] = get_args(CassetteMode)

# Ids of the cassettes whose real tool is executing in this context: calls made
# *inside* such a tool are neither recorded nor played back, because playing the
# tool back skips its body (and the calls it makes).
_bypass: ContextVar[frozenset[int]] = ContextVar("agentlens_cassette_bypass", default=frozenset())


def _canonical(value: JsonValue) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def tool_key(name: str, arguments: JsonValue) -> str:
    """Cassette key of a tool call: sha256 of the canonical JSON of name and arguments.

    ``arguments`` is the JSON of the call's arguments by parameter name (what a
    ``tool_call`` span records as input).
    """
    payload = _canonical({"name": name, "arguments": arguments})
    return hashlib.sha256(payload.encode()).hexdigest()


def _llm_request_json(request: LLMRequest) -> JsonValue:
    # Same shape as the ``llm_call`` span input: defaults omitted, still round-trips.
    return request.model_dump(mode="json", exclude_defaults=True)


class CassetteHeader(SchemaModel):
    """First line of a cassette file: format marker, version and free metadata."""

    format: Literal["agentlens-cassette"] = CASSETTE_FORMAT
    version: Literal[1] = CASSETTE_VERSION
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


class CassetteEntry(SchemaModel):
    """One recorded call (one line of a cassette file).

    For ``llm`` entries ``request`` is the :class:`~agentlens.llm.LLMRequest`
    JSON (default fields omitted) and ``response`` the full
    :class:`~agentlens.llm.LLMResponse` JSON (usage and raw included); both are
    validated. For ``tool`` entries ``name`` is the tool name, ``request`` the
    arguments by parameter name and ``response`` the JSON of the return value.
    ``key`` is the sha256 hex digest compared in ``strict`` mode.
    """

    index: NonNegativeInt
    kind: EntryKind
    key: str = Field(pattern=r"^[0-9a-f]{64}$")
    name: str | None = None
    request: JsonValue = None
    response: JsonValue = None

    @model_validator(mode="after")
    def _check_kind(self) -> Self:
        if self.kind == "llm":
            LLMRequest.model_validate(self.request)
            LLMResponse.model_validate(self.response)
        elif self.name is None:
            raise ValueError("tool entries need a name")
        return self


class Cassette:
    """Recorded LLM and tool calls, with record and playback modes.

    Wrap clients with :meth:`wrap_llm` and tools with :meth:`wrap_tool`; all
    wrappers of one cassette share its call counters. Wrap *inside* the
    tracing wrappers (``agentlens.wrap_llm(cassette.wrap_llm(client))``) so
    played-back calls are traced exactly like live ones. Errors raised by the
    real client or tool are not recorded (the call does not consume an index).
    Calls made from inside a real tool execution are passed straight through.

    Calls are indexed in the order they reach the cassette; concurrent calls
    through one cassette are safe but their order (and so their index) is only
    as deterministic as the agent's scheduling.

    Args:
        entries: Initial entries (e.g. from a file); ``(kind, index)`` must be unique.
        mode: ``"record"`` (default), ``"strict"``, ``"lenient"`` or ``"fallthrough"``.
            It can be changed later through the :attr:`mode` attribute.
        record_new: In ``fallthrough`` mode, store calls answered by the real
            client/tool (replacing a mismatching entry at that index).
        metadata: Free JSON metadata saved in the file header.

    Raises:
        ValueError: for an unknown mode.
        CassetteFormatError: if two entries share a ``(kind, index)``.
    """

    def __init__(
        self,
        entries: Iterable[CassetteEntry] = (),
        *,
        mode: CassetteMode = "record",
        record_new: bool = False,
        metadata: dict[str, JsonValue] | None = None,
    ) -> None:
        self.mode = mode
        self.record_new = record_new
        """In ``fallthrough`` mode, store calls answered by the real client/tool."""
        self.metadata: dict[str, JsonValue] = dict(metadata or {})
        """Free JSON metadata (saved in the header); ``model``/``provider`` are
        used as defaults by :meth:`wrap_llm` during playback."""
        self._lock = threading.Lock()
        self._entries: list[CassetteEntry] = []
        self._positions: dict[tuple[EntryKind, int], int] = {}
        self._cursor: dict[EntryKind, int] = {"llm": 0, "tool": 0}
        self._divergences: list[ReplayDivergence] = []
        for entry in entries:
            if (entry.kind, entry.index) in self._positions:
                raise CassetteFormatError(None, None, f"duplicate {entry.kind} call #{entry.index}")
            self._put(entry)

    @property
    def mode(self) -> CassetteMode:
        """The current mode; assign to switch (e.g. from ``"record"`` to ``"strict"``)."""
        return self._mode

    @mode.setter
    def mode(self, value: CassetteMode) -> None:
        if value not in _MODES:
            raise ValueError(f"unknown cassette mode {value!r} (expected one of {_MODES})")
        self._mode: CassetteMode = value

    def __repr__(self) -> str:
        return f"Cassette(mode={self.mode!r}, entries={len(self)})"

    def __len__(self) -> int:
        return len(self._entries)

    @property
    def entries(self) -> list[CassetteEntry]:
        """All entries, in the order they were recorded (or loaded)."""
        with self._lock:
            return list(self._entries)

    def entries_of(self, kind: EntryKind) -> list[CassetteEntry]:
        """The entries of one kind, sorted by index."""
        return sorted((e for e in self.entries if e.kind == kind), key=lambda e: e.index)

    @property
    def unplayed(self) -> list[CassetteEntry]:
        """Entries not reached yet by playback (useful to assert a replay was complete)."""
        with self._lock:
            return [e for e in self._entries if e.index >= self._cursor[e.kind]]

    @property
    def divergences(self) -> list[ReplayDivergence]:
        """Calls of this playback whose request differed from the recording, in call order.

        Filled in every playback mode: ``strict`` (each one also raised a
        :class:`CassetteMissError`), ``lenient`` (the recorded response was
        played anyway) and ``fallthrough`` (the real client/tool answered).
        Cleared by :meth:`rewind`.
        """
        with self._lock:
            return list(self._divergences)

    @property
    def first_divergence(self) -> ReplayDivergence | None:
        """The first entry of :attr:`divergences`, or ``None`` if the replay matched so far."""
        with self._lock:
            return self._divergences[0] if self._divergences else None

    def rewind(self) -> None:
        """Reset the call counters and divergences, to record or play again from the start."""
        with self._lock:
            self._cursor = {"llm": 0, "tool": 0}
            self._divergences = []

    # -- wrappers ---------------------------------------------------------------

    def wrap_llm(
        self,
        client: LLMClient | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> CassetteLLM:
        """Return an :class:`~agentlens.llm.LLMClient` answering through this cassette.

        ``client`` is the real client: required to record and for
        ``fallthrough`` misses, optional for ``strict``/``lenient`` playback.
        The returned client's ``model``/``provider`` attributes (read by
        :func:`agentlens.wrap_llm` for span names and attributes) are the
        arguments, else the client's, else the cassette ``metadata`` values;
        the client's values are stored in ``metadata`` when it has none yet.
        """
        llm = CassetteLLM(self, client, model=model, provider=provider)
        for attribute in ("model", "provider"):
            value = getattr(llm, attribute)
            if client is not None and value is not None:
                self.metadata.setdefault(attribute, value)
        return llm

    @overload
    def wrap_tool(self, fn: Callable[P, R], /, *, name: str | None = ...) -> Callable[P, R]: ...

    @overload
    def wrap_tool(
        self, /, *, name: str | None = ...
    ) -> Callable[[Callable[P, R]], Callable[P, R]]: ...

    def wrap_tool(
        self,
        fn: Callable[P, R] | None = None,
        /,
        *,
        name: str | None = None,
    ) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
        """Record/play back calls of a tool (sync or async), like :meth:`wrap_llm` for LLMs.

        Usable as a wrapper or decorator (``@cassette.wrap_tool``). Calls are
        keyed by :func:`tool_key` of the tool name (``name``, else
        ``fn.__name__``) and the JSON of the arguments by parameter name.
        Playback returns the recorded return value **as JSON** (dicts, lists,
        strings, numbers), so tools returning other objects get their JSON form.

        The wrapper's ``__name__`` is the tool name, so wrapping it with
        :func:`agentlens.wrap_tool` gives the span that name too.
        A function already wrapped by :func:`agentlens.wrap_tool` is unwrapped,
        put behind the cassette and traced again, so its ``tool_call`` span is
        still recorded when the tool is played back.
        """

        def decorate(func: Callable[P, R]) -> Callable[P, R]:
            traced_name = getattr(func, TOOL_NAME_ATTR, None)
            if isinstance(traced_name, str):
                tool_name = name or traced_name
                inner = self._tool_wrapper(func.__wrapped__, tool_name)  # type: ignore[attr-defined]
                return _trace_tool(inner, name=tool_name)
            tool_name = name or getattr(func, "__name__", None) or type(func).__name__
            return self._tool_wrapper(func, tool_name)

        if fn is not None:
            return decorate(fn)
        return decorate

    def _tool_wrapper(self, func: Callable[P, R], tool_name: str) -> Callable[P, R]:
        signature = _signature(func)

        def prepare(args: tuple[Any, ...], kwargs: dict[str, Any]) -> tuple[JsonValue, str]:
            arguments = to_json_value(bind_arguments(signature, args, kwargs))
            return arguments, tool_key(tool_name, arguments)

        if inspect.iscoroutinefunction(func):
            async_func = cast(Callable[..., Awaitable[Any]], func)

            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if self._bypassed():
                    return await async_func(*args, **kwargs)
                arguments, key = prepare(args, kwargs)
                index, hit = self._begin("tool", key, arguments, tool_name)
                if hit is not None:
                    return hit.response
                token = _bypass.set(_bypass.get() | {id(self)})
                try:
                    result = await async_func(*args, **kwargs)
                except BaseException:
                    self._rollback("tool", index)
                    raise
                finally:
                    _bypass.reset(token)
                self._finish("tool", index, key, tool_name, arguments, to_json_value(result))
                return result

            async_wrapper.__name__ = tool_name  # so agentlens.wrap_tool() picks the same name
            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(func)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if self._bypassed():
                return func(*args, **kwargs)
            arguments, key = prepare(args, kwargs)
            index, hit = self._begin("tool", key, arguments, tool_name)
            if hit is not None:
                return cast(R, hit.response)
            token = _bypass.set(_bypass.get() | {id(self)})
            try:
                result = func(*args, **kwargs)
            except BaseException:
                self._rollback("tool", index)
                raise
            finally:
                _bypass.reset(token)
            self._finish("tool", index, key, tool_name, arguments, to_json_value(result))
            return result

        wrapper.__name__ = tool_name
        return wrapper

    # -- call protocol (used by the wrappers) -------------------------------------

    def _bypassed(self) -> bool:
        return id(self) in _bypass.get()

    def _put(self, entry: CassetteEntry) -> None:
        slot = (entry.kind, entry.index)
        position = self._positions.get(slot)
        if position is None:
            self._positions[slot] = len(self._entries)
            self._entries.append(entry)
        else:
            self._entries[position] = entry

    def _begin(
        self, kind: EntryKind, key: str, request: JsonValue, name: str | None
    ) -> tuple[int, CassetteEntry | None]:
        """Reserve the next index of ``kind``; return it with the entry to play back.

        ``None`` means "call the real client/tool". Raises :class:`CassetteMissError`
        (without consuming the index) on a strict/lenient miss. Every mismatch
        is recorded as a divergence (see :meth:`_diverge`).
        """
        with self._lock:
            index = self._cursor[kind]
            mode = self._mode
            hit: CassetteEntry | None = None
            recorded: CassetteEntry | None = None
            matches = True
            if mode != "record":
                position = self._positions.get((kind, index))
                recorded = self._entries[position] if position is not None else None
                matches = recorded is not None and recorded.key == key and recorded.name == name
                if matches or (mode == "lenient" and recorded is not None):
                    hit = recorded
            raises = not matches and hit is None and mode != "fallthrough"
            if not raises:
                self._cursor[kind] = index + 1
        if matches:
            return index, hit
        divergence = self._diverge(kind, index, mode, recorded, request, key, name)
        if raises:
            raise CassetteMissError(
                kind=kind,
                index=index,
                mode=mode,
                expected=recorded,
                actual_request=request,
                actual_key=key,
                actual_name=name,
                divergence=divergence,
            )
        return index, hit

    def _diverge(
        self,
        kind: EntryKind,
        index: int,
        mode: CassetteMode,
        recorded: CassetteEntry | None,
        request: JsonValue,
        key: str,
        name: str | None,
    ) -> ReplayDivergence:
        """Describe a mismatching call, store it and mark it on the current trace."""
        span = current_span()
        if span is not None and span.kind is not _SPAN_KINDS[kind]:
            span = None  # the cassette is not wrapped inside the tracing wrappers
        divergence = ReplayDivergence.create(
            kind,
            index,
            expected_request=recorded.request if recorded is not None else None,
            actual_request=request,
            expected_key=recorded.key if recorded is not None else None,
            actual_key=key,
            expected_name=recorded.name if recorded is not None else None,
            actual_name=name,
            mode=mode,
            span_id=span.id if span is not None else None,
        )
        with self._lock:
            self._divergences.append(divergence)
        _mark_trace(span, divergence)
        return divergence

    def _rollback(self, kind: EntryKind, index: int) -> None:
        """Give back ``index`` after a failed real call (if no later call took one)."""
        with self._lock:
            if self._cursor[kind] == index + 1:
                self._cursor[kind] = index

    def _finish(
        self,
        kind: EntryKind,
        index: int,
        key: str,
        name: str | None,
        request: JsonValue,
        response: JsonValue,
    ) -> None:
        """Store a call answered by the real client/tool, if the mode records."""
        with self._lock:
            if self._mode == "record" or (self._mode == "fallthrough" and self.record_new):
                entry = CassetteEntry(
                    index=index, kind=kind, key=key, name=name, request=request, response=response
                )
                self._put(entry)

    # -- persistence ----------------------------------------------------------------

    def save(self, path: str | os.PathLike[str]) -> int:
        """Write the cassette as JSONL; return the number of entries written.

        Line 1 is the :class:`CassetteHeader`, then one :class:`CassetteEntry`
        per line in recording order. The file is written to a temporary
        sibling and renamed atomically; parent directories are created.
        """
        header = CassetteHeader(metadata=self.metadata)
        entries = self.entries
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
        try:
            with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
                fh.write(header.model_dump_json() + "\n")
                for entry in entries:
                    fh.write(entry.model_dump_json() + "\n")
            os.replace(tmp_name, target)
        except BaseException:
            Path(tmp_name).unlink(missing_ok=True)
            raise
        return len(entries)

    @classmethod
    def load(
        cls,
        path: str | os.PathLike[str],
        *,
        mode: CassetteMode = "strict",
        record_new: bool = False,
    ) -> Cassette:
        """Read a cassette written by :meth:`save` (default mode: ``strict``).

        The header line is optional (a hand-written file may be just entries)
        but, if present, must be the first non-blank line. Blank lines are
        skipped.

        Raises:
            CassetteFormatError: on an invalid line, an unsupported header or a
                duplicate ``(kind, index)``, naming the file and line.
        """
        metadata: dict[str, JsonValue] = {}
        entries: list[CassetteEntry] = []
        seen: set[tuple[EntryKind, int]] = set()
        first = True
        with Path(path).open(encoding="utf-8") as fh:
            for number, line in enumerate(fh, start=1):
                if not line.strip():
                    continue
                try:
                    if first and '"format"' in line:
                        metadata = CassetteHeader.model_validate_json(line).metadata
                        first = False
                        continue
                    entry = CassetteEntry.model_validate_json(line)
                except ValidationError as exc:
                    reason = exc.errors(include_url=False)[0]
                    location = ".".join(str(part) for part in reason["loc"])
                    raise CassetteFormatError(
                        path, number, f"{location}: {reason['msg']}" if location else reason["msg"]
                    ) from exc
                first = False
                if (entry.kind, entry.index) in seen:
                    raise CassetteFormatError(
                        path, number, f"duplicate {entry.kind} call #{entry.index}"
                    )
                seen.add((entry.kind, entry.index))
                entries.append(entry)
        return cls(entries, mode=mode, record_new=record_new, metadata=metadata)

    @classmethod
    def from_trace(
        cls,
        trace: Trace,
        *,
        mode: CassetteMode = "strict",
        record_new: bool = False,
    ) -> Cassette:
        """Build a cassette from the ``llm_call`` and ``tool_call`` spans of a recorded trace.

        Works for traces recorded with :func:`agentlens.wrap_llm` /
        :func:`agentlens.wrap_tool` (with or without a cassette): spans are
        taken in start order, failed or unfinished spans and spans nested
        inside a ``tool_call`` are skipped (mirroring what recording stores).
        LLM responses get the span's ``usage`` back; the provider ``raw``
        payload is only available if the trace was recorded with
        ``include_raw=True``. ``metadata`` records the trace id and name and,
        when known, the client's default ``model`` and ``provider``.

        Raises:
            CassetteFormatError: if an ``llm_call`` span's input/output is not a
                valid request/response.
        """
        metadata: dict[str, JsonValue] = {"trace_id": trace.id, "trace_name": trace.name}
        entries: list[CassetteEntry] = []
        counts: dict[EntryKind, int] = {"llm": 0, "tool": 0}
        for span in replayable_spans(trace):
            if span.kind is SpanKind.LLM_CALL:
                entry = _llm_entry_from_span(span, counts["llm"])
                if not metadata.keys() & {"model", "provider"}:
                    for attribute in ("model", "provider"):
                        value = span.attributes.get(attribute)
                        if isinstance(value, str):
                            metadata[attribute] = value
            else:
                entry = CassetteEntry(
                    index=counts["tool"],
                    kind="tool",
                    key=tool_key(span.name, span.input),
                    name=span.name,
                    request=span.input,
                    response=span.output,
                )
            counts[entry.kind] += 1
            entries.append(entry)
        return cls(entries, mode=mode, record_new=record_new, metadata=metadata)


def replayable_spans(trace: Trace) -> list[Span]:
    """The spans of ``trace`` that :meth:`Cassette.from_trace` turns into entries, in order.

    Finished, successful ``llm_call``/``tool_call`` spans not nested inside a
    ``tool_call`` (mirroring what recording through a cassette stores).
    """
    by_id = {span.id: span for span in trace.spans}

    def inside_tool(span: Span) -> bool:
        parent_id = span.parent_id
        while parent_id is not None and parent_id in by_id:
            parent = by_id[parent_id]
            if parent.kind is SpanKind.TOOL_CALL:
                return True
            parent_id = parent.parent_id
        return False

    return [
        span
        for span in trace.spans
        if span.kind in (SpanKind.LLM_CALL, SpanKind.TOOL_CALL)
        and span.error is None
        and span.end_time is not None
        and not inside_tool(span)
    ]


def _mark_trace(span: Span | None, divergence: ReplayDivergence) -> None:
    """Annotate the replayed span and (first divergence only) the trace; never raises."""
    try:
        if span is not None:
            span.attributes = {
                **span.attributes,
                DIVERGED_ATTR: True,
                DIVERGENCE_ATTR: divergence.summary(),
            }
        trace = current_trace()
        if trace is not None and DIVERGENCE_ATTR not in trace.metadata:
            trace.metadata = {
                **trace.metadata,
                DIVERGENCE_ATTR: divergence.to_dict(include_requests=False),
            }
    except Exception:  # tracing must never break the agent
        logger.exception("could not mark the replay divergence on the trace")


def _llm_entry_from_span(span: Span, index: int) -> CassetteEntry:
    try:
        request = LLMRequest.model_validate(span.input)
        output = span.output if isinstance(span.output, dict) else {}
        response = LLMResponse.model_validate({**output, "usage": span.usage})
    except ValidationError as exc:
        raise CassetteFormatError(
            None, None, f"llm_call span {span.id} ({span.name!r}) is not a valid call: {exc}"
        ) from exc
    return CassetteEntry(
        index=index,
        kind="llm",
        key=request.fingerprint(),
        request=_llm_request_json(request),
        response=response.model_dump(mode="json"),
    )


class CassetteLLM:
    """An :class:`~agentlens.llm.LLMClient` answering from a :class:`Cassette`.

    Created by :meth:`Cassette.wrap_llm`. Attributes not defined here are
    looked up on the real client (when there is one).
    """

    def __init__(
        self,
        cassette: Cassette,
        client: LLMClient | None = None,
        *,
        model: str | None = None,
        provider: str | None = None,
    ) -> None:
        self.cassette = cassette
        """The cassette answering the calls."""
        self.client = client
        """The real client, or ``None`` for pure playback."""
        self.model: str | None = model or _str_attr(client, "model", cassette, "model")
        """Default model (for span names); see :meth:`Cassette.wrap_llm`."""
        self.provider: str | None = provider or _str_attr(client, "provider", cassette, "provider")
        """Provider name (span attribute); see :meth:`Cassette.wrap_llm`."""

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes missing on the wrapper itself.
        client = self.__dict__.get("client")
        if client is None:
            raise AttributeError(name)
        return getattr(client, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.client!r}, mode={self.cassette.mode!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Play back the recorded response, or call (and maybe record) the real client."""
        if self.cassette._bypassed():
            return self._real_client(None).complete(request)
        index, key, hit = self._begin(request)
        if hit is not None:
            return LLMResponse.model_validate(hit.response)
        try:
            response = self._real_client(index).complete(request)
        except BaseException:
            self.cassette._rollback("llm", index)
            raise
        self._finish(index, key, request, response)
        return response

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Async variant of :meth:`complete` (calls the real client's ``acomplete``)."""
        if self.cassette._bypassed():
            return await self._real_client(None).acomplete(request)
        index, key, hit = self._begin(request)
        if hit is not None:
            return LLMResponse.model_validate(hit.response)
        try:
            response = await self._real_client(index).acomplete(request)
        except BaseException:
            self.cassette._rollback("llm", index)
            raise
        self._finish(index, key, request, response)
        return response

    def _begin(self, request: LLMRequest) -> tuple[int, str, CassetteEntry | None]:
        key = request.fingerprint()
        index, hit = self.cassette._begin("llm", key, _llm_request_json(request), None)
        return index, key, hit

    def _real_client(self, index: int | None) -> LLMClient:
        if self.client is not None:
            return self.client
        mode = self.cassette.mode
        if index is not None:
            self.cassette._rollback("llm", index)
        if mode == "fallthrough":
            raise CassetteError(f"fallthrough miss on llm call #{index} but no real client given")
        raise CassetteError(f"cassette mode {mode!r} needs a real client (wrap_llm(client))")

    def _finish(self, index: int, key: str, request: LLMRequest, response: LLMResponse) -> None:
        self.cassette._finish(
            "llm", index, key, None, _llm_request_json(request), response.model_dump(mode="json")
        )


def _str_attr(client: object, attribute: str, cassette: Cassette, meta_key: str) -> str | None:
    value = getattr(client, attribute, None) if client is not None else None
    if isinstance(value, str):
        return value
    meta = cassette.metadata.get(meta_key)
    return meta if isinstance(meta, str) else None
