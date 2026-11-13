"""Errors raised by cassette recording and playback."""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

from pydantic import JsonValue

from agentlens.replay._types import CassetteMode, EntryKind
from agentlens.replay.divergence import ReplayDivergence

if TYPE_CHECKING:
    from agentlens.replay.cassette import CassetteEntry

__all__ = ["CassetteError", "CassetteFormatError", "CassetteMissError"]


class CassetteError(Exception):
    """Base class for every cassette error."""


class CassetteMissError(CassetteError):
    """A call during playback has no usable recorded counterpart.

    Raised in ``strict`` mode when the recorded call at this index has a
    different request key (or a different tool name), and in ``strict`` and
    ``lenient`` mode when there is no recorded call at this index at all. In
    ``fallthrough`` mode a miss calls the real client instead.

    The message ends with a unified diff of the recorded and actual requests;
    :attr:`divergence` holds the same information as a
    :class:`~agentlens.replay.ReplayDivergence` (for reports).

    Attributes:
        kind: ``"llm"`` or ``"tool"``.
        index: The call index (per kind, 0-based) that missed.
        mode: The cassette mode at the time of the call.
        expected: The recorded entry at ``index``, or ``None`` if the cassette
            has no call at that index.
        actual_request: The request JSON of the call being made (same shape as
            ``CassetteEntry.request``).
        actual_key: Its key (sha256 of the normalised request).
        actual_name: The tool name for tool calls, ``None`` for LLM calls.
        divergence: The :class:`~agentlens.replay.ReplayDivergence` (call
            index, reason, bounded request diff).
    """

    def __init__(
        self,
        *,
        kind: EntryKind,
        index: int,
        mode: CassetteMode,
        expected: CassetteEntry | None,
        actual_request: JsonValue,
        actual_key: str,
        actual_name: str | None = None,
        divergence: ReplayDivergence | None = None,
    ) -> None:
        self.kind: EntryKind = kind
        self.index = index
        self.mode: CassetteMode = mode
        self.expected = expected
        self.actual_request = actual_request
        self.actual_key = actual_key
        self.actual_name = actual_name
        self.divergence = divergence or ReplayDivergence.create(
            kind,
            index,
            expected_request=self.expected_request,
            actual_request=actual_request,
            expected_key=expected.key if expected is not None else None,
            actual_key=actual_key,
            expected_name=expected.name if expected is not None else None,
            actual_name=actual_name,
            mode=mode,
        )
        message = self._describe()
        if self.divergence.diff:
            message += "\n" + self.divergence.diff
        super().__init__(message)

    @property
    def expected_request(self) -> JsonValue:
        """The recorded request JSON, or ``None`` if nothing was recorded at ``index``."""
        return self.expected.request if self.expected is not None else None

    def _describe(self) -> str:
        call = f"{self.kind} call #{self.index}"
        if self.actual_name is not None:
            call += f" ({self.actual_name!r})"
        if self.expected is None:
            return f"{self.mode} cassette miss: no recorded {call}"
        if self.expected.name != self.actual_name:
            return f"{self.mode} cassette miss: {call}: recorded tool is {self.expected.name!r}"
        return (
            f"{self.mode} cassette miss: {call}: request key {self.actual_key[:12]} "
            f"!= recorded {self.expected.key[:12]}"
        )


class CassetteFormatError(CassetteError, ValueError):
    """A cassette file (or a trace given to ``Cassette.from_trace``) is invalid.

    For files, the message names the path and the 1-based line number.
    """

    def __init__(self, path: str | os.PathLike[str] | None, line: int | None, reason: str) -> None:
        self.path = os.fspath(path) if path is not None else None
        self.line = line
        where = self.path or "<cassette>"
        if line is not None:
            where += f":{line}"
        super().__init__(f"{where}: invalid cassette: {reason}")
