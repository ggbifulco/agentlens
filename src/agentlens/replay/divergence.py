"""Where a replay departs from its recording (FR-R3, ``docs/design.md`` §4).

A :class:`ReplayDivergence` describes the first (or any) call of a replay whose
request differs from the recorded call at the same index: which call, why, and a
unified diff of the two requests. Cassettes create them during playback
(:attr:`Cassette.divergences <agentlens.replay.Cassette.divergences>`, the
``divergence`` of a :class:`~agentlens.replay.CassetteMissError`), and
:func:`~agentlens.replay.find_divergence` compares two recorded traces.

The diff is computed over pretty-printed canonical JSON (sorted keys, 2-space
indent), so it is stable whatever the dict order of the requests, and it is
bounded (:data:`DIFF_MAX_LINES` lines of at most :data:`DIFF_MAX_LINE_CHARS`
characters) so a divergence can be put in an error message, a trace or a report.
"""

from __future__ import annotations

import difflib
import json
from typing import Final, Literal

from pydantic import JsonValue

from agentlens.replay._types import CassetteMode, EntryKind
from agentlens.schema._base import SchemaModel

__all__ = [
    "DIFF_MAX_LINES",
    "DIFF_MAX_LINE_CHARS",
    "DivergenceReason",
    "ReplayDivergence",
    "request_diff",
]

DIFF_MAX_LINES: Final = 200
"""Default maximum number of lines kept in a divergence diff."""

DIFF_MAX_LINE_CHARS: Final = 500
"""Default maximum length of one diff line (longer lines are cut with ``...``)."""

DivergenceReason = Literal["request_changed", "tool_changed", "extra_call", "missing_call"]
"""Why a call diverged:

* ``request_changed``: same call (and tool name), different request;
* ``tool_changed``: the replay called another tool than the recording at this index;
* ``extra_call``: the replay made a call the recording does not have;
* ``missing_call``: the recording has a call the replay did not make (only
  detectable once the replay is over, e.g. by :func:`~agentlens.replay.find_divergence`).
"""


def _pretty(document: JsonValue) -> list[str]:
    text = json.dumps(document, sort_keys=True, indent=2, ensure_ascii=False)
    return text.splitlines(keepends=True)


def _clip(line: str, max_chars: int) -> str:
    body = line.rstrip("\n")
    if len(body) > max_chars:
        body = body[: max(max_chars - 3, 0)] + "..."
    return body + "\n"


def request_diff(
    expected: JsonValue,
    actual: JsonValue,
    *,
    expected_label: str = "recorded",
    actual_label: str = "replayed",
    present: tuple[bool, bool] = (True, True),
    max_lines: int = DIFF_MAX_LINES,
    max_line_chars: int = DIFF_MAX_LINE_CHARS,
) -> tuple[str, bool]:
    """Unified diff of two request JSON documents; return ``(diff, truncated)``.

    Both documents are pretty-printed as canonical JSON (sorted keys, 2-space
    indent) before diffing, so key order never shows up as a change. A side
    marked absent in ``present`` (no call at all) diffs as an empty document.
    The result is empty when the documents are equal; otherwise it keeps at
    most ``max_lines`` lines of at most ``max_line_chars`` characters, with a
    final ``... (N more diff lines)`` marker when lines were dropped.
    ``truncated`` tells whether anything (lines or line ends) was cut.
    """
    before = _pretty(expected) if present[0] else []
    after = _pretty(actual) if present[1] else []
    lines = list(difflib.unified_diff(before, after, fromfile=expected_label, tofile=actual_label))
    # Lines without a trailing newline (last JSON line) get one, so joins stay readable.
    lines = [line if line.endswith("\n") else line + "\n" for line in lines]
    truncated = False
    if len(lines) > max_lines:
        dropped = len(lines) - max_lines
        lines = [*lines[:max_lines], f"... ({dropped} more diff lines)\n"]
        truncated = True
    clipped = [_clip(line, max_line_chars) for line in lines]
    truncated = truncated or clipped != lines
    return "".join(clipped).rstrip("\n"), truncated


class ReplayDivergence(SchemaModel):
    """One call where a replay differs from the recording.

    ``*_request`` are the request JSON documents as stored in a cassette entry
    (an :class:`~agentlens.llm.LLMRequest` without defaults for LLM calls, the
    arguments by parameter name for tool calls); ``None`` with a ``None`` key
    when that side has no call at ``index``. ``diff`` is a bounded unified diff
    of the two (for tool calls, of ``{"name", "arguments"}``), see
    :func:`request_diff`. Build one with :meth:`create`.

    Attributes:
        kind: ``"llm"`` or ``"tool"``.
        index: Call index (per kind, 0-based), as in the cassette.
        reason: See :data:`DivergenceReason`.
        mode: Cassette mode during playback; ``None`` for trace comparisons.
        expected_name / actual_name: Tool names (``None`` for LLM calls).
        expected_key / actual_key: Request keys (sha256); ``None`` for a side
            without a call.
        expected_request / actual_request: The request JSON of each side.
        diff: The unified diff (``recorded`` → ``replayed``).
        diff_truncated: Whether ``diff`` was cut to keep it bounded.
        span_id: The replayed ``llm_call``/``tool_call`` span, when traced.
    """

    kind: EntryKind
    index: int
    reason: DivergenceReason
    mode: CassetteMode | None = None
    expected_name: str | None = None
    actual_name: str | None = None
    expected_key: str | None = None
    actual_key: str | None = None
    expected_request: JsonValue = None
    actual_request: JsonValue = None
    diff: str = ""
    diff_truncated: bool = False
    span_id: str | None = None

    @classmethod
    def create(
        cls,
        kind: EntryKind,
        index: int,
        *,
        expected_request: JsonValue = None,
        actual_request: JsonValue = None,
        expected_key: str | None = None,
        actual_key: str | None = None,
        expected_name: str | None = None,
        actual_name: str | None = None,
        mode: CassetteMode | None = None,
        span_id: str | None = None,
        max_lines: int = DIFF_MAX_LINES,
        max_line_chars: int = DIFF_MAX_LINE_CHARS,
    ) -> ReplayDivergence:
        """Build a divergence, deriving ``reason`` and the bounded ``diff``.

        A side whose key is ``None`` has no call at ``index`` (``extra_call``
        when the recording lacks it, ``missing_call`` when the replay does).
        """
        reason: DivergenceReason
        if expected_key is None:
            reason = "extra_call"
        elif actual_key is None:
            reason = "missing_call"
        elif expected_name != actual_name:
            reason = "tool_changed"
        else:
            reason = "request_changed"

        def document(name: str | None, request: JsonValue) -> JsonValue:
            return {"name": name, "arguments": request} if kind == "tool" else request

        diff, truncated = request_diff(
            document(expected_name, expected_request),
            document(actual_name, actual_request),
            expected_label=f"recorded {kind} call #{index}",
            actual_label=f"replayed {kind} call #{index}",
            present=(expected_key is not None, actual_key is not None),
            max_lines=max_lines,
            max_line_chars=max_line_chars,
        )
        return cls(
            kind=kind,
            index=index,
            reason=reason,
            mode=mode,
            expected_name=expected_name,
            actual_name=actual_name,
            expected_key=expected_key,
            actual_key=actual_key,
            expected_request=expected_request,
            actual_request=actual_request,
            diff=diff,
            diff_truncated=truncated,
            span_id=span_id,
        )

    def summary(self) -> str:
        """One line describing the divergence, e.g. ``llm call #2: request differs ...``."""
        call = f"{self.kind} call #{self.index}"
        name = self.actual_name if self.actual_name is not None else self.expected_name
        if name is not None:
            call += f" ({name!r})"
        if self.reason == "extra_call":
            text = f"{call}: not in the recording (the replay made an extra call)"
        elif self.reason == "missing_call":
            text = f"{call}: recorded but not made by the replay"
        elif self.reason == "tool_changed":
            text = f"{call}: recorded tool is {self.expected_name!r}"
        else:
            text = f"{call}: request differs from the recording"
        return f"{text} [{self.mode}]" if self.mode is not None else text

    def to_dict(self, *, include_requests: bool = True) -> dict[str, JsonValue]:
        """JSON-ready dict (for reports and trace metadata).

        ``include_requests=False`` leaves out the two full requests (the diff and
        the keys remain), which keeps the dict small; the replayed request is
        also the input of the span ``span_id``.
        """
        exclude = {"expected_request", "actual_request"} if not include_requests else None
        data: dict[str, JsonValue] = self.model_dump(mode="json", exclude=exclude)
        data["summary"] = self.summary()
        return data

    def __str__(self) -> str:
        return f"{self.summary()}\n{self.diff}" if self.diff else self.summary()
