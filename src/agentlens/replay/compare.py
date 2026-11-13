"""Compare the calls of two recorded traces (FR-R3 for traces, not only cassettes)."""

from __future__ import annotations

from agentlens.replay._types import EntryKind
from agentlens.replay.cassette import _SPAN_KINDS, Cassette, replayable_spans
from agentlens.replay.divergence import ReplayDivergence
from agentlens.schema import Trace

__all__ = ["find_divergence"]


def find_divergence(
    recorded: Trace, replayed: Trace, *, kind: EntryKind = "llm"
) -> ReplayDivergence | None:
    """First call of ``kind`` where ``replayed`` departs from ``recorded``, or ``None``.

    The calls of each trace are selected and keyed exactly like
    :meth:`Cassette.from_trace <agentlens.replay.Cassette.from_trace>` does
    (``llm_call`` or ``tool_call`` spans in start order; failed, unfinished and
    tool-nested spans skipped), then compared index by index: a different
    request key or tool name gives ``request_changed``/``tool_changed``, a call
    only in ``replayed`` ``extra_call`` and a call only in ``recorded``
    ``missing_call``. ``span_id`` is the replayed span (``None`` for a
    missing call). Works for any two traces, e.g. a recorded run and a live
    replay against another model (FR-R2).

    .. code-block:: python

        divergence = find_divergence(baseline_trace, new_trace)
        if divergence is not None:
            print(divergence)  # summary + unified diff of the two requests

    Raises:
        CassetteFormatError: if an ``llm_call`` span is not a valid call.
    """
    expected = Cassette.from_trace(recorded).entries_of(kind)
    actual = Cassette.from_trace(replayed).entries_of(kind)
    actual_spans = _call_span_ids(replayed, kind)
    for index in range(max(len(expected), len(actual))):
        before = expected[index] if index < len(expected) else None
        after = actual[index] if index < len(actual) else None
        if (
            before is not None
            and after is not None
            and (before.key, before.name) == (after.key, after.name)
        ):
            continue
        return ReplayDivergence.create(
            kind,
            index,
            expected_request=before.request if before is not None else None,
            actual_request=after.request if after is not None else None,
            expected_key=before.key if before is not None else None,
            actual_key=after.key if after is not None else None,
            expected_name=before.name if before is not None else None,
            actual_name=after.name if after is not None else None,
            span_id=actual_spans[index] if after is not None else None,
        )
    return None


def _call_span_ids(trace: Trace, kind: EntryKind) -> list[str]:
    """Ids of the spans :meth:`Cassette.from_trace` turns into entries of ``kind``, in order."""
    return [span.id for span in replayable_spans(trace) if span.kind is _SPAN_KINDS[kind]]
