"""The recorder: owns one in-progress :class:`~agentlens.schema.Trace`.

A :class:`Recorder` is created by :func:`agentlens.run` and lives for one agent
run. Spans are buffered in memory on the trace and handed to the *sink* (later:
the store) once, when the run ends (``docs/design.md`` §2).

The recorder is the only object that mutates the trace. It is safe to use from
several threads at once (spans may be started from a thread pool), and the
public helpers in :mod:`agentlens.tracing.api` call it through guards that log
instead of raising, so a bug here can never break the agent (FR-T5).
"""

from __future__ import annotations

import logging
import threading
from collections.abc import Callable, Mapping

from pydantic import JsonValue

from agentlens.schema import Span, SpanKind, Trace
from agentlens.schema._base import utc_now
from agentlens.tracing._serialize import format_exception, to_json_value
from agentlens.tracing.redaction import DEFAULT_REDACTOR, Redactor, redact_trace

__all__ = ["Recorder", "TraceSink"]

logger = logging.getLogger("agentlens.tracing")

TraceSink = Callable[[Trace], None]
"""Receives each finished trace, e.g. ``store.save_trace`` or ``list.append``."""


def _json_mapping(values: Mapping[str, object] | None) -> dict[str, JsonValue]:
    return {str(k): to_json_value(v) for k, v in (values or {}).items()}


class Recorder:
    """Builds one trace span by span and flushes it to a sink when finished."""

    def __init__(
        self,
        name: str,
        *,
        input: object = None,
        metadata: Mapping[str, object] | None = None,
        sink: TraceSink | None = None,
        redactor: Redactor | None = DEFAULT_REDACTOR,
    ) -> None:
        self.trace = Trace(
            name=name,
            input=to_json_value(input),
            metadata=_json_mapping(metadata),
        )
        self._sink = sink
        self._redactor = redactor
        self._lock = threading.Lock()
        self._finished = False

    @property
    def finished(self) -> bool:
        """Whether :meth:`finish` has already run."""
        return self._finished

    def start_span(
        self,
        kind: SpanKind,
        name: str,
        *,
        parent_id: str | None = None,
        input: object = None,
        attributes: Mapping[str, object] | None = None,
    ) -> Span:
        """Create a running span and append it to the trace.

        Spans are appended in start order, so a parent always precedes its
        children in ``trace.spans``.
        """
        span = Span(
            trace_id=self.trace.id,
            parent_id=parent_id,
            kind=kind,
            name=name,
            input=to_json_value(input),
            attributes=_json_mapping(attributes),
        )
        with self._lock:
            if self._finished:
                logger.warning("span %r started after run %r ended", name, self.trace.name)
            self.trace.spans.append(span)
        return span

    def end_span(
        self, span: Span, *, output: object = None, error: BaseException | None = None
    ) -> None:
        """Close ``span`` with its output, or with the exception that ended it."""
        if error is not None:
            span.error = format_exception(error)
        else:
            span.output = to_json_value(output)
        # Wall clocks can step backwards (NTP); never produce end < start.
        span.end_time = max(utc_now(), span.start_time)

    def finish(self, *, error: BaseException | None = None) -> Trace:
        """End the run and hand the trace to the sink. Idempotent.

        The sink receives a redacted copy (see :func:`redact_trace`); the
        returned live trace is left untouched. If redaction fails the trace is
        *not* passed to the sink: dropping a trace is better than persisting a
        secret (NFR-5).
        """
        with self._lock:
            if self._finished:
                return self.trace
            self._finished = True
        if error is not None:
            self.trace.error = format_exception(error)
        self.trace.end_time = max(utc_now(), self.trace.start_time)
        if self._sink is not None:
            try:
                persisted = (
                    redact_trace(self.trace, self._redactor)
                    if self._redactor is not None
                    else self.trace
                )
            except Exception:
                logger.exception(
                    "redaction failed for run %r; the trace was not persisted", self.trace.name
                )
                return self.trace
            try:
                self._sink(persisted)
            except Exception:
                logger.exception("trace sink failed for run %r", self.trace.name)
        return self.trace
