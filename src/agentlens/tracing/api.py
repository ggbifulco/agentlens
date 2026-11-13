"""Public recording API: :func:`run`, :func:`trace`, :func:`span` (FR-T1).

The *current* recorder and span live in :mod:`contextvars`, not in globals or
thread-locals:

* every ``asyncio`` task gets a copy of the context when it is created, so
  ``asyncio.gather(a(), b())`` inside a traced function gives two sibling spans
  with the right parent;
* ``asyncio.to_thread`` and ``contextvars.copy_context().run`` carry the
  context into worker threads, while unrelated threads see no active run.

Code that runs outside :func:`run` is not recorded: decorated functions behave
exactly like the undecorated ones, at the cost of one ``ContextVar.get``.

Recording problems are logged on the ``agentlens.tracing`` logger and never
raised into user code; exceptions raised *by* user code are recorded on the
span/trace and re-raised unchanged.
"""

from __future__ import annotations

import functools
import inspect
import logging
from collections.abc import Awaitable, Callable, Iterator, Mapping
from contextlib import contextmanager
from contextvars import ContextVar, Token
from typing import Any, ParamSpec, TypeVar, cast, overload

from agentlens.schema import Span, SpanKind, Trace, Usage
from agentlens.tracing._serialize import bind_arguments
from agentlens.tracing.recorder import Recorder, TraceSink
from agentlens.tracing.redaction import DEFAULT_REDACTOR, Redactor

__all__ = ["current_span", "current_trace", "run", "span", "trace"]

logger = logging.getLogger("agentlens.tracing")

P = ParamSpec("P")
R = TypeVar("R")

_current_recorder: ContextVar[Recorder | None] = ContextVar("agentlens_recorder", default=None)
_current_span: ContextVar[Span | None] = ContextVar("agentlens_span", default=None)


def current_trace() -> Trace | None:
    """The trace being recorded in this context, or ``None`` outside :func:`run`."""
    recorder = _current_recorder.get()
    return recorder.trace if recorder is not None else None


def current_span() -> Span | None:
    """The innermost open span in this context, or ``None``."""
    return _current_span.get()


# -- runs ----------------------------------------------------------------------


@contextmanager
def run(
    name: str,
    *,
    input: object = None,
    metadata: Mapping[str, object] | None = None,
    sink: TraceSink | None = None,
    redactor: Redactor | None = DEFAULT_REDACTOR,
) -> Iterator[Trace]:
    """Record one agent run as a :class:`~agentlens.schema.Trace`.

    .. code-block:: python

        with agentlens.run("support-agent", input=question) as trace:
            trace.output = agent(question)
        print(trace.latency_ms, trace.tool_calls)

    Spans created inside the block (by :func:`trace`, :func:`span` or the LLM
    wrappers) are attached to the yielded trace. When the block exits the trace
    gets its ``end_time`` (and ``error`` if the block raised) and a redacted
    copy is passed to ``sink``; the yielded trace itself is not redacted.
    Exceptions from the block are re-raised after recording.

    Runs may be nested; an inner run records an independent trace.

    Args:
        name: Trace name, e.g. the agent or suite name.
        input: The agent input; converted to JSON (``repr`` as a fallback).
        metadata: Run context such as git sha, case id or repetition.
        sink: Called once with the finished trace (e.g. ``store.save_trace``).
            Errors raised by the sink are logged, not propagated.
        redactor: Masks secrets in the copy given to ``sink`` (NFR-5); defaults
            to :data:`~agentlens.tracing.DEFAULT_REDACTOR`. ``None`` disables
            redaction. If it raises, the error is logged and the trace is not
            passed to the sink.
    """
    try:
        recorder: Recorder | None = Recorder(
            name, input=input, metadata=metadata, sink=sink, redactor=redactor
        )
    except Exception:
        logger.exception("could not start run %r; it will not be recorded", name)
        recorder = None
    if recorder is None:
        # Keep the user's ``with ... as trace`` working with a detached trace.
        yield Trace(name=str(name))
        return

    recorder_token = _current_recorder.set(recorder)
    span_token = _current_span.set(None)  # spans of this run start at the root
    try:
        yield recorder.trace
    except BaseException as exc:
        _finish_safely(recorder, exc)
        raise
    else:
        _finish_safely(recorder, None)
    finally:
        _reset(_current_span, span_token)
        _reset(_current_recorder, recorder_token)


def _finish_safely(recorder: Recorder, error: BaseException | None) -> None:
    try:
        recorder.finish(error=error)
    except Exception:
        logger.exception("could not finish run %r", recorder.trace.name)


def _reset(var: ContextVar[Any], token: Token[Any]) -> None:
    try:
        var.reset(token)
    except ValueError:  # token created in another context
        logger.warning("agentlens context exited in a different context than it was entered")


# -- spans ---------------------------------------------------------------------


class _OpenSpan:
    """A span started by this module plus the context token to restore."""

    __slots__ = ("recorder", "span", "token")

    def __init__(self, recorder: Recorder, span: Span, token: Token[Span | None]) -> None:
        self.recorder = recorder
        self.span = span
        self.token = token


def _open(
    kind: SpanKind,
    name: str,
    *,
    input: object = None,
    input_factory: Callable[[], object] | None = None,
    attributes: Mapping[str, object] | None = None,
) -> _OpenSpan | None:
    """Start a span under the current one; ``None`` if not recording."""
    recorder = _current_recorder.get()
    if recorder is None:
        return None
    try:
        parent = _current_span.get()
        value = input_factory() if input_factory is not None else input
        new = recorder.start_span(
            kind,
            name,
            parent_id=parent.id if parent is not None else None,
            input=value,
            attributes=attributes,
        )
        return _OpenSpan(recorder, new, _current_span.set(new))
    except Exception:
        logger.exception("could not start span %r", name)
        return None


def _close(
    opened: _OpenSpan | None,
    *,
    output: object = None,
    output_factory: Callable[[], object] | None = None,
    error: BaseException | None = None,
    usage: Usage | None = None,
    attributes: Mapping[str, object] | None = None,
) -> None:
    """End a span opened by :func:`_open`; never raises.

    ``output_factory`` (only called on success) lets callers defer building the
    output until here, so a failure while building it is logged like any other
    recording problem instead of reaching user code.
    """
    if opened is None:
        return
    _reset(_current_span, opened.token)
    try:
        if error is None and output_factory is not None:
            output = output_factory()
        opened.recorder.end_span(
            opened.span, output=output, error=error, usage=usage, attributes=attributes
        )
    except Exception:
        logger.exception("could not end span %r", opened.span.name)


@contextmanager
def span(
    name: str,
    *,
    kind: SpanKind | str = SpanKind.CUSTOM,
    input: object = None,
    attributes: Mapping[str, object] | None = None,
) -> Iterator[Span | None]:
    """Record a block of code as a span.

    Yields the running :class:`~agentlens.schema.Span` (set ``span.output`` to
    record a result), or ``None`` when no run is active.

    .. code-block:: python

        with agentlens.span("search", kind="retrieval", input=query) as s:
            docs = index.search(query)
            if s is not None:
                s.output = [d.id for d in docs]
    """
    opened = _open(SpanKind(kind), name, input=input, attributes=attributes)
    try:
        yield opened.span if opened is not None else None
    except BaseException as exc:
        _close(opened, error=exc)
        raise
    else:
        _close(opened, output=opened.span.output if opened is not None else None)


# -- decorator -------------------------------------------------------------------


def _signature(fn: Callable[..., Any]) -> inspect.Signature | None:
    """``inspect.signature(fn)``, or ``None`` for callables without one (some builtins)."""
    try:
        return inspect.signature(fn)
    except (TypeError, ValueError):
        return None


@overload
def trace(func: Callable[P, R], /) -> Callable[P, R]: ...


@overload
def trace(
    *, kind: SpanKind | str = ..., name: str | None = ...
) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def trace(
    func: Callable[P, R] | None = None,
    /,
    *,
    kind: SpanKind | str = SpanKind.CUSTOM,
    name: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Decorator recording each call of a sync or async function as a span.

    The span input is the call's arguments by parameter name (without
    ``self``/``cls``), the output is the return value, and an exception is
    stored in ``span.error`` before being re-raised.

    .. code-block:: python

        @agentlens.trace(kind="tool_call")
        def lookup_order(order_id: str) -> dict: ...

        @agentlens.trace                      # kind=custom, name=function name
        async def plan(question: str) -> list[str]: ...

    Args:
        kind: The :class:`~agentlens.schema.SpanKind` (or its string value).
        name: Span name; defaults to the function's ``__qualname__``.

    Raises:
        TypeError: if applied to a generator function (not supported yet).
        ValueError: if ``kind`` is not a valid span kind.
    """
    span_kind = SpanKind(kind)

    def decorate(fn: Callable[P, R]) -> Callable[P, R]:
        if inspect.isgeneratorfunction(fn) or inspect.isasyncgenfunction(fn):
            raise TypeError(f"@trace does not support generator functions ({fn.__qualname__})")
        span_name = name or getattr(fn, "__qualname__", None) or type(fn).__qualname__
        signature = _signature(fn)

        if inspect.iscoroutinefunction(fn):
            async_fn = cast(Callable[..., Awaitable[Any]], fn)

            @functools.wraps(fn)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                if _current_recorder.get() is None:
                    return await async_fn(*args, **kwargs)
                opened = _open(
                    span_kind,
                    span_name,
                    input_factory=lambda: bind_arguments(signature, args, kwargs),
                )
                try:
                    result = await async_fn(*args, **kwargs)
                except BaseException as exc:
                    _close(opened, error=exc)
                    raise
                _close(opened, output=result)
                return result

            return cast(Callable[P, R], async_wrapper)

        @functools.wraps(fn)
        def wrapper(*args: P.args, **kwargs: P.kwargs) -> R:
            if _current_recorder.get() is None:
                return fn(*args, **kwargs)
            opened = _open(
                span_kind,
                span_name,
                input_factory=lambda: bind_arguments(signature, args, kwargs),
            )
            try:
                result = fn(*args, **kwargs)
            except BaseException as exc:
                _close(opened, error=exc)
                raise
            _close(opened, output=result)
            return result

        return wrapper

    if func is not None:
        return decorate(func)
    return decorate
