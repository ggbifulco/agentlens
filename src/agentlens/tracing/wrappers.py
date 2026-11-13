"""Automatic spans for LLM clients and tools: :func:`wrap_llm`, :func:`wrap_tool`.

.. code-block:: python

    llm = agentlens.wrap_llm(MockLLM(["It shipped."]))

    @agentlens.wrap_tool
    def lookup_order(order_id: str) -> dict[str, str]: ...

    with agentlens.run("support-agent") as trace:
        llm.complete(LLMRequest(messages="Where is order 1234?"))  # -> llm_call span
        lookup_order("1234")                                        # -> tool_call span

Both wrappers follow the rules of :mod:`agentlens.tracing.api`: outside
:func:`~agentlens.run` they call straight through (one ``ContextVar.get`` of
overhead), exceptions from the wrapped call are recorded on the span and
re-raised unchanged, and recording problems are logged, never raised (FR-T5).

They are also the seams replay builds on (``docs/design.md`` §4): an
``llm_call`` span carries the request's ``request_hash``
(:meth:`~agentlens.llm.LLMRequest.fingerprint`) and the full request/response
JSON, a ``tool_call`` span the tool name and bound arguments, so a recorded
trace contains everything a cassette needs. A cassette is itself an
:class:`~agentlens.llm.LLMClient`, so ``wrap_llm(cassette)`` traces playback the
same way as live calls.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any, ParamSpec, TypeVar, overload

from pydantic import JsonValue

from agentlens.llm.base import LLMClient
from agentlens.llm.types import LLMRequest, LLMResponse
from agentlens.schema import SpanKind, Usage
from agentlens.tracing.api import _close, _current_recorder, _open, _OpenSpan, trace

__all__ = ["TOOL_NAME_ATTR", "TracedLLM", "wrap_llm", "wrap_tool"]

logger = logging.getLogger("agentlens.tracing")

P = ParamSpec("P")
R = TypeVar("R")

TOOL_NAME_ATTR = "__agentlens_tool__"
"""Attribute set on functions returned by :func:`wrap_tool`: the tool (span) name."""


class TracedLLM:
    """An :class:`~agentlens.llm.LLMClient` that records each call as an ``llm_call`` span.

    Created by :func:`wrap_llm`; see there for what is recorded. Attributes not
    defined here are looked up on the wrapped client, so client-specific helpers
    (e.g. ``MockLLM.requests``) keep working through the wrapper.
    """

    def __init__(
        self,
        client: LLMClient,
        *,
        name: str | None = None,
        provider: str | None = None,
        include_raw: bool = False,
    ) -> None:
        self.client = client
        """The wrapped client; calls go straight to it outside a run."""
        self.include_raw = include_raw
        cls = type(client)
        default_model = getattr(client, "model", None)
        self._default_model = default_model if isinstance(default_model, str) else None
        self._name = name
        self._static_attributes: dict[str, object] = {
            "client": f"{cls.__module__}.{cls.__qualname__}"
        }
        if provider is None:
            client_provider = getattr(client, "provider", None)
            provider = client_provider if isinstance(client_provider, str) else None
        if provider is not None:
            self._static_attributes["provider"] = provider

    def __getattr__(self, name: str) -> Any:
        # Only called for attributes missing on the wrapper itself.
        if name == "client":  # not yet set (e.g. during unpickling): avoid recursion
            raise AttributeError(name)
        return getattr(self.client, name)

    def __repr__(self) -> str:
        return f"{type(self).__name__}({self.client!r})"

    # -- LLMClient -----------------------------------------------------------

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Call the wrapped client's ``complete``, recording an ``llm_call`` span."""
        if _current_recorder.get() is None:
            return self.client.complete(request)
        opened = _open(
            SpanKind.LLM_CALL,
            self._span_name(request),
            input_factory=lambda: _request_input(request),
            attributes=self._request_attributes(request),
        )
        try:
            response = self.client.complete(request)
        except BaseException as exc:
            _close(opened, error=exc)
            raise
        self._finish(opened, response)
        return response

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Call the wrapped client's ``acomplete``, recording an ``llm_call`` span."""
        if _current_recorder.get() is None:
            return await self.client.acomplete(request)
        opened = _open(
            SpanKind.LLM_CALL,
            self._span_name(request),
            input_factory=lambda: _request_input(request),
            attributes=self._request_attributes(request),
        )
        try:
            response = await self.client.acomplete(request)
        except BaseException as exc:
            _close(opened, error=exc)
            raise
        self._finish(opened, response)
        return response

    # -- recording helpers (only reached inside a run) -----------------------

    def _span_name(self, request: LLMRequest) -> str:
        return self._name or request.model or self._default_model or type(self.client).__name__

    def _request_attributes(self, request: LLMRequest) -> Mapping[str, object] | None:
        try:
            attributes = dict(self._static_attributes)
            attributes.update(
                model=request.model or self._default_model,
                temperature=request.temperature,
                max_tokens=request.max_tokens,
                n_messages=len(request.messages),
                n_tools=len(request.tools),
                request_hash=request.fingerprint(),
            )
            if request.tools:
                attributes["tool_names"] = [tool.name for tool in request.tools]
            if request.response_format is not None:
                attributes["response_format"] = request.response_format.get("type")
            return attributes
        except Exception:
            logger.exception("could not build llm_call attributes")
            return None

    def _finish(self, opened: _OpenSpan | None, response: object) -> None:
        usage: Usage | None = None
        attributes: dict[str, object] = {}
        if isinstance(response, LLMResponse):
            usage = response.usage
            attributes = {
                "response_model": response.model,
                "n_tool_calls": len(response.tool_calls),
            }
        _close(
            opened,
            output_factory=lambda: _response_output(response, include_raw=self.include_raw),
            usage=usage,
            attributes=attributes,
        )


def _request_input(request: LLMRequest) -> JsonValue:
    return request.model_dump(mode="json", exclude_defaults=True)


def _response_output(response: object, *, include_raw: bool) -> object:
    if not isinstance(response, LLMResponse):  # a client breaking the protocol
        return response
    # ``usage`` lives on the span itself; ``raw`` is often large provider JSON.
    exclude = {"usage"} if include_raw else {"usage", "raw"}
    return response.model_dump(mode="json", exclude=exclude)


def wrap_llm(
    client: LLMClient,
    *,
    name: str | None = None,
    provider: str | None = None,
    include_raw: bool = False,
) -> TracedLLM:
    """Wrap an LLM client so every call inside :func:`~agentlens.run` is recorded.

    Each ``complete``/``acomplete`` call becomes an ``llm_call`` span:

    * **name**: ``name`` if given, else the requested model, else the client's
      ``model`` attribute, else the client class name;
    * **input**: the :class:`~agentlens.llm.LLMRequest` as JSON (fields left at
      their defaults omitted);
    * **output**: the :class:`~agentlens.llm.LLMResponse` as JSON, without
      ``usage`` (stored in ``span.usage``) and without the provider's ``raw``
      payload unless ``include_raw=True``;
    * **usage**: the response's token counts and cost (``None`` if unreported);
    * **attributes**: ``client`` (class path), ``provider`` (if known), ``model``,
      ``temperature``, ``max_tokens``, ``n_messages``, ``n_tools``,
      ``tool_names``, ``response_format``, ``request_hash`` (cassette key, see
      ADR 0004) and, once answered, ``response_model`` and ``n_tool_calls``.

    Errors raised by the client are stored in ``span.error`` and re-raised.
    Outside a run the wrapper calls straight through. Wrapping an already
    wrapped client returns it unchanged.

    .. code-block:: python

        llm = agentlens.wrap_llm(MockLLM(["Hello!"]))
        with agentlens.run("agent") as trace:
            llm.complete(LLMRequest(messages="Hi"))
        assert trace.spans[0].usage is not None

    Args:
        client: Any object implementing :class:`~agentlens.llm.LLMClient`.
        name: Fixed span name instead of the model name.
        provider: ``provider`` attribute; defaults to the client's own
            ``provider`` attribute when it is a string.
        include_raw: Also record ``LLMResponse.raw`` in the span output.
    """
    if isinstance(client, TracedLLM):
        return client
    return TracedLLM(client, name=name, provider=provider, include_raw=include_raw)


@overload
def wrap_tool(fn: Callable[P, R], /, *, name: str | None = ...) -> Callable[P, R]: ...


@overload
def wrap_tool(*, name: str | None = ...) -> Callable[[Callable[P, R]], Callable[P, R]]: ...


def wrap_tool(
    fn: Callable[P, R] | None = None,
    /,
    *,
    name: str | None = None,
) -> Callable[P, R] | Callable[[Callable[P, R]], Callable[P, R]]:
    """Record each call of a tool (sync or async callable) as a ``tool_call`` span.

    Use it as a plain wrapper or as a decorator, with or without a name:

    .. code-block:: python

        search = agentlens.wrap_tool(index.search, name="search_docs")

        @agentlens.wrap_tool
        def lookup_order(order_id: str) -> dict[str, str]: ...

        @agentlens.wrap_tool(name="refund")
        async def issue_refund(order_id: str, amount: float) -> bool: ...

    The span name is the tool name (``name``, else the function's
    ``__name__``), which is what ``Trace.tool_calls`` and the
    ``tool_sequence`` evaluator compare. The input is the call's arguments by
    parameter name (``self``/``cls`` dropped), the output the return value; an
    exception is recorded in ``span.error`` and re-raised. Outside a run the
    tool is called directly. The returned function exposes the original as
    ``__wrapped__`` and the tool name as ``__agentlens_tool__``; wrapping it
    again returns it unchanged, or re-wraps the original under a new ``name``
    (never two nested spans).

    Raises:
        TypeError: if ``fn`` is a generator function (not supported yet).
    """

    def decorate(func: Callable[P, R]) -> Callable[P, R]:
        existing = getattr(func, TOOL_NAME_ATTR, None)
        if isinstance(existing, str):
            if name in (None, existing):
                return func
            func = func.__wrapped__  # type: ignore[attr-defined]  # rename: re-wrap the original
        tool_name = name or getattr(func, "__name__", None) or type(func).__name__
        wrapped = trace(kind=SpanKind.TOOL_CALL, name=tool_name)(func)
        setattr(wrapped, TOOL_NAME_ATTR, tool_name)  # always a plain function from trace()
        return wrapped

    if fn is not None:
        return decorate(fn)
    return decorate
