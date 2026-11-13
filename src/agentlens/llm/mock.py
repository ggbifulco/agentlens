"""``MockLLM``: a scripted, deterministic :class:`~agentlens.llm.LLMClient` (FR-M2).

Used by CI (no API keys, NFR-2), the example agent and tests of everything that
talks to an LLM. Three ways to script it::

    MockLLM(["Hello!", ToolCall(id="c1", name="lookup_order", arguments={"id": 1})])
    MockLLM(lambda request: request.last_user_message.upper())
    MockLLM({r"refund": "Refund issued.", r"order \\d+": "Shipped."}, default="?")
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Callable, Mapping, Sequence
from typing import TypeAlias

from agentlens.llm.types import LLMRequest, LLMResponse, ToolCall
from agentlens.schema import Usage

__all__ = [
    "MockExhaustedError",
    "MockLLM",
    "MockLLMError",
    "MockNoMatchError",
    "ScriptItem",
    "count_tokens",
]

ScriptItem: TypeAlias = LLMResponse | str | ToolCall | list[ToolCall]
"""What a script may produce: a full response, plain text, or tool call(s)."""

Responder: TypeAlias = Callable[[LLMRequest], ScriptItem]
Script: TypeAlias = Sequence[ScriptItem] | Responder | Mapping[str | re.Pattern[str], ScriptItem]


class MockLLMError(RuntimeError):
    """Base class for scripting errors raised by :class:`MockLLM`."""


class MockExhaustedError(MockLLMError):
    """A list script ran out of responses (and ``cycle`` is off)."""


class MockNoMatchError(MockLLMError):
    """No pattern of a regex script matched and no ``default`` was given."""


def count_tokens(text: str | None) -> int:
    """Deterministic stand-in for a tokenizer: the number of whitespace-separated words."""
    return len(text.split()) if text else 0


def _tool_calls_text(tool_calls: Sequence[ToolCall]) -> str:
    return " ".join(
        f"{call.name} {json.dumps(call.arguments, sort_keys=True)}" for call in tool_calls
    )


def _request_tokens(request: LLMRequest) -> int:
    return sum(
        count_tokens(m.content) + count_tokens(_tool_calls_text(m.tool_calls))
        for m in request.messages
    )


def _to_response(item: object) -> LLMResponse:
    if isinstance(item, LLMResponse):
        return item
    if isinstance(item, str):
        return LLMResponse(content=item)
    if isinstance(item, ToolCall):
        return LLMResponse(tool_calls=[item])
    if isinstance(item, list) and all(isinstance(call, ToolCall) for call in item):
        return LLMResponse(tool_calls=item)
    raise TypeError(
        "MockLLM script items must be LLMResponse, str, ToolCall or list[ToolCall], "
        f"got {type(item).__name__}"
    )


class MockLLM:
    """A deterministic LLM client driven by a script.

    ``script`` is one of:

    * a **sequence** of items, returned in order, one per call. When exhausted it
      raises :class:`MockExhaustedError` (loud failure: an agent making more calls
      than scripted is usually a bug), or starts over if ``cycle=True``;
    * a **callable** ``(request) -> item``, for responses computed from the request;
    * a **mapping** ``pattern -> item``: the first pattern (in insertion order) that
      ``re.search``-matches the last user message wins; otherwise ``default`` is
      returned, or :class:`MockNoMatchError` is raised if there is no default.

    With a sequence script, ``default`` (if given) also answers every call after
    the script is exhausted, instead of raising.

    Items are an :class:`LLMResponse`, plain text, a :class:`ToolCall` or a list of
    them. Every response gets ``model`` (the request's, else this client's) and,
    unless the item set its own, deterministic ``usage``: whitespace word counts of
    the request messages (input) and of the response (output), at ``cost_usd=0``.

    Every request is recorded in :attr:`requests` for assertions in tests.
    """

    def __init__(
        self,
        script: Script,
        *,
        model: str = "mock",
        cycle: bool = False,
        default: ScriptItem | None = None,
    ) -> None:
        self.model = model
        self.cycle = cycle
        self.default = default
        self.requests: list[LLMRequest] = []
        self._lock = threading.Lock()
        self._responder: Responder | None = None
        self._items: list[ScriptItem] = []
        self._patterns: list[tuple[re.Pattern[str], ScriptItem]] | None = None
        if callable(script):
            self._responder = script
        elif isinstance(script, Mapping):
            self._patterns = [(re.compile(p), item) for p, item in script.items()]
        elif isinstance(script, str):
            raise TypeError("MockLLM script must be a list of items, not a single string")
        else:
            self._items = list(script)

    @property
    def calls(self) -> int:
        """Number of requests served (or attempted) so far."""
        return len(self.requests)

    def reset(self) -> None:
        """Forget recorded requests and restart a list script from its first item."""
        with self._lock:
            self.requests.clear()

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Return the next scripted response for ``request``."""
        with self._lock:
            index = len(self.requests)
            self.requests.append(request)
        response = _to_response(self._pick(request, index))
        usage = response.usage or Usage(
            input_tokens=_request_tokens(request),
            output_tokens=count_tokens(response.content)
            + count_tokens(_tool_calls_text(response.tool_calls)),
            cost_usd=0.0,
        )
        return response.model_copy(
            update={"usage": usage, "model": response.model or request.model or self.model}
        )

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Async variant of :meth:`complete` (no I/O, returns immediately)."""
        return self.complete(request)

    def _pick(self, request: LLMRequest, index: int) -> object:
        if self._responder is not None:
            return self._responder(request)
        if self._patterns is not None:
            text = request.last_user_message or ""
            for pattern, item in self._patterns:
                if pattern.search(text):
                    return item
            if self.default is None:
                raise MockNoMatchError(f"no MockLLM pattern matches user message {text!r}")
            return self.default
        if index < len(self._items):
            return self._items[index]
        if self.cycle and self._items:
            return self._items[index % len(self._items)]
        if self.default is not None:
            return self.default
        raise MockExhaustedError(
            f"MockLLM script has {len(self._items)} responses but call #{index + 1} was made"
        )
