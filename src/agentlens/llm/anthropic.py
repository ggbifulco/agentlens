"""``AnthropicClient``: an adapter for the Anthropic Messages API.

Talks to ``POST {base_url}/v1/messages`` with ``httpx`` directly (no vendor SDK,
ADR 0002), reusing the retries, backoff and error mapping of
:mod:`agentlens.llm._http`::

    from agentlens.llm import AnthropicClient, LLMRequest, Message

    llm = AnthropicClient(model="claude-3-5-haiku-latest")  # key from ANTHROPIC_API_KEY
    response = llm.complete(
        LLMRequest(messages=[Message.system("Be brief."), Message.user("Say hi")])
    )

The provider-neutral :class:`~agentlens.llm.LLMRequest` is translated as follows:

* ``system`` messages (wherever they appear) are joined with a blank line into the
  top-level ``system`` field;
* :class:`~agentlens.llm.ToolSpec` → ``{"name", "description", "input_schema"}``;
* assistant ``tool_calls`` → ``tool_use`` content blocks (after the text, if any);
* ``tool`` messages → ``tool_result`` blocks in a ``user`` turn; consecutive tool
  results (and a user message right after them) are merged into one user turn,
  as the API expects all results of one assistant turn together;
* ``max_tokens`` is required by Anthropic: the client's ``max_tokens`` (default
  :data:`DEFAULT_MAX_TOKENS`) is used when the request has none;
* ``response_format``: Anthropic has no JSON mode, so a JSON request
  (``json_object`` or ``json_schema``) becomes an instruction appended to the
  system prompt; ``{"type": "text"}`` is ignored and any other type raises
  :class:`~agentlens.llm.LLMError`. The output is *not* validated as JSON.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from types import TracebackType
from typing import Any, Self

import httpx
from pydantic import JsonValue

from agentlens.llm._http import JSONHTTPClient, RetryPolicy
from agentlens.llm.errors import LLMError, LLMResponseError
from agentlens.llm.types import LLMRequest, LLMResponse, Message, ToolCall, ToolSpec
from agentlens.schema import Usage

__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "AnthropicClient",
    "parse_response",
]

DEFAULT_ANTHROPIC_BASE_URL = "https://api.anthropic.com"
"""The ``base_url`` of the official Anthropic API (the client appends ``/v1/messages``)."""

ANTHROPIC_VERSION = "2023-06-01"
"""Value of the ``anthropic-version`` header sent with every request."""

DEFAULT_MAX_TOKENS = 1024
"""``max_tokens`` sent when neither the request nor the client sets one."""

_API_KEY_ENV = "ANTHROPIC_API_KEY"

_JSON_INSTRUCTION = (
    "Respond only with a single valid JSON object. "
    "Do not add any text, explanation or Markdown code fences around it."
)


class AnthropicClient:
    """An :class:`~agentlens.llm.LLMClient` for the Anthropic Messages API.

    Args:
        model: Default model (e.g. ``"claude-3-5-haiku-latest"``), used when a
            request has no ``model``. One of the two must be set.
        base_url: API root without ``/v1`` (a proxy or gateway, for instance).
        api_key: Sent as ``x-api-key``. ``None`` falls back to the
            ``ANTHROPIC_API_KEY`` environment variable; when neither is set no key
            header is sent (e.g. behind a gateway that adds it). Pass ``""`` to never
            send a key, even if the variable is set.
        max_tokens: ``max_tokens`` used when a request has none (Anthropic
            requires the field).
        timeout: Per-request timeout in seconds (or an ``httpx.Timeout``).
        max_retries: Retries on HTTP 408/429/5xx (including Anthropic's 529
            "overloaded") and transport errors, with exponential backoff
            (``Retry-After`` is honoured). Ignored when ``retry_policy`` is given.
        retry_policy: Full control over retries/backoff/jitter (see :class:`RetryPolicy`).
        headers: Extra HTTP headers sent with every request (e.g. ``anthropic-beta``).
        provider: Value of the ``provider`` attribute that
            :func:`agentlens.wrap_llm` records (default ``"anthropic"``).
        http_client: An ``httpx.Client`` to use for sync calls (not closed by this client).
        async_http_client: An ``httpx.AsyncClient`` to use for async calls.

    The API key is never logged, never part of ``repr`` and never copied into
    :attr:`LLMResponse.raw` or error messages. Use the client as a context manager
    (or call :meth:`close`) to release its connection pool.
    """

    def __init__(
        self,
        model: str | None = None,
        *,
        base_url: str = DEFAULT_ANTHROPIC_BASE_URL,
        api_key: str | None = None,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        timeout: float | httpx.Timeout = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
        headers: Mapping[str, str] | None = None,
        provider: str = "anthropic",
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        if max_tokens < 1:
            raise ValueError("max_tokens must be >= 1")
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.max_tokens = max_tokens
        self.provider = provider
        self._api_key = os.environ.get(_API_KEY_ENV) if api_key is None else api_key
        self._headers = dict(headers or {})
        self._http = JSONHTTPClient(
            timeout=timeout,
            retry=retry_policy or RetryPolicy(max_retries=max_retries),
            http_client=http_client,
            async_http_client=async_http_client,
        )

    @property
    def retry_policy(self) -> RetryPolicy:
        """The retry policy in use."""
        return self._http.retry

    @property
    def url(self) -> str:
        """The Messages API endpoint."""
        return f"{self.base_url}/v1/messages"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` to ``/v1/messages`` and return the normalised response.

        Raises:
            LLMError: Subclasses for HTTP errors (:class:`LLMHTTPError`,
                :class:`LLMRateLimitError`), unreachable servers
                (:class:`LLMConnectionError`, :class:`LLMTimeoutError`) and
                unparsable answers (:class:`LLMResponseError`); the base class for
                requests that cannot be expressed (no model, only system messages,
                unsupported ``response_format``).
        """
        payload = self.build_payload(request)
        return parse_response(self._http.post_json(self.url, payload, self._request_headers()))

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Async variant of :meth:`complete`."""
        payload = self.build_payload(request)
        data = await self._http.apost_json(self.url, payload, self._request_headers())
        return parse_response(data)

    def build_payload(self, request: LLMRequest) -> dict[str, Any]:
        """The JSON body sent for ``request`` (useful for debugging and tests)."""
        model = request.model or self.model
        if not model:
            raise LLMError("no model: set AnthropicClient(model=...) or LLMRequest.model")
        system_parts = [m.content or "" for m in request.messages if m.role == "system"]
        instruction = _response_format_instruction(request.response_format)
        if instruction:
            system_parts.append(instruction)
        messages = _messages_to_anthropic(request.messages)
        if not messages:
            raise LLMError("Anthropic needs at least one non-system message")
        payload: dict[str, Any] = {
            "model": model,
            "max_tokens": request.max_tokens or self.max_tokens,
            "messages": messages,
        }
        if system_parts:
            payload["system"] = "\n\n".join(system_parts)
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.tools:
            payload["tools"] = [_tool_to_anthropic(t) for t in request.tools]
        return payload

    def _request_headers(self) -> dict[str, str]:
        headers = {"anthropic-version": ANTHROPIC_VERSION, **self._headers}
        if self._api_key:
            headers["x-api-key"] = self._api_key
        return headers

    def close(self) -> None:
        """Release the sync connection pool (an injected ``http_client`` is left open)."""
        self._http.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()


def _response_format_instruction(response_format: Mapping[str, JsonValue] | None) -> str | None:
    """The system-prompt instruction emulating ``response_format`` (Anthropic has no JSON mode)."""
    if response_format is None:
        return None
    kind = response_format.get("type")
    if kind == "text":
        return None
    if kind == "json_object":
        return _JSON_INSTRUCTION
    if kind == "json_schema":
        spec = response_format.get("json_schema")
        schema = spec.get("schema") if isinstance(spec, dict) else None
        if schema is None:
            return _JSON_INSTRUCTION
        return (
            f"{_JSON_INSTRUCTION} It must conform to this JSON Schema:\n"
            f"{json.dumps(schema, sort_keys=True)}"
        )
    raise LLMError(f"unsupported response_format type for Anthropic: {kind!r}")


def _tool_to_anthropic(tool: ToolSpec) -> dict[str, Any]:
    out: dict[str, Any] = {"name": tool.name, "input_schema": tool.parameters}
    if tool.description:
        out["description"] = tool.description
    return out


def _assistant_content(message: Message) -> str | list[dict[str, Any]]:
    if not message.tool_calls:
        return message.content or ""
    blocks: list[dict[str, Any]] = []
    if message.content:  # the API rejects empty text blocks
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {"type": "tool_use", "id": call.id, "name": call.name, "input": call.arguments}
        for call in message.tool_calls
    )
    return blocks


def _messages_to_anthropic(messages: list[Message]) -> list[dict[str, Any]]:
    """Non-system messages as Anthropic turns, tool results folded into user turns."""
    out: list[dict[str, Any]] = []
    for message in messages:
        if message.role == "system":
            continue
        if message.role == "assistant":
            out.append({"role": "assistant", "content": _assistant_content(message)})
            continue
        if message.role == "tool":
            block: dict[str, Any] = {
                "type": "tool_result",
                "tool_use_id": message.tool_call_id,
                "content": message.content,
            }
        else:
            block = {"type": "text", "text": message.content}
        previous = out[-1] if out else None
        # Merge into the previous user turn only if it already carries tool results.
        if (
            previous is not None
            and previous["role"] == "user"
            and isinstance(previous["content"], list)
        ):
            previous["content"].append(block)
        elif message.role == "tool":
            out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": message.content})
    return out


def _parse_tool_use(index: int, block: dict[str, Any]) -> ToolCall:
    name = block.get("name")
    if not isinstance(name, str) or not name:
        raise LLMResponseError(f"tool_use block #{index} has no name: {block!r}")
    arguments = block.get("input")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise LLMResponseError(
            f"tool_use {name!r} input must be a JSON object, got {type(arguments).__name__}"
        )
    call_id = block.get("id")
    return ToolCall(
        id=call_id if isinstance(call_id, str) and call_id else f"toolu_{index}",
        name=name,
        arguments=arguments,
    )


def _parse_usage(raw: object) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    counts = {
        key: raw.get(key)
        for key in (
            "input_tokens",
            "output_tokens",
            "cache_creation_input_tokens",
            "cache_read_input_tokens",
        )
    }
    if not any(isinstance(value, int) for value in counts.values()):
        return None

    def count(key: str) -> int:
        value = counts[key]
        return value if isinstance(value, int) else 0

    # Prompt-cache tokens are input tokens too (Usage has no separate field);
    # cost is filled in by the pricing table (#12).
    return Usage(
        input_tokens=count("input_tokens")
        + count("cache_creation_input_tokens")
        + count("cache_read_input_tokens"),
        output_tokens=count("output_tokens"),
        cost_usd=None,
    )


def parse_response(data: dict[str, Any]) -> LLMResponse:
    """Normalise a Messages API JSON body into an :class:`LLMResponse`.

    ``text`` blocks are concatenated into ``content`` (``None`` if there are none);
    ``tool_use`` blocks become :class:`ToolCall` (missing ids synthesised as
    ``toolu_<index>``); other block types (e.g. ``thinking``) are ignored but kept
    in ``raw``. ``usage`` maps ``input_tokens`` (plus prompt-cache tokens) and
    ``output_tokens`` (``None`` if not reported).
    """
    blocks = data.get("content")
    if not isinstance(blocks, list):
        raise LLMResponseError(f"message has no content list: {json.dumps(data)[:300]}")
    texts: list[str] = []
    calls: list[ToolCall] = []
    for index, block in enumerate(blocks):
        if not isinstance(block, dict):
            raise LLMResponseError(f"content block #{index} is not an object: {block!r}")
        kind = block.get("type")
        if kind == "text":
            text = block.get("text")
            if isinstance(text, str):
                texts.append(text)
        elif kind == "tool_use":
            calls.append(_parse_tool_use(index, block))
    model = data.get("model")
    return LLMResponse(
        content="".join(texts) if texts else None,
        tool_calls=calls,
        usage=_parse_usage(data.get("usage")),
        raw=data,
        model=model if isinstance(model, str) else None,
    )
