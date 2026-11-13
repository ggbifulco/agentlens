"""``OpenAICompatClient``: one adapter for every OpenAI-compatible chat completions API.

OpenAI itself, Ollama, vLLM, LM Studio, llama.cpp server, Together, Groq... all
expose ``POST {base_url}/chat/completions``; only ``base_url`` (and maybe the API
key) changes::

    from agentlens.llm import LLMRequest, OpenAICompatClient

    openai = OpenAICompatClient(model="gpt-4o-mini")  # key from OPENAI_API_KEY
    ollama = OpenAICompatClient(model="llama3.1", base_url="http://localhost:11434/v1")
    response = ollama.complete(LLMRequest(messages="Say hi"))

Talks HTTP with ``httpx`` directly (no vendor SDK, ADR 0002). Retries, backoff and
error mapping live in :mod:`agentlens.llm._http`, shared with the Anthropic adapter.
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

__all__ = ["DEFAULT_OPENAI_BASE_URL", "OpenAICompatClient", "parse_response"]

DEFAULT_OPENAI_BASE_URL = "https://api.openai.com/v1"
"""The ``base_url`` of the official OpenAI API."""

_API_KEY_ENV = "OPENAI_API_KEY"


class OpenAICompatClient:
    """An :class:`~agentlens.llm.LLMClient` for OpenAI-compatible ``/chat/completions`` APIs.

    Args:
        model: Default model, used when a request has no ``model``. One of the two
            must be set.
        base_url: API root, e.g. ``http://localhost:11434/v1`` (Ollama),
            ``http://localhost:8000/v1`` (vLLM), ``http://localhost:1234/v1`` (LM Studio).
        api_key: Sent as ``Authorization: Bearer ...``. ``None`` falls back to the
            ``OPENAI_API_KEY`` environment variable; when neither is set no
            ``Authorization`` header is sent (fine for local servers). Pass ``""``
            to never send a key, even if the variable is set.
        timeout: Per-request timeout in seconds (or an ``httpx.Timeout``).
        max_retries: Retries on HTTP 408/429/5xx and transport errors, with
            exponential backoff (``Retry-After`` is honoured). Ignored when
            ``retry_policy`` is given.
        retry_policy: Full control over retries/backoff/jitter (see :class:`RetryPolicy`).
        headers: Extra HTTP headers sent with every request (e.g. ``OpenAI-Organization``).
        provider: Value of the ``provider`` attribute that
            :func:`agentlens.wrap_llm` records; defaults to ``"openai"`` for the
            official API and ``"openai_compat"`` for any other ``base_url``.
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
        base_url: str = DEFAULT_OPENAI_BASE_URL,
        api_key: str | None = None,
        timeout: float | httpx.Timeout = 60.0,
        max_retries: int = 2,
        retry_policy: RetryPolicy | None = None,
        headers: Mapping[str, str] | None = None,
        provider: str | None = None,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.model = model
        self.base_url = base_url.rstrip("/")
        self.provider = provider or (
            "openai" if self.base_url == DEFAULT_OPENAI_BASE_URL else "openai_compat"
        )
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
        """The chat completions endpoint."""
        return f"{self.base_url}/chat/completions"

    def __repr__(self) -> str:
        return f"{type(self).__name__}(model={self.model!r}, base_url={self.base_url!r})"

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Send ``request`` to ``/chat/completions`` and return the normalised response.

        Raises:
            LLMError: Subclasses for HTTP errors (:class:`LLMHTTPError`,
                :class:`LLMRateLimitError`), unreachable servers
                (:class:`LLMConnectionError`, :class:`LLMTimeoutError`) and
                unparsable answers (:class:`LLMResponseError`).
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
            raise LLMError("no model: set OpenAICompatClient(model=...) or LLMRequest.model")
        payload: dict[str, Any] = {
            "model": model,
            "messages": [_message_to_openai(m) for m in request.messages],
        }
        if request.temperature is not None:
            payload["temperature"] = request.temperature
        if request.max_tokens is not None:
            payload["max_tokens"] = request.max_tokens
        if request.tools:
            payload["tools"] = [_tool_to_openai(t) for t in request.tools]
        if request.response_format is not None:
            payload["response_format"] = request.response_format
        return payload

    def _request_headers(self) -> dict[str, str]:
        headers = dict(self._headers)
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
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


def _message_to_openai(message: Message) -> dict[str, Any]:
    out: dict[str, Any] = {"role": message.role, "content": message.content}
    if message.tool_calls:
        out["tool_calls"] = [
            {
                "id": call.id,
                "type": "function",
                "function": {"name": call.name, "arguments": json.dumps(call.arguments)},
            }
            for call in message.tool_calls
        ]
    if message.tool_call_id is not None:
        out["tool_call_id"] = message.tool_call_id
    return out


def _tool_to_openai(tool: ToolSpec) -> dict[str, Any]:
    function: dict[str, Any] = {"name": tool.name, "parameters": tool.parameters}
    if tool.description:
        function["description"] = tool.description
    return {"type": "function", "function": function}


def _parse_arguments(name: str, arguments: object) -> dict[str, JsonValue]:
    if arguments is None or arguments == "":
        return {}
    if isinstance(arguments, str):
        try:
            arguments = json.loads(arguments)
        except ValueError as exc:
            raise LLMResponseError(
                f"tool call {name!r} has arguments that are not valid JSON: {arguments!r}"
            ) from exc
    if not isinstance(arguments, dict):
        raise LLMResponseError(
            f"tool call {name!r} arguments must be a JSON object, got {type(arguments).__name__}"
        )
    return arguments


def _parse_tool_calls(raw_calls: object) -> list[ToolCall]:
    if not raw_calls:
        return []
    if not isinstance(raw_calls, list):
        raise LLMResponseError("message.tool_calls must be a list")
    calls: list[ToolCall] = []
    for index, raw in enumerate(raw_calls):
        function = raw.get("function") if isinstance(raw, dict) else None
        if not isinstance(function, dict) or not isinstance(function.get("name"), str):
            raise LLMResponseError(f"tool call #{index} has no function name: {raw!r}")
        name = function["name"]
        call_id = raw.get("id")
        calls.append(
            ToolCall(
                # Some local servers omit ids; tool messages still need one to answer.
                id=call_id if isinstance(call_id, str) and call_id else f"call_{index}",
                name=name,
                arguments=_parse_arguments(name, function.get("arguments")),
            )
        )
    return calls


def _parse_usage(raw: object) -> Usage | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    if not isinstance(prompt, int) and not isinstance(completion, int):
        return None
    # Cost is filled in by the pricing table (#12); unknown here.
    return Usage(
        input_tokens=prompt if isinstance(prompt, int) else 0,
        output_tokens=completion if isinstance(completion, int) else 0,
        cost_usd=None,
    )


def parse_response(data: dict[str, Any]) -> LLMResponse:
    """Normalise a ``/chat/completions`` JSON body into an :class:`LLMResponse`.

    Uses the first choice; tool-call arguments are decoded into dicts; ``usage``
    maps ``prompt_tokens``/``completion_tokens`` (``None`` if not reported).
    """
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        raise LLMResponseError(f"chat completion has no choices: {json.dumps(data)[:300]}")
    message = choices[0].get("message")
    if not isinstance(message, dict):
        raise LLMResponseError("chat completion choice has no message")
    content = message.get("content")
    model = data.get("model")
    return LLMResponse(
        content=content if isinstance(content, str) else None,
        tool_calls=_parse_tool_calls(message.get("tool_calls")),
        usage=_parse_usage(data.get("usage")),
        raw=data,
        model=model if isinstance(model, str) else None,
    )
