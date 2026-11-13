"""Shared HTTP plumbing for the LLM adapters: JSON POST with retries and backoff.

Private module: the OpenAI-compatible adapter (and the Anthropic one) build on
:class:`JSONHTTPClient`; only :class:`RetryPolicy` is public API (re-exported from
:mod:`agentlens.llm`).

Retry rules (see ``docs/design.md`` §3):

* retried: HTTP 408/429/5xx, timeouts, network and remote-protocol errors;
* not retried: every other 4xx, malformed URLs/unsupported protocols;
* delay: ``Retry-After`` / ``retry-after-ms`` when the server sends one (capped at
  :attr:`RetryPolicy.max_retry_after`), else exponential backoff
  ``initial_delay * 2**n`` capped at ``max_delay``, passed through ``jitter``.

Request headers (and so the API key) are never logged nor copied into errors.
"""

from __future__ import annotations

import asyncio
import email.utils
import json
import logging
import random
import threading
import time
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass
from typing import Any

import httpx

from agentlens.llm.errors import (
    LLMConnectionError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)

__all__ = [
    "JSONHTTPClient",
    "RetryPolicy",
    "equal_jitter",
    "is_retryable_status",
    "parse_retry_after",
]

logger = logging.getLogger("agentlens.llm")

_RETRYABLE_TRANSPORT_ERRORS = (
    httpx.TimeoutException,
    httpx.NetworkError,
    httpx.RemoteProtocolError,
)
_MAX_ERROR_BODY = 4000
_MAX_ERROR_DETAIL = 500


def equal_jitter(delay: float) -> float:
    """Default jitter: a random delay in ``[delay / 2, delay]``.

    Spreads out clients that failed at the same moment while keeping at least
    half of the backoff.
    """
    return delay / 2 + random.uniform(0, delay / 2)


@dataclass(frozen=True, kw_only=True)
class RetryPolicy:
    """How an adapter retries failed requests.

    Args:
        max_retries: Retries after the first attempt (``0`` disables retrying).
        initial_delay: Backoff before the first retry, in seconds.
        max_delay: Upper bound of the exponential backoff, in seconds.
        max_retry_after: Upper bound applied to a server's ``Retry-After``, in seconds.
        jitter: Maps the computed backoff to the actual delay; ``lambda d: d``
            disables jitter (handy in tests). Not applied to ``Retry-After``.
        sleep: Blocking sleep used by sync calls (inject a fake in tests).
        async_sleep: Sleep used by async calls.
    """

    max_retries: int = 2
    initial_delay: float = 0.5
    max_delay: float = 8.0
    max_retry_after: float = 60.0
    jitter: Callable[[float], float] = equal_jitter
    sleep: Callable[[float], None] = time.sleep
    async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep

    def __post_init__(self) -> None:
        if self.max_retries < 0:
            raise ValueError("max_retries must be >= 0")
        if min(self.initial_delay, self.max_delay, self.max_retry_after) < 0:
            raise ValueError("delays must be >= 0")

    def backoff(self, retry_index: int) -> float:
        """Jittered exponential backoff before retry number ``retry_index`` (0-based)."""
        return self.jitter(min(self.max_delay, self.initial_delay * 2**retry_index))

    def delay(self, retry_index: int, retry_after: float | None = None) -> float:
        """Delay before retry ``retry_index``: the server's hint if any, else :meth:`backoff`."""
        if retry_after is not None:
            return min(retry_after, self.max_retry_after)
        return self.backoff(retry_index)


def is_retryable_status(status_code: int) -> bool:
    """``True`` for statuses worth retrying: 408 (timeout), 429 (rate limit) and 5xx."""
    return status_code in (408, 429) or status_code >= 500


def parse_retry_after(headers: Mapping[str, str], *, now: float | None = None) -> float | None:
    """Seconds to wait according to ``retry-after-ms`` or ``Retry-After``, else ``None``.

    ``Retry-After`` may be a number of seconds or an HTTP date (relative to ``now``,
    default: the current time). Negative or unparsable values are ignored.
    """
    lowered = {key.lower(): value for key, value in headers.items()}
    millis = lowered.get("retry-after-ms")
    if millis is not None:
        try:
            value = float(millis) / 1000
        except ValueError:
            value = -1.0
        if value >= 0:
            return value
    header = lowered.get("retry-after")
    if header is None:
        return None
    try:
        seconds = float(header)
    except ValueError:
        try:
            when = email.utils.parsedate_to_datetime(header)
        except (TypeError, ValueError):
            return None
        # A date in the past means "retry now".
        return max(0.0, when.timestamp() - (time.time() if now is None else now))
    return seconds if seconds >= 0 else None


def _safe_url(request: httpx.Request) -> str:
    return str(request.url.copy_with(query=None))


def _error_detail(response: httpx.Response) -> str:
    """The provider's error message (``{"error": {"message": ...}}`` and variants) or the body."""
    try:
        payload = response.json()
    except ValueError:
        payload = None
    if isinstance(payload, dict):
        error = payload.get("error", payload)
        if isinstance(error, dict):
            message = error.get("message")
            if isinstance(message, str):
                return message
        elif isinstance(error, str):
            return error
    return response.text[:_MAX_ERROR_DETAIL] or response.reason_phrase


def _http_error(response: httpx.Response) -> LLMHTTPError:
    status = response.status_code
    message = (
        f"POST {_safe_url(response.request)} failed with HTTP {status}: {_error_detail(response)}"
    )
    cls = LLMRateLimitError if status == 429 else LLMHTTPError
    return cls(message, status_code=status, body=response.text[:_MAX_ERROR_BODY])


def _transport_error(exc: httpx.TransportError, url: str) -> LLMConnectionError:
    cls = LLMTimeoutError if isinstance(exc, httpx.TimeoutException) else LLMConnectionError
    return cls(f"POST {url} failed: {type(exc).__name__}: {exc}")


def _decode(response: httpx.Response) -> dict[str, Any]:
    try:
        payload = response.json()
    except (ValueError, UnicodeDecodeError) as exc:
        raise LLMResponseError(
            f"POST {_safe_url(response.request)} returned a non-JSON body: {response.text[:200]!r}"
        ) from exc
    if not isinstance(payload, dict):
        raise LLMResponseError(
            f"POST {_safe_url(response.request)} returned JSON {type(payload).__name__}, "
            "expected an object"
        )
    return payload


class JSONHTTPClient:
    """POSTs JSON payloads and returns the decoded JSON object, retrying per a :class:`RetryPolicy`.

    The sync ``httpx.Client`` is created lazily and reused (connection pooling)
    unless one is injected; async calls use the injected ``httpx.AsyncClient`` or a
    short-lived one per call (safe across event loops, e.g. repeated ``asyncio.run``).
    Injected clients are never closed by this object.
    """

    def __init__(
        self,
        *,
        timeout: float | httpx.Timeout,
        retry: RetryPolicy,
        http_client: httpx.Client | None = None,
        async_http_client: httpx.AsyncClient | None = None,
    ) -> None:
        self.timeout = timeout
        self.retry = retry
        self._client = http_client
        self._owns_client = http_client is None
        self._async_client = async_http_client
        self._lock = threading.Lock()

    def _sync_client(self) -> httpx.Client:
        with self._lock:
            if self._client is None:
                self._client = httpx.Client(timeout=self.timeout)
            return self._client

    def close(self) -> None:
        """Close the sync connection pool created by this object (no-op otherwise)."""
        with self._lock:
            if self._owns_client and self._client is not None:
                self._client.close()
                self._client = None

    def _next_delay(
        self,
        attempt: int,
        url: str,
        response: httpx.Response | None,
        error: httpx.TransportError | None,
    ) -> float:
        """Delay before retrying ``attempt``, or raise the final :class:`LLMError`."""
        final = attempt >= self.retry.max_retries
        if response is not None:
            if final or not is_retryable_status(response.status_code):
                raise _http_error(response)
            delay = self.retry.delay(attempt, parse_retry_after(response.headers))
            reason = f"HTTP {response.status_code}"
        else:
            assert error is not None
            if final or not isinstance(error, _RETRYABLE_TRANSPORT_ERRORS):
                raise _transport_error(error, url) from error
            delay = self.retry.delay(attempt)
            reason = type(error).__name__
        logger.debug(
            "retrying POST %s in %.2fs (retry %d/%d) after %s",
            url,
            delay,
            attempt + 1,
            self.retry.max_retries,
            reason,
        )
        return delay

    def post_json(
        self, url: str, payload: Mapping[str, Any], headers: Mapping[str, str]
    ) -> dict[str, Any]:
        """POST ``payload`` as JSON to ``url`` and return the JSON object answered."""
        client = self._sync_client()
        content = json.dumps(payload).encode()
        attempt = 0
        while True:
            try:
                response = client.post(
                    url, content=content, headers=_json_headers(headers), timeout=self.timeout
                )
            except httpx.TransportError as exc:
                delay = self._next_delay(attempt, url, None, exc)
            else:
                if response.is_success:
                    return _decode(response)
                delay = self._next_delay(attempt, url, response, None)
            self.retry.sleep(delay)
            attempt += 1

    async def apost_json(
        self, url: str, payload: Mapping[str, Any], headers: Mapping[str, str]
    ) -> dict[str, Any]:
        """Async variant of :meth:`post_json`."""
        if self._async_client is not None:
            return await self._apost(self._async_client, url, payload, headers)
        async with httpx.AsyncClient(timeout=self.timeout) as client:
            return await self._apost(client, url, payload, headers)

    async def _apost(
        self,
        client: httpx.AsyncClient,
        url: str,
        payload: Mapping[str, Any],
        headers: Mapping[str, str],
    ) -> dict[str, Any]:
        content = json.dumps(payload).encode()
        attempt = 0
        while True:
            try:
                response = await client.post(
                    url, content=content, headers=_json_headers(headers), timeout=self.timeout
                )
            except httpx.TransportError as exc:
                delay = self._next_delay(attempt, url, None, exc)
            else:
                if response.is_success:
                    return _decode(response)
                delay = self._next_delay(attempt, url, response, None)
            await self.retry.async_sleep(delay)
            attempt += 1


def _json_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {"Content-Type": "application/json", "Accept": "application/json", **headers}
