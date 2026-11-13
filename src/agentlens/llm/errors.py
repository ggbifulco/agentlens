"""Errors raised by the HTTP LLM adapters (OpenAI-compatible, Anthropic).

Everything derives from :class:`LLMError`, so callers can catch one type::

    try:
        response = client.complete(request)
    except LLMRateLimitError:
        ...  # still rate limited after all retries
    except LLMError:
        ...  # any other provider/transport problem

Messages never contain the API key or request headers.
"""

from __future__ import annotations

__all__ = [
    "LLMConfigError",
    "LLMConnectionError",
    "LLMError",
    "LLMHTTPError",
    "LLMRateLimitError",
    "LLMResponseError",
    "LLMTimeoutError",
]


class LLMError(RuntimeError):
    """Base class for failures of an LLM adapter call."""


class LLMConfigError(LLMError, ValueError):
    """Invalid client configuration: a bad ``get_client`` spec or pricing file.

    Also a :class:`ValueError`, since it is raised for bad input before any call.
    """


class LLMConnectionError(LLMError):
    """The provider could not be reached (DNS, refused connection, broken stream...).

    Raised after the retries are exhausted; the underlying ``httpx`` error is the
    ``__cause__``.
    """


class LLMTimeoutError(LLMConnectionError):
    """The request timed out (connect, read, write or pool timeout)."""


class LLMHTTPError(LLMError):
    """The provider answered with a non-success HTTP status.

    Attributes:
        status_code: The HTTP status.
        body: The response body (text, truncated to a few KB) for debugging.
    """

    def __init__(self, message: str, *, status_code: int, body: str = "") -> None:
        super().__init__(message)
        self.status_code = status_code
        self.body = body


class LLMRateLimitError(LLMHTTPError):
    """HTTP 429: still rate limited after all retries."""


class LLMResponseError(LLMError):
    """A success response that cannot be understood (not JSON, no choices, bad tool arguments)."""
