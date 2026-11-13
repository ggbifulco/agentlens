"""Masking secrets before a trace is persisted (NFR-5, ``docs/design.md`` §2).

A :data:`Redactor` is any callable that takes a JSON value and returns a JSON
value with secrets masked. :func:`agentlens.run` applies one to every payload
of the finished trace (trace input/output/metadata/error, span
input/output/attributes/error) *before* handing the trace to its sink, so
secrets that leak into tool arguments or LLM payloads never reach the store.

The default, :data:`DEFAULT_REDACTOR`, is a :class:`PatternRedactor` that

* replaces the whole value of a mapping entry whose key looks sensitive
  (``api_key``, ``Authorization``, ``password``, ``client_secret``, ...), and
* masks well-known credential formats inside any string (OpenAI/Anthropic
  ``sk-...`` keys, ``Bearer`` tokens, AWS access key ids, GitHub tokens, ...).

Pattern-based redaction is a safety net, not a guarantee: keep secrets in
environment variables and out of agent inputs in the first place.

Use it standalone on traces recorded elsewhere::

    from agentlens.tracing import redact_trace

    store.save_trace(redact_trace(trace))
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable

from pydantic import JsonValue

from agentlens.schema import Span, Trace
from agentlens.tracing._serialize import to_json_value

__all__ = [
    "DEFAULT_PATTERNS",
    "DEFAULT_REDACTOR",
    "DEFAULT_SENSITIVE_KEYS",
    "REDACTED",
    "PatternRedactor",
    "Redactor",
    "redact_trace",
]

REDACTED = "[REDACTED]"
"""The default replacement for masked secrets."""

Redactor = Callable[[JsonValue], JsonValue]
"""Takes a JSON payload and returns it with secrets masked. Must not mutate its input."""

DEFAULT_PATTERNS: tuple[str, ...] = (
    # PEM private key blocks.
    r"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----",
    # Anthropic (sk-ant-...) and OpenAI (sk-..., sk-proj-...) style API keys.
    r"\bsk-[A-Za-z0-9][A-Za-z0-9_\-]{15,}",
    # HTTP bearer tokens: keep the scheme, mask the credential.
    r"(?i)\bBearer\s+(?P<secret>[A-Za-z0-9\-._~+/]{8,}=*)",
    # AWS access key ids.
    r"\b(?:AKIA|ASIA)[0-9A-Z]{16}\b",
    # GitHub personal/OAuth/app tokens and fine-grained PATs.
    r"\bgh[pousr]_[A-Za-z0-9]{36,}\b",
    r"\bgithub_pat_[A-Za-z0-9_]{22,}\b",
    # Google API keys.
    r"\bAIza[0-9A-Za-z\-_]{35}\b",
    # Slack tokens.
    r"\bxox[abprs]-[A-Za-z0-9\-]{10,}",
    # JSON Web Tokens.
    r"\beyJ[A-Za-z0-9_\-]{5,}\.eyJ[A-Za-z0-9_\-]{5,}\.[A-Za-z0-9_\-]+",
    # ``api_key=...`` / ``password: ...`` assignments inside free text.
    r"(?i)\b[\w-]*(?:api[_-]?key|secret|password|passwd|token)[\"']?\s*[:=]\s*[\"']?"
    r"(?P<secret>[^\s\"',;&]{8,})",
)
"""Regular expressions for secrets inside strings. When a pattern has a group
named ``secret`` only that group is masked (``Bearer [REDACTED]``), otherwise
the whole match is."""

DEFAULT_SENSITIVE_KEYS: frozenset[str] = frozenset(
    {
        "api_key",
        "apikey",
        "x_api_key",
        "authorization",
        "proxy_authorization",
        "cookie",
        "set_cookie",
        "password",
        "passwd",
        "secret",
        "client_secret",
        "secret_key",
        "private_key",
        "access_token",
        "refresh_token",
        "id_token",
        "auth_token",
        "token",
        "aws_secret_access_key",
        "aws_session_token",
    }
)
"""Mapping keys (compared case-insensitively, ``-`` treated as ``_``) whose
values are replaced wholesale, whatever their type."""

_SENSITIVE_SUFFIXES = ("_api_key", "_apikey", "_secret", "_password", "_secret_key")


class PatternRedactor:
    """Redactor masking sensitive mapping keys and secret-looking substrings.

    It walks nested lists and dicts and returns a new value; the input is never
    mutated.

    .. code-block:: python

        redactor = PatternRedactor(
            patterns=(*DEFAULT_PATTERNS, r"\\bINTERNAL-[0-9]{6}\\b"),
            keys=DEFAULT_SENSITIVE_KEYS | {"session_id"},
        )
        agentlens.run("agent", sink=store.save_trace, redactor=redactor)

    Args:
        patterns: Regular expressions matched inside every string. A group
            named ``secret`` limits masking to that group.
        keys: Mapping keys whose values are always masked. Keys ending in
            ``_api_key``, ``_secret`` or ``_password`` are masked as well.
        replacement: The text that replaces each secret.
    """

    def __init__(
        self,
        patterns: Iterable[str | re.Pattern[str]] = DEFAULT_PATTERNS,
        *,
        keys: Iterable[str] = DEFAULT_SENSITIVE_KEYS,
        replacement: str = REDACTED,
    ) -> None:
        self.patterns: tuple[re.Pattern[str], ...] = tuple(re.compile(p) for p in patterns)
        self.keys: frozenset[str] = frozenset(_normalise_key(k) for k in keys)
        self.replacement = replacement

    def __call__(self, value: JsonValue) -> JsonValue:
        """Return ``value`` with secrets masked."""
        if isinstance(value, str):
            return self.redact_text(value)
        if isinstance(value, list):
            return [self(item) for item in value]
        if isinstance(value, dict):
            return {
                key: self.replacement if self.is_sensitive_key(key) else self(item)
                for key, item in value.items()
            }
        return value

    def redact_text(self, text: str) -> str:
        """Mask every pattern match in ``text``."""
        for pattern in self.patterns:
            text = pattern.sub(self._replace, text)
        return text

    def is_sensitive_key(self, key: str) -> bool:
        """Whether the value stored under ``key`` must be masked entirely."""
        normalised = _normalise_key(key)
        return normalised in self.keys or normalised.endswith(_SENSITIVE_SUFFIXES)

    def _replace(self, match: re.Match[str]) -> str:
        if "secret" not in match.re.groupindex or match.group("secret") is None:
            return self.replacement
        start, end = match.span("secret")
        whole_start = match.start()
        text = match.group(0)
        return text[: start - whole_start] + self.replacement + text[end - whole_start :]


def _normalise_key(key: str) -> str:
    return key.strip().lower().replace("-", "_")


DEFAULT_REDACTOR: Redactor = PatternRedactor()
"""The redactor :func:`agentlens.run` uses unless told otherwise."""


def redact_trace(trace: Trace, redactor: Redactor | None = None) -> Trace:
    """Return a copy of ``trace`` with every payload passed through ``redactor``.

    Redacted fields: the trace's ``input``, ``output``, ``metadata`` and
    ``error`` and each span's ``input``, ``output``, ``attributes`` and
    ``error``. Ids, names, timings and usage are kept as they are. The original
    trace is not modified.

    Args:
        trace: The trace to redact.
        redactor: Defaults to :data:`DEFAULT_REDACTOR`.

    Raises:
        Exception: whatever ``redactor`` raises; :func:`agentlens.run` catches
            it and drops the trace rather than persisting unredacted data.
    """
    redact = redactor if redactor is not None else DEFAULT_REDACTOR

    def payload(value: JsonValue) -> JsonValue:
        # Custom redactors may return anything; keep the copy JSON-only.
        return to_json_value(redact(value))

    def mapping(values: dict[str, JsonValue]) -> dict[str, JsonValue]:
        result = payload(values)
        return result if isinstance(result, dict) else {"redacted": result}

    def error(value: str | None) -> str | None:
        if value is None:
            return None
        result = redact(value)
        return result if isinstance(result, str) else str(result)

    def span(s: Span) -> Span:
        return s.model_copy(
            update={
                "input": payload(s.input),
                "output": payload(s.output),
                "attributes": mapping(s.attributes),
                "error": error(s.error),
            }
        )

    return trace.model_copy(
        update={
            "input": payload(trace.input),
            "output": payload(trace.output),
            "metadata": mapping(trace.metadata),
            "error": error(trace.error),
            "spans": [span(s) for s in trace.spans],
        }
    )
