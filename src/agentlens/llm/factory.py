"""``get_client``: build an :class:`~agentlens.llm.LLMClient` from a spec string (FR-M3).

The provider and model are configuration, not code::

    from agentlens.llm import get_client

    llm = get_client("openai:gpt-4o-mini")        # key from OPENAI_API_KEY
    llm = get_client("ollama:llama3.1")           # $OLLAMA_BASE_URL or localhost:11434
    llm = get_client("anthropic:claude-3-5-haiku-latest")
    llm = get_client("mock")                      # echoes the last user message
    llm = get_client()                            # spec from $AGENTLENS_MODEL

A spec is ``"<provider>:<model>"``; only the first ``:`` separates them, so Ollama tags
work (``"ollama:llama3.1:8b"``). Keyword ``overrides`` are passed to the client
constructor and win over environment variables (``get_client("ollama:llama3.1",
base_url="http://gpu-box:11434/v1", timeout=120)``).

HTTP clients come wrapped in a :class:`~agentlens.llm.PricedClient`, so responses
carry ``usage.cost_usd`` from the pricing table (``None`` for unknown models).
"""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from agentlens.llm.anthropic import AnthropicClient
from agentlens.llm.base import LLMClient
from agentlens.llm.errors import LLMConfigError
from agentlens.llm.mock import MockLLM
from agentlens.llm.openai_compat import DEFAULT_OPENAI_BASE_URL, OpenAICompatClient
from agentlens.llm.pricing import PricedClient, Pricing
from agentlens.llm.types import LLMRequest

__all__ = ["MODEL_ENV", "PROVIDERS", "get_client", "parse_spec"]

MODEL_ENV = "AGENTLENS_MODEL"
"""Environment variable read by :func:`get_client` when no spec is given."""

_Builder = Callable[[str, dict[str, Any]], LLMClient]

_OLLAMA_DEFAULT = "http://localhost:11434/v1"
_VLLM_DEFAULT = "http://localhost:8000/v1"
_LMSTUDIO_DEFAULT = "http://localhost:1234/v1"


def _env(name: str) -> str | None:
    return os.environ.get(name) or None


def _openai(model: str, overrides: dict[str, Any]) -> LLMClient:
    kwargs: dict[str, Any] = {"base_url": _env("OPENAI_BASE_URL") or DEFAULT_OPENAI_BASE_URL}
    # api_key: OpenAICompatClient falls back to OPENAI_API_KEY itself.
    return OpenAICompatClient(model, **{**kwargs, **overrides})


def _local(provider: str, base_env: str, default_url: str, key_env: str | None) -> _Builder:
    def build(model: str, overrides: dict[str, Any]) -> LLMClient:
        kwargs: dict[str, Any] = {
            "base_url": _env(base_env) or default_url,
            # Never forward OPENAI_API_KEY to a local server.
            "api_key": (_env(key_env) if key_env else None) or "",
            "provider": provider,
        }
        return OpenAICompatClient(model, **{**kwargs, **overrides})

    return build


def _openai_compat(model: str, overrides: dict[str, Any]) -> LLMClient:
    kwargs: dict[str, Any] = {
        "api_key": _env("OPENAI_COMPAT_API_KEY") or "",
        "provider": "openai_compat",
    }
    base_url = overrides.get("base_url") or _env("OPENAI_COMPAT_BASE_URL")
    if not base_url:
        raise LLMConfigError(
            "openai-compat needs a base_url: set OPENAI_COMPAT_BASE_URL or pass base_url=..."
        )
    kwargs["base_url"] = base_url
    return OpenAICompatClient(model, **{**kwargs, **overrides})


def _anthropic(model: str, overrides: dict[str, Any]) -> LLMClient:
    kwargs: dict[str, Any] = {}
    base_url = _env("ANTHROPIC_BASE_URL")
    if base_url:
        kwargs["base_url"] = base_url
    # api_key: AnthropicClient falls back to ANTHROPIC_API_KEY itself.
    return AnthropicClient(model, **{**kwargs, **overrides})


def _echo(request: LLMRequest) -> str:
    return request.last_user_message or ""


PROVIDERS: dict[str, _Builder] = {
    "openai": _openai,
    "anthropic": _anthropic,
    "ollama": _local("ollama", "OLLAMA_BASE_URL", _OLLAMA_DEFAULT, None),
    "vllm": _local("vllm", "VLLM_BASE_URL", _VLLM_DEFAULT, "VLLM_API_KEY"),
    "lmstudio": _local("lmstudio", "LMSTUDIO_BASE_URL", _LMSTUDIO_DEFAULT, None),
    "openai-compat": _openai_compat,
}
"""Provider name -> builder ``(model, overrides) -> LLMClient``. ``mock`` is handled
separately (it needs no model and is not priced)."""

_ALIASES = {"openai_compat": "openai-compat", "lm-studio": "lmstudio", "lm_studio": "lmstudio"}

_KNOWN = ", ".join(sorted([*PROVIDERS, "mock"]))


def parse_spec(spec: str) -> tuple[str, str]:
    """Split ``"provider:model"`` into a normalised ``(provider, model)``.

    The provider is lower-cased and aliases resolved (``openai_compat`` ->
    ``openai-compat``, ``lm-studio`` -> ``lmstudio``); ``"mock"`` alone means
    ``("mock", "mock")``.

    Raises:
        LLMConfigError: Empty spec, missing ``provider:`` / model, or unknown provider.
    """
    text = spec.strip()
    if not text:
        raise LLMConfigError(f"empty model spec: use '<provider>:<model>' (providers: {_KNOWN})")
    provider, sep, model = text.partition(":")
    provider = provider.strip().lower()
    provider = _ALIASES.get(provider, provider)
    model = model.strip()
    if provider == "mock":
        return provider, model or "mock"
    if provider not in PROVIDERS:
        hint = "" if sep else " (expected '<provider>:<model>', e.g. 'openai:gpt-4o-mini')"
        raise LLMConfigError(
            f"unknown LLM provider {provider!r} in spec {spec!r}{hint}; known: {_KNOWN}"
        )
    if not model:
        raise LLMConfigError(f"model spec {spec!r} has no model: use '{provider}:<model>'")
    return provider, model


def get_client(
    spec: str | None = None,
    *,
    pricing: Pricing | str | os.PathLike[str] | None = None,
    **overrides: Any,
) -> LLMClient:
    """Build an LLM client from ``"<provider>:<model>"`` (default: ``$AGENTLENS_MODEL``).

    Providers and the environment variables they read (``overrides`` win):

    * ``openai:<model>`` — :class:`OpenAICompatClient`; ``OPENAI_API_KEY``,
      ``OPENAI_BASE_URL`` (default ``https://api.openai.com/v1``).
    * ``anthropic:<model>`` — :class:`AnthropicClient`; ``ANTHROPIC_API_KEY``,
      ``ANTHROPIC_BASE_URL``.
    * ``ollama:<model>`` — ``OLLAMA_BASE_URL`` (default ``http://localhost:11434/v1``), no key.
    * ``vllm:<model>`` — ``VLLM_BASE_URL`` (default ``http://localhost:8000/v1``),
      optional ``VLLM_API_KEY``.
    * ``lmstudio:<model>`` — ``LMSTUDIO_BASE_URL`` (default ``http://localhost:1234/v1``).
    * ``openai-compat:<model>`` — any other compatible server: ``OPENAI_COMPAT_BASE_URL``
      (required, or ``base_url=``), optional ``OPENAI_COMPAT_API_KEY``.
    * ``mock`` / ``mock:<name>`` — :class:`MockLLM` reporting model ``<name>`` (default
      ``"mock"``) that echoes the last user message; pass ``script=`` (and ``cycle=``,
      ``default=``) to script it. Costs 0 and needs no network.

    Local servers never receive ``OPENAI_API_KEY``. Every client except the mock is
    wrapped in a :class:`PricedClient`, which fills ``usage.cost_usd`` from ``pricing``:
    a :class:`Pricing`, a YAML file merged over the packaged table, or ``None`` for
    :meth:`Pricing.load` (packaged table + ``$AGENTLENS_PRICING``).

    Raises:
        LLMConfigError: No spec and no ``AGENTLENS_MODEL``, an invalid spec, an unknown
            provider, a missing required setting or an invalid pricing file.
    """
    if spec is None:
        spec = _env(MODEL_ENV)
        if spec is None:
            raise LLMConfigError(
                f"no model configured: pass get_client('<provider>:<model>') "
                f"(e.g. 'openai:gpt-4o-mini') or set {MODEL_ENV}; providers: {_KNOWN}"
            )
    provider, model = parse_spec(spec)
    if provider == "mock":
        options = {"script": _echo, "model": model, **overrides}
        return MockLLM(**options)
    table = pricing if isinstance(pricing, Pricing) else Pricing.load(pricing)
    # PricedClient reads the client's ``provider`` (e.g. "ollama" -> free).
    return PricedClient(PROVIDERS[provider](model, dict(overrides)), table)
