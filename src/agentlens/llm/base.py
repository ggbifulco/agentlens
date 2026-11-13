"""The :class:`LLMClient` protocol every adapter implements (ADR 0002)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

from agentlens.llm.types import LLMRequest, LLMResponse

__all__ = ["LLMClient"]


@runtime_checkable
class LLMClient(Protocol):
    """A minimal, provider-neutral LLM client.

    Implementations translate :class:`LLMRequest` to a provider call and the
    provider's answer back to :class:`LLMResponse`. The protocol is structural:
    any object with these two methods is a client, no subclassing needed.
    """

    def complete(self, request: LLMRequest) -> LLMResponse:
        """Run ``request`` and block until the response is available."""
        ...

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        """Async variant of :meth:`complete`."""
        ...
