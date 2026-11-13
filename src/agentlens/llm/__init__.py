"""Provider-neutral LLM access (``docs/design.md`` §3, ADR 0002).

One small :class:`LLMClient` protocol plus request/response models, so judges,
example agents and replay can swap providers (mock, OpenAI-compatible,
Anthropic) without code changes::

    from agentlens.llm import LLMRequest, Message, MockLLM

    llm = MockLLM(["Hi! How can I help?"])
    response = llm.complete(LLMRequest(messages=[Message.user("Hello")]))
    assert response.content == "Hi! How can I help?"
"""

from agentlens.llm.base import LLMClient
from agentlens.llm.mock import (
    MockExhaustedError,
    MockLLM,
    MockLLMError,
    MockNoMatchError,
    ScriptItem,
    count_tokens,
)
from agentlens.llm.types import LLMRequest, LLMResponse, Message, Role, ToolCall, ToolSpec

__all__ = [
    "LLMClient",
    "LLMRequest",
    "LLMResponse",
    "Message",
    "MockExhaustedError",
    "MockLLM",
    "MockLLMError",
    "MockNoMatchError",
    "Role",
    "ScriptItem",
    "ToolCall",
    "ToolSpec",
    "count_tokens",
]
