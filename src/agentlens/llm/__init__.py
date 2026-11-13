"""Provider-neutral LLM access (``docs/design.md`` §3, ADR 0002).

One small :class:`LLMClient` protocol plus request/response models, so judges,
example agents and replay can swap providers (mock, OpenAI-compatible,
Anthropic) without code changes::

    from agentlens.llm import LLMRequest, Message, MockLLM

    llm = MockLLM(["Hi! How can I help?"])
    response = llm.complete(LLMRequest(messages=[Message.user("Hello")]))
    assert response.content == "Hi! How can I help?"

Real providers: :class:`OpenAICompatClient` (OpenAI, Ollama, vLLM, LM Studio via
``base_url``) and :class:`AnthropicClient` (Anthropic Messages API).
Adapter failures raise subclasses of :class:`LLMError`.
"""

from agentlens.llm._http import RetryPolicy
from agentlens.llm.anthropic import (
    ANTHROPIC_VERSION,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_MAX_TOKENS,
    AnthropicClient,
)
from agentlens.llm.base import LLMClient
from agentlens.llm.errors import (
    LLMConnectionError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from agentlens.llm.mock import (
    MockExhaustedError,
    MockLLM,
    MockLLMError,
    MockNoMatchError,
    ScriptItem,
    count_tokens,
)
from agentlens.llm.openai_compat import DEFAULT_OPENAI_BASE_URL, OpenAICompatClient
from agentlens.llm.types import LLMRequest, LLMResponse, Message, Role, ToolCall, ToolSpec

__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OPENAI_BASE_URL",
    "AnthropicClient",
    "LLMClient",
    "LLMConnectionError",
    "LLMError",
    "LLMHTTPError",
    "LLMRateLimitError",
    "LLMRequest",
    "LLMResponse",
    "LLMResponseError",
    "LLMTimeoutError",
    "Message",
    "MockExhaustedError",
    "MockLLM",
    "MockLLMError",
    "MockNoMatchError",
    "OpenAICompatClient",
    "RetryPolicy",
    "Role",
    "ScriptItem",
    "ToolCall",
    "ToolSpec",
    "count_tokens",
]
