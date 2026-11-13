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

:func:`get_client` builds any of them from a ``"provider:model"`` spec (or
``$AGENTLENS_MODEL``), with ``usage.cost_usd`` filled from the :class:`Pricing` table::

    from agentlens.llm import get_client

    llm = get_client("ollama:llama3.1")  # or "openai:gpt-4o-mini", "anthropic:...", "mock"
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
    LLMConfigError,
    LLMConnectionError,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMResponseError,
    LLMTimeoutError,
)
from agentlens.llm.factory import MODEL_ENV, PROVIDERS, get_client, parse_spec
from agentlens.llm.mock import (
    MockExhaustedError,
    MockLLM,
    MockLLMError,
    MockNoMatchError,
    ScriptItem,
    count_tokens,
)
from agentlens.llm.openai_compat import DEFAULT_OPENAI_BASE_URL, OpenAICompatClient
from agentlens.llm.pricing import PRICING_ENV, ModelPrice, PricedClient, Pricing
from agentlens.llm.types import LLMRequest, LLMResponse, Message, Role, ToolCall, ToolSpec

__all__ = [
    "ANTHROPIC_VERSION",
    "DEFAULT_ANTHROPIC_BASE_URL",
    "DEFAULT_MAX_TOKENS",
    "DEFAULT_OPENAI_BASE_URL",
    "MODEL_ENV",
    "PRICING_ENV",
    "PROVIDERS",
    "AnthropicClient",
    "LLMClient",
    "LLMConfigError",
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
    "ModelPrice",
    "OpenAICompatClient",
    "PricedClient",
    "Pricing",
    "RetryPolicy",
    "Role",
    "ScriptItem",
    "ToolCall",
    "ToolSpec",
    "count_tokens",
    "get_client",
    "parse_spec",
]
