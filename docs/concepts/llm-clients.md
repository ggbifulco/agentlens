# LLM clients

Judges, the example agent and replay talk to models through one small `LLMClient` protocol
(`complete(LLMRequest) -> LLMResponse`, ADR 0002) instead of a heavy provider SDK. `MockLLM`
scripts answers for tests and offline demos; the adapters below talk to real providers.

`OpenAICompatClient` talks to any OpenAI-compatible chat completions API — OpenAI, Ollama,
vLLM, LM Studio — through `base_url` (retries with backoff on 429/5xx, tools, JSON mode):

```python
from agentlens.llm import LLMRequest, OpenAICompatClient

llm = OpenAICompatClient(model="gpt-4o-mini")  # API key from OPENAI_API_KEY
local = OpenAICompatClient(model="llama3.1", base_url="http://localhost:11434/v1")  # Ollama
request = LLMRequest(messages="Greet me in JSON", response_format={"type": "json_object"})
print(llm.complete(request).content)  # e.g. {"greeting": "Hello!"}
```

`AnthropicClient` speaks the Anthropic Messages API with the same `LLMRequest`/`LLMResponse`
(system prompts, tools and tool results mapped for you; `max_tokens` defaults to 1024;
JSON `response_format` becomes a system-prompt instruction):

```python
from agentlens.llm import AnthropicClient, LLMRequest, Message

claude = AnthropicClient(model="claude-3-5-haiku-latest")  # API key from ANTHROPIC_API_KEY
request = LLMRequest(messages=[Message.system("Be brief."), Message.user("Say hi")])
print(claude.complete(request).content)  # e.g. Hi!
```

Or pick the provider from config instead of code with `agentlens.get_client` — a
`"provider:model"` spec, or the `AGENTLENS_MODEL` env var when called without one:

```python
import agentlens
from agentlens.llm import LLMRequest

llm = agentlens.get_client("mock")  # echo client, no network; try "ollama:llama3.1",
# "openai:gpt-4o-mini" (OPENAI_API_KEY) or "anthropic:claude-3-5-haiku-latest" (ANTHROPIC_API_KEY)
response = llm.complete(LLMRequest(messages="Hello"))
print(response.content, response.usage)  # Hello input_tokens=1 output_tokens=1 cost_usd=0.0
```

Providers: `openai`, `anthropic`, `ollama` (`OLLAMA_BASE_URL`, default
`http://localhost:11434/v1`), `vllm`, `lmstudio`, `openai-compat` (`OPENAI_COMPAT_BASE_URL`)
and `mock`. Responses get `usage.cost_usd` from a packaged table of approximate prices
(`agentlens/llm/pricing.yaml`; local models cost 0, unknown models stay `None`); merge your
own prices with `AGENTLENS_PRICING=my-pricing.yaml` or `get_client(..., pricing=...)`.


See also: [ADR 0002](../adr/0002-own-llm-protocol.md), [design §3](../design.md#3-llm-adapters),
API: [`agentlens.llm`](../api/llm.md).
