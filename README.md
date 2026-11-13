# AgentLens

[![CI](https://github.com/ggbifulco/agentlens/actions/workflows/ci.yml/badge.svg?branch=main)](https://github.com/ggbifulco/agentlens/actions/workflows/ci.yml)

**Regression testing & evaluation for LLM agents.** Record traces, replay runs, score them with
LLM-as-judge and deterministic checks, and catch regressions in CI — before they ship.

> 🚧 Under active development. See the [roadmap](#roadmap) and the project board.

## Why
You changed a prompt, swapped a model or refactored a tool. Did your agent get worse?
Agent behaviour is non-deterministic and quality is fuzzy, so normal unit tests don't answer that.
AgentLens turns agent runs into a **statistically sound regression suite** that runs locally and in CI.

## Planned features
- 📼 **Record** agent runs as structured traces (LLM calls, tool calls, cost, latency) — framework-agnostic.
- 🔁 **Replay** deterministically from cassettes, or live against a new model/prompt.
- ⚖️ **Evaluate** with deterministic checks (tool sequence, JSON schema, budgets) and LLM-as-judge.
- 📊 **Compare** experiments with paired-bootstrap confidence intervals, not just averages.
- 🚦 **Gate** PRs in CI and post a report as a PR comment.
- 🔌 **Swappable models**: mock, OpenAI-compatible (OpenAI, Ollama, vLLM), Anthropic.

## Quick look: recording a trace
Tracing works today (`agentlens.run`, `@agentlens.trace`, `agentlens.wrap_llm`,
`agentlens.wrap_tool`); storage and cassette replay are in place, evaluation is next.

```python
import agentlens
from agentlens.llm import LLMRequest, MockLLM

llm = agentlens.wrap_llm(MockLLM(["Let me check.", "Order 1234 has shipped."]))

@agentlens.wrap_tool
def lookup_order(order_id: str) -> dict[str, str]:
    return {"id": order_id, "status": "shipped"}

@agentlens.trace(kind="agent_step")
def agent(question: str) -> str:
    llm.complete(LLMRequest(messages=question))
    order = lookup_order("1234")
    answer = llm.complete(LLMRequest(messages=f"{question} -> {order}"))
    return answer.content or ""

with agentlens.run("support-agent", input="Where is order 1234?") as trace:
    trace.output = agent("Where is order 1234?")

print(trace.output, trace.tool_calls, trace.total_tokens, f"{trace.latency_ms:.2f} ms")
# Order 1234 has shipped. ['lookup_order'] 20 0.86 ms
```

`wrap_llm` records each call as an `llm_call` span (request/response, token usage, model,
temperature, tools, request hash); `wrap_tool` records `tool_call` spans with the arguments
and result. Both work as decorators or plain wrappers, sync or async.
Spans nest automatically across `async`/`await`, `asyncio.gather` and `asyncio.to_thread`
(context lives in `contextvars`); outside a `run` the decorators and wrappers are a no-op.

Before a finished trace reaches its sink (e.g. `store.save_trace`), secrets are masked:
values under keys such as `api_key`/`authorization` and strings that look like API keys
(`sk-...`, `Bearer ...`, `AKIA...`, `ghp_...`) become `[REDACTED]`. Pass your own
`redactor=` (any `Callable[[JsonValue], JsonValue]`, e.g. a configured
`agentlens.PatternRedactor`) or use `agentlens.redact_trace(trace)` standalone.

### Real models
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

## Replay with cassettes
Record the LLM and tool calls of a run once, then play them back offline and
deterministically (ADR 0004). Wrap *inside* the tracing wrappers so playback is traced
exactly like the live run:

```python
import agentlens
from agentlens.llm import LLMRequest, MockLLM
from agentlens.replay import Cassette

def lookup_order(order_id: str) -> dict[str, str]:
    return {"id": order_id, "status": "shipped"}

def agent(llm, lookup) -> str:
    order = lookup("1234")
    return llm.complete(LLMRequest(messages=f"Summarise {order}")).content or ""

cassette = Cassette()  # mode="record": calls go to the real client/tool and are stored
llm = agentlens.wrap_llm(cassette.wrap_llm(MockLLM(["Order 1234 has shipped."])))
print(agent(llm, agentlens.wrap_tool(cassette.wrap_tool(lookup_order))))
cassette.save("cassettes/support.jsonl")  # header line + one call per line

replay = Cassette.load("cassettes/support.jsonl", mode="strict")  # no real client needed
llm = agentlens.wrap_llm(replay.wrap_llm())
print(agent(llm, agentlens.wrap_tool(replay.wrap_tool(lookup_order))))  # same answer
```

Calls are keyed by `(index, sha256 of the normalised request)`, counted separately for LLM
and tool calls. `strict` raises `CassetteMissError` (with the recorded and actual request)
when the request at an index changed, `lenient` matches by index only, and `fallthrough`
sends misses to the real client (`record_new=True` also stores them).
`Cassette.from_trace(trace)` builds a cassette from an already recorded trace.

When a replay departs from the recording you get the first divergence: call index and a
unified diff of the recorded vs actual request (`ReplayDivergence`). `strict` puts it in the
`CassetteMissError` message (and `exc.divergence`); `lenient`/`fallthrough` keep going and
collect them in `cassette.divergences`; inside a run the diverging span gets
`replay_diverged=True` and the trace metadata `replay_divergence`:

```python
def agent_v2(llm, lookup) -> str:  # the prompt changed
    order = lookup("1234")
    return llm.complete(LLMRequest(messages=f"Summarise briefly: {order}")).content or ""

replay = Cassette.load("cassettes/support.jsonl", mode="lenient")  # plays recorded answers
with agentlens.run("support-v2") as trace:
    agent_v2(agentlens.wrap_llm(replay.wrap_llm()),
             agentlens.wrap_tool(replay.wrap_tool(lookup_order)))
print(replay.first_divergence)  # llm call #0: request differs from the recording [lenient] + diff
print(trace.metadata["replay_divergence"]["index"])  # 0
```

`find_divergence(recorded_trace, replayed_trace, kind="llm")` compares two traces the same way.

## Architecture
See [docs/architecture.md](docs/architecture.md) · [requirements](docs/requirements.md) ·
[design](docs/design.md) · [ADRs](docs/adr/).

## Roadmap
| Milestone | Scope |
|---|---|
| M0 Foundations | tooling, CI, board automation |
| M1 Tracing core | schema, recorder, SQLite store |
| M2 LLM adapters & replay | swappable clients, cassettes |
| M3 Evaluators | deterministic + LLM judges |
| M4 Suites & regression | runner, bootstrap comparison, gates, CLI |
| M5 Reporting & CI | terminal/HTML reports, GitHub Action |
| M6 Demo & integrations | example agent, regression story, UI, OTel |
| M7 Release | docs site, v0.1.0 on PyPI |

## How this repo is built
Developed in public by an AI agent working one backlog issue per session, with human review.
The process is documented in [AGENTS.md](AGENTS.md) and every session is logged in [docs/PROGRESS.md](docs/PROGRESS.md).

## License
Apache-2.0
