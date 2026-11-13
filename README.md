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
`agentlens.wrap_tool`); storage is in place, replay and evaluation are next.

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
