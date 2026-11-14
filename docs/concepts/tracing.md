# Tracing

A **trace** is one run of your agent: its input and output, plus a tree of **spans** for every
step, LLM call and tool call, with token usage, cost, latency and errors. Four entry points
record it, with no dependency on any agent framework: `agentlens.run` (one run),
`@agentlens.trace` (a step of your own), `agentlens.wrap_llm` (an LLM client) and
`agentlens.wrap_tool` (a tool function).


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


## Where traces go

`agentlens.run(..., sink=store.save_trace)` hands the finished (redacted) trace to any callable;
the [store](store.md) is the usual one, the [OpenTelemetry exporter](../opentelemetry.md)
another. When a [suite](suites.md) runs, every run is traced and saved for you.

See also: [design §2](../design.md#2-tracing), API: [`agentlens.tracing`](../api/tracing.md).
