# Replay with cassettes

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


See also: [ADR 0004](../adr/0004-cassette-replay.md), [design §4](../design.md#4-replay),
API: [`agentlens.replay`](../api/replay.md).
