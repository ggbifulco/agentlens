# OpenTelemetry export

`pip install "agentlens[otel]"` adds an exporter that sends traces to any OpenTelemetry backend
(Jaeger, Grafana Tempo, Phoenix, Langfuse, ...) using the
[GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/). Each run is
exported as `invoke_agent <name>`, LLM calls as `chat <model>` (`gen_ai.provider.name`,
`gen_ai.request.model`, `gen_ai.usage.input_tokens`, ...), tools as `execute_tool <tool>`
(`gen_ai.tool.name`), with the recorded timings and errors:

```python
import agentlens
from agentlens.integrations.otel import OTelExporter, export_trace, fan_out
from agentlens.store import SQLiteStore

exporter = OTelExporter()          # the global TracerProvider; configure OTLP as usual
with SQLiteStore("runs.db") as store:
    with agentlens.run("support", sink=fan_out(store.save_trace, exporter)):
        ...                        # your agent, with wrap_llm / wrap_tool
    for trace in store.list_traces(name="support"):    # or export what is already stored
        export_trace(trace)
```

Prompts, outputs and tool arguments are **not** exported unless you pass
`capture_content=True` (they may hold personal data). Export errors are logged, never raised into
the agent.


API: [`agentlens.integrations.otel`](api/integrations.md).
