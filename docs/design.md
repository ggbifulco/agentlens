# AgentLens — Detailed design

## 1. Data model (`agentlens.schema`)

```python
class SpanKind(StrEnum): AGENT_STEP, LLM_CALL, TOOL_CALL, RETRIEVAL, CUSTOM

class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0
    cost_usd: float | None = None

class Span(BaseModel):
    id: str                      # uuid7 (time-ordered)
    trace_id: str
    parent_id: str | None
    kind: SpanKind
    name: str
    start_time: datetime
    end_time: datetime | None
    input: JsonValue
    output: JsonValue | None
    error: str | None
    usage: Usage | None          # LLM spans
    attributes: dict[str, JsonValue]   # model, temperature, tool name, ...

class Trace(BaseModel):
    id: str
    name: str
    input: JsonValue
    output: JsonValue | None
    error: str | None                  # agent raised; run still recorded
    start_time: datetime
    end_time: datetime | None
    spans: list[Span]
    metadata: dict[str, JsonValue]     # git sha, agent version, case id, repetition
    # derived: total_cost, total_tokens, latency_ms, tool_calls, n_steps

class EvaluatorSpec(BaseModel):  # "no_error" | {max_steps: 8} | {tool_sequence: {mode: ordered}}
    name: str
    params: dict[str, JsonValue]       # scalar shorthand stored as {"value": 8}

class Case(BaseModel):          # one dataset row
    id: str
    input: JsonValue
    expected: JsonValue | None
    evaluators: list[EvaluatorSpec]
    tags: list[str]

class EvalResult(BaseModel):
    evaluator: str
    score: float                 # normalised 0..1
    passed: bool
    reason: str | None
    details: dict[str, JsonValue]

class Experiment(BaseModel):
    id: str; name: str; suite: str; created_at: datetime
    config: dict[str, JsonValue] # model, prompt version, git sha
    is_baseline: bool
```

Implementation notes (`src/agentlens/schema/`):
- All models forbid unknown fields and validate on assignment. Datetimes must be
  timezone-aware and are normalised to UTC.
- Ids are UUIDv7 strings (`schema.ids.new_id`): time-ordered, monotonic within a process.
- `Trace` validates that spans share its id, span ids are unique and parents exist.
- Derived metrics are plain properties, never serialised, so stored totals cannot go stale:
  `total_cost` is `None` when no span has a cost (unknown ≠ free); `latency_ms` is
  `end_time - start_time` (`None` while running); `tool_calls` lists tool span names in
  start order; `n_steps` counts `AGENT_STEP` spans, falling back to `LLM_CALL` spans.

## 2. Tracing
- Current trace/span held in `contextvars` → works with threads and asyncio.
- API: `agentlens.run(name, input=...)` (context manager), `@agentlens.trace(kind=...)`,
  `agentlens.wrap_llm(client)` (auto-records LLM spans), `agentlens.wrap_tool(fn)`.
- Recorder buffers spans and flushes to the store on run end; exceptions inside the recorder
  are logged, never raised into user code (FR-T5).
- Redaction: pluggable `Redactor` applied before persistence (default: API-key patterns).

Implementation notes (`src/agentlens/tracing/`):
- `run(name, input=, metadata=, sink=)` yields the live `Trace`; on exit it sets `end_time`
  (and `error` if the block raised, then re-raises) and calls `sink(trace)` once. Until the
  `Store` exists (#5) a sink is any `Callable[[Trace], None]` (e.g. `store.save_trace`).
- Two `ContextVar`s hold the current `Recorder` and the innermost open `Span`. A span's parent
  is the current span when it starts; `run` resets the current span so nested runs are
  independent traces. Plain `threading.Thread`s do not inherit the context (no recording);
  use `asyncio.to_thread` or `contextvars.copy_context().run` to carry it over.
- `@trace(kind=, name=)` (also bare `@trace`) wraps sync and async functions; generator
  functions are rejected at decoration time. Span input = bound arguments without
  `self`/`cls`; output = return value. `span(name, kind=)` is the context-manager equivalent.
- Values are converted to JSON with pydantic's serializer, falling back to `repr()`.
- Every recorder operation runs behind a guard that logs to the `agentlens.tracing` logger;
  user exceptions (including `CancelledError`) are recorded as `"Type: message"` and re-raised.
- Measured overhead ≈ 40 µs/span (budget 5 ms, checked by `tests/tracing/test_overhead.py`).

## 3. LLM adapters
```python
class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
    async def acomplete(self, request: LLMRequest) -> LLMResponse: ...
```
`LLMRequest(messages, model, temperature, max_tokens, tools, response_format)`;
`LLMResponse(content, tool_calls, usage, raw)`. Factory: `get_client("openai:gpt-4o-mini")`,
`"ollama:llama3.1"`, `"anthropic:claude-..."`, `"mock:scripted"`. Pricing table in YAML.

## 4. Replay
- **Cassette**: during recording, each LLM/tool call is stored with
  `key = (call_index, sha256(normalised request))`. During playback the wrapped client returns
  the stored response. Modes: `strict` (hash must match), `lenient` (index only), `fallthrough`
  (miss → call real client).
- **Divergence**: first call whose request hash differs is reported with a diff of the request.
- **Live replay**: re-run case inputs with a different `LLMClient`/config, recording a new trace.

## 5. Evaluators
```python
class Evaluator(Protocol):
    name: str
    def evaluate(self, trace: Trace, case: Case) -> EvalResult: ...
```
Built-ins: `exact_match`, `contains`, `regex`, `json_schema`, `tool_sequence(mode=strict|ordered|subset)`,
`max_steps`, `cost_budget`, `latency_budget`, `no_error`, `llm_rubric`, `llm_pass_fail`, `pairwise`.
Judges use `LLMClient`, temperature 0, prompt templates in `evals/prompts/`, cache keyed by
`sha256(judge_model + prompt + trace output)`.

## 6. Suites (YAML)
```yaml
name: support-agent
agent: examples.support_agent:run_agent   # import path, callable(input) -> output
repetitions: 3
defaults:
  evaluators: [no_error, {max_steps: 8}]
cases:
  - id: refund-simple
    input: "I want a refund for order 1234"
    expected: {tools: [lookup_order, issue_refund]}
    evaluators:
      - {tool_sequence: {mode: ordered}}
      - {llm_rubric: {criteria: "Polite, confirms refund amount"}}
```

## 7. Comparison & gates
- Metrics per case: pass rate, mean score per evaluator, cost, latency, steps.
- Aggregate delta with **paired bootstrap** (10k resamples over cases) → 95% CI.
  Regression = CI entirely below 0 for "higher is better" metrics.
- Gates YAML: `success_rate: {min: 0.9}`, `cost_usd: {max_delta_pct: 20}`, `regressions: {max: 0}`.

## 8. CLI
`agentlens run`, `agentlens compare`, `agentlens report`, `agentlens baseline set`,
`agentlens traces ls/show`, `agentlens export/import`, `agentlens ui`.

## 9. Testing strategy
- Unit tests per module; property tests (hypothesis) for schema round-trips and bootstrap.
- Integration tests run the example agent end-to-end with `MockLLM` (no keys, deterministic).
- Optional live tests marked `@pytest.mark.live`, skipped without env vars.
