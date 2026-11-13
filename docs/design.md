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
  (and `error` if the block raised, then re-raises) and calls `sink(trace)` once. A sink is any
  `Callable[[Trace], None]`; `store.save_trace` is the usual one.
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
- Redaction (`tracing/redaction.py`, NFR-5): a `Redactor` is any `Callable[[JsonValue], JsonValue]`
  (no protocol class: a plain function or lambda is enough). `run(..., redactor=)` defaults to
  `DEFAULT_REDACTOR`; `None` disables it. It is applied once, in `Recorder.finish`, to a
  *copy* built by `redact_trace(trace, redactor)`: trace input/output/metadata/error and span
  input/output/attributes/error; ids, names, timings and usage are untouched. The sink gets the
  copy, while the live trace yielded to user code stays unredacted (in-process code already
  holds those values; the boundary NFR-5 protects is persistence). Redacting at finish rather
  than per span keeps the hot path cheap and also covers fields user code sets directly
  (`trace.output = ...`).
- `PatternRedactor(patterns=, keys=, replacement="[REDACTED]")` walks nested dicts/lists without
  mutating them. Mapping keys are compared case-insensitively with `-` ≡ `_`; a sensitive key
  (`api_key`, `authorization`, `password`, `token`, `client_secret`, `cookie`, ..., or any key
  ending in `_api_key`/`_secret`/`_password`) has its whole value masked. Strings are scanned
  with `DEFAULT_PATTERNS`: PEM private keys, `sk-...` (OpenAI and Anthropic `sk-ant-`),
  `Bearer <token>`, AWS `AKIA`/`ASIA`, GitHub `gh?_`/`github_pat_`, Google `AIza`, Slack
  `xox?-`, JWTs and `api_key=...`/`password: ...` assignments. A named group `secret` limits
  masking to that part (`Bearer [REDACTED]`). Custom redactor results are coerced back to JSON.
- Fail closed: if the redactor raises, the error is logged and the trace is **not** passed to
  the sink (losing a trace beats persisting a secret); the agent is never affected (FR-T5).
  Stores and `import_jsonl` do not redact on their own: call `redact_trace` before
  `store.save_trace` for traces recorded outside `run`.
- Auto-tracing (`tracing/wrappers.py`): `wrap_llm(client, name=, provider=, include_raw=False)`
  returns a `TracedLLM` — itself an `LLMClient` — whose `complete`/`acomplete` record an
  `llm_call` span. Name: `name`, else `request.model`, else the client's `model` attribute, else
  the class name. Input: the `LLMRequest` JSON with default fields omitted (still validates back
  into an equal `LLMRequest`). Output: the `LLMResponse` JSON without `usage` (stored on
  `span.usage`, so `Trace.total_tokens/total_cost` work) and without `raw` (provider payloads are
  bulky and may hold ids/headers) unless `include_raw=True`. Attributes: `client`
  (`module.Class`), `provider` (argument or the client's string `provider` attribute; omitted if
  unknown), `model`, `temperature`, `max_tokens`, `n_messages`, `n_tools`, `tool_names`,
  `response_format` (its `type`), `request_hash` (`LLMRequest.fingerprint()`, the cassette key of
  ADR 0004) and, after the call, `response_model` and `n_tool_calls`. Unknown attributes are
  delegated to the wrapped client (`llm.requests` on a wrapped `MockLLM`); wrapping a
  `TracedLLM` again returns it unchanged.
- `wrap_tool(fn, name=)` / `@wrap_tool` / `@wrap_tool(name=...)` is `@trace(kind="tool_call")`
  with tool naming: the span name is `name` or `fn.__name__` (not `__qualname__`: it is the
  tool name that `Trace.tool_calls`/`tool_sequence` compare). It accepts any callable
  (`functools.partial`, objects with `__call__`, builtins without a signature → raw
  `args`/`kwargs` as input). The wrapper carries `__wrapped__` and `__agentlens_tool__`
  (`TOOL_NAME_ATTR`); re-wrapping returns it unchanged, or re-wraps the original under a new
  name, so a tool never produces two nested spans.
- Both are no-op passthroughs outside `run` (one `ContextVar.get`, no request hashing or
  serialisation). Client/tool exceptions are recorded (`"Type: message"`) and re-raised.
  Attribute building and output serialisation run behind the FR-T5 guards
  (`_close(output_factory=...)` evaluates the output inside the guard), so even a client that
  returns a non-`LLMResponse` or a dump that raises only produces a log line.
  `Recorder.end_span(usage=, attributes=)` records usage and merges late attributes.
  Measured overhead ≈ 65 µs per `llm_call` span with `MockLLM` (checked in `test_overhead.py`).
- Replay (#13) builds on this: a cassette is an `LLMClient` wrapping the real one, so
  `wrap_llm(cassette)` traces playback exactly like live calls, and a recorded trace already
  holds per-call `request_hash` + request/response JSON (call index = order of `llm_call`
  spans) to build a cassette from. Tool playback can hook in via `__agentlens_tool__`.

## 2b. Store (`agentlens.store`, ADR 0003)
- `Store` protocol (`runtime_checkable`): `save_trace(trace, experiment_id=)`, `get_trace`,
  `list_traces(name=, experiment_id=, metadata=, limit=)`, `save_experiment`, `get_experiment`,
  `list_experiments(name=, suite=, is_baseline=)`, `save_eval_results(trace_id, results)`,
  `get_eval_results(trace_id)`, `list_eval_results(experiment_id) -> {trace_id: [EvalResult]}`.
  `get_*` return `None` when missing; lists are most recent first.
- Saves are upserts. Re-saving a trace without `experiment_id` keeps its existing link.
  `save_eval_results` *replaces* a trace's results (re-evaluation is idempotent) and keeps their
  order; the same evaluator may appear twice (e.g. two `contains` checks).
- `SQLiteStore(path=".agentlens/agentlens.db")`: each object is one JSON document (`data`)
  plus copied columns used to filter/sort (`name`, `experiment_id`, `start_time`, ...).
  Timestamps are fixed-width UTC ISO strings so text order = time order.
- Metadata filter = equality on JSON scalars via `json_each`, comparing the JSON type too
  (`True` ≠ `1`, `None` ≠ missing key; `3` = `3.0`).
- Relations: trace → experiment and eval result → trace are foreign keys (`ON DELETE CASCADE`);
  a dangling reference raises `StoreError` and the write is rolled back.
- Migrations: append-only tuple `MIGRATIONS` in code; version in `PRAGMA user_version`; each
  migration + version bump in one transaction. A database newer than the code raises
  `SchemaVersionError`.
- One connection in autocommit mode, explicit `BEGIN IMMEDIATE` per write, shared behind an
  `RLock` (`check_same_thread=False`) so runs in a thread pool can use it as sink.

## 2c. JSONL interchange (`agentlens.store.jsonl`, ADR 0003)
- Format: one `Trace` JSON document per line (`Trace.model_dump_json()`), UTF-8, `\n` line ends.
  No header or record-type wrapper: every line is directly a trace, so files work with
  `grep`/`jq` and diff cleanly. Blank lines are ignored on read.
- `export_jsonl(store, path, name=, experiment_id=, metadata=, limit=) -> int` takes the
  `Store.list_traces` filters and writes the selection **oldest first** (`limit` still keeps the
  most recent N). `import_jsonl(store, path, experiment_id=) -> int` upserts every trace, so
  re-importing is idempotent; `experiment_id` optionally links them to a stored experiment.
- Only traces travel: the experiment link and eval results are not part of the file
  (they can be recomputed; a richer format can be added later if needed).
- Safety: writes go to a temp sibling + `os.replace` (no half-written file); import validates the
  whole file first, so a malformed line raises `JSONLFormatError(path:line)` and saves nothing.
- `write_jsonl(traces, path)` / `read_jsonl(path)` are the store-free building blocks
  (fixtures, cassettes). Both functions use only the `Store` protocol.

## 3. LLM adapters
```python
class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
    async def acomplete(self, request: LLMRequest) -> LLMResponse: ...
```
`LLMRequest(messages, model, temperature, max_tokens, tools, response_format)`;
`LLMResponse(content, tool_calls, usage, raw)`. Factory: `get_client("openai:gpt-4o-mini")`,
`"ollama:llama3.1"`, `"anthropic:claude-..."`, `"mock:scripted"`. Pricing table in YAML.

Implementation notes (`src/agentlens/llm/`):
- `types.py`: `Message(role, content, tool_calls, tool_call_id)` with roles
  `system|user|assistant|tool`; validated so only assistant messages carry `tool_calls`, only
  (and every) tool message has `tool_call_id`, and only assistant messages may omit `content`.
  Constructors `Message.system/user/assistant/tool`. `ToolCall(id, name, arguments)` keeps
  arguments **parsed** (`dict`), never a JSON string — adapters decode/encode at the edge.
  `ToolSpec(name, description, parameters)` with a JSON Schema (default: empty object schema).
- `LLMRequest`: `None` means "client default" for `model`/`temperature`/`max_tokens`;
  `messages` also accepts a plain string on validation (YAML/JSON shorthand for one user
  message). `canonical_json()` (sorted keys, compact, UTF-8) and `fingerprint()` (sha256 of it)
  are the normalised request key for cassettes (§4) — equal models give equal keys regardless
  of dict ordering.
- `LLMResponse(content, tool_calls, usage, raw, model)`: `usage` is the shared
  `schema.Usage` and is `None` when a provider reports nothing (unknown ≠ zero); `raw` keeps the
  provider payload for debugging; `to_message()` turns it into the next assistant turn.
- `LLMClient` is a `runtime_checkable` `Protocol` (structural: no base class to inherit).
- `MockLLM(script, model="mock", cycle=False, default=None)`: script is a list (in order), a
  callable `(request) -> item`, or a `{regex: item}` map searched (`re.search`, insertion order)
  against the **last user message**. Items: `LLMResponse | str | ToolCall | list[ToolCall]`.
  Exhausted list → `MockExhaustedError` by default (an agent making more calls than scripted is
  usually a bug), unless `cycle=True` or a `default` is given; no regex match →
  `default` or `MockNoMatchError`. Usage is deterministic: whitespace word counts of all request
  messages (tool calls counted as `name + sorted-JSON args`) and of the response, `cost_usd=0`;
  an item's own `usage`/`model` win. All requests are kept in `.requests`; thread-safe;
  `acomplete` delegates to `complete`.
- `errors.py`: adapter failures derive from `LLMError` — `LLMHTTPError(status_code, body)`
  (with `LLMRateLimitError` for 429), `LLMConnectionError` (with `LLMTimeoutError`) after the
  retries are exhausted, and `LLMResponseError` for 2xx answers that cannot be understood (not
  JSON, no choices, tool arguments that are not a JSON object). Messages carry the provider's
  error message and the URL without query string, never request headers.
- `_http.py` (private, shared by the OpenAI-compatible and Anthropic adapters):
  `JSONHTTPClient.post_json/apost_json` on `httpx` (ADR 0002: no vendor SDKs). Retries
  408/429/5xx, timeouts, network and remote-protocol errors; other 4xx fail at once.
  `RetryPolicy(max_retries=2, initial_delay=0.5, max_delay=8, max_retry_after=60, jitter,
  sleep, async_sleep)`: delay = server hint (`retry-after-ms`, then `Retry-After` seconds or
  HTTP date, capped at `max_retry_after`, no jitter) else `jitter(min(max_delay,
  initial_delay * 2**n))`; default jitter is "equal jitter" (`[d/2, d]`). `jitter`/`sleep` are
  injectable so tests are instant and deterministic. The sync `httpx.Client` is lazy and reused
  (pooling, closed by `close()`/context manager); async calls use an injected `AsyncClient` or
  a short-lived one per call, which keeps the client safe across event loops (repeated
  `asyncio.run`). Retries are logged at DEBUG on `agentlens.llm`, without headers.
- `openai_compat.py`: `OpenAICompatClient(model=None, *, base_url="https://api.openai.com/v1",
  api_key=None, timeout=60, max_retries=2, retry_policy=None, headers=None, provider=None,
  http_client=None, async_http_client=None)` posts to `{base_url}/chat/completions`, so one
  adapter covers OpenAI, Ollama (`:11434/v1`), vLLM (`:8000/v1`), LM Studio (`:1234/v1`) and
  other compatible servers. Key: argument, else `OPENAI_API_KEY`, else no `Authorization`
  header (local servers); `api_key=""` never sends one. Mapping: `ToolSpec` → `{"type":
  "function", "function": {...}}`; assistant `tool_calls` arguments JSON-encoded, tool messages
  keep `tool_call_id`; `temperature`/`max_tokens`/`response_format` only sent when set
  (`max_tokens`, not `max_completion_tokens`, as that is what compatible servers accept).
  Response: first choice; tool-call arguments decoded (strings or already-objects, empty →
  `{}`), missing ids synthesised as `call_<index>`; `usage` → `Usage(prompt_tokens,
  completion_tokens, cost_usd=None)` (pricing is #12), `None` if not reported; `raw` is the
  response body only. `provider` (read by `wrap_llm`) defaults to `"openai"` for the official
  URL and `"openai_compat"` otherwise. The key is kept private and excluded from `repr`.
- `anthropic.py`: `AnthropicClient(model=None, *, base_url="https://api.anthropic.com",
  api_key=None, max_tokens=1024, timeout=60, max_retries=2, retry_policy=None, headers=None,
  provider="anthropic", http_client=None, async_http_client=None)` posts to
  `{base_url}/v1/messages` through the same `JSONHTTPClient` (529 "overloaded" is a retried
  5xx). Headers: `anthropic-version: 2023-06-01` (overridable via `headers`) and `x-api-key`
  from the argument, else `ANTHROPIC_API_KEY`, else none (`api_key=""` never sends one).
  Request mapping: every `system` message, wherever it appears, is joined (blank line) into the
  top-level `system` string; `ToolSpec` → `{name, description?, input_schema}`; assistant
  `tool_calls` → `tool_use` blocks after an optional text block (empty text is dropped, as the
  API rejects it); `tool` messages → `tool_result` blocks in a `user` turn, merged with the
  previous user turn when that one already holds tool results — so the results of one assistant
  turn (and a user message right after them) form a single user turn; other messages keep plain
  string content. `max_tokens` is mandatory for Anthropic: request value, else the client's
  `max_tokens` (default `DEFAULT_MAX_TOKENS = 1024`). `temperature` only when set.
  **`response_format`**: Anthropic has no JSON mode, so `json_object`/`json_schema` are emulated
  by appending an instruction ("Respond only with a single valid JSON object…", plus the schema
  for `json_schema`) to the system prompt; `{"type": "text"}` is a no-op and other types raise
  `LLMError` (fail loudly rather than silently ignore). The answer is not validated as JSON —
  that is the `json_schema` evaluator's job. Chosen over ignoring the field (judges asking for
  JSON would get prose) and over tool-forcing (`tool_choice` with a synthetic tool), which would
  change the response shape (`tool_calls` instead of `content`) and differ from the other
  adapters. Response: `text` blocks concatenated into `content` (`None` if none), `tool_use`
  blocks → `ToolCall` (`input` must be an object; missing ids → `toolu_<index>`), other blocks
  (e.g. `thinking`) ignored but kept in `raw`; `usage.input_tokens` = `input_tokens` +
  `cache_creation_input_tokens` + `cache_read_input_tokens` (all are prompt tokens; `Usage`
  has no cache fields), `output_tokens` as is, `cost_usd=None` until pricing (#12).

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
