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

Implementation notes — OpenTelemetry export (`integrations/otel.py`, #32, FR-T6):
- Optional extra `agentlens[otel]` (`opentelemetry-api`, `opentelemetry-sdk`); the module
  raises an `ImportError` naming the extra when they are missing, and nothing else imports it.
- **After the fact, not live**: `export_trace(trace, tracer=|tracer_provider=,
  capture_content=False, context=None)` replays a *finished* (redacted) trace into OTel with
  explicit start/end timestamps (exact integer nanoseconds). Alternative considered: emitting OTel
  spans live from the recorder — rejected because it would bypass redaction (NFR-5), put the
  OTel API on the hot path, and couple the core to an optional dependency. `OTelExporter` is a
  sink (`run(..., sink=)`; errors logged, FR-T5); `fan_out(*sinks)` combines it with the store.
- **Mapping** (GenAI semconv): trace → root `invoke_agent {name}` (`gen_ai.agent.name`,
  `agentlens.metadata.*` scalars, total cost/tokens); `agent_step` → `invoke_agent {name}`;
  `llm_call` → `chat {model}`, kind CLIENT, `gen_ai.provider.name`, `gen_ai.request.model/
  temperature/max_tokens`, `gen_ai.response.model`, `gen_ai.usage.input_tokens/output_tokens`
  (from the `wrap_llm` attributes and `span.usage`); `tool_call` → `execute_tool {tool}`,
  `gen_ai.tool.name`; `retrieval`/`custom` keep their name (no standard operation). All spans:
  `agentlens.trace_id`/`span_id`/`span.kind`, `agentlens.usage.cost_usd` (no gen_ai equivalent).
  Errors → status ERROR + `error.type` (`Type` of `"Type: message"`, else `_OTHER`).
- **Content opt-in**: `capture_content=True` adds `gen_ai.input.messages`,
  `gen_ai.output.messages`, `gen_ai.tool.call.arguments`/`result` (JSON strings), as the
  conventions recommend for sensitive data. The root span starts a new OTel trace unless a parent
  `context` is given (e.g. a CI job span). Span/trace ids are the SDK's; the AgentLens ids are
  attributes, so a backend span links back to the store.

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
- `list_traces(..., has_error=, offset=)` (#31): `has_error=True` keeps traces where the run or
  any span recorded an error (`json_extract` on `$.error` + `json_each` over `$.spans`), `False`
  the others; `offset` + `limit` paginate (SQLite `LIMIT -1 OFFSET n` when only an offset).
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
- `factory.py`: `get_client(spec=None, *, pricing=None, **overrides) -> LLMClient` (also
  `agentlens.get_client`). `spec` is `"<provider>:<model>"`, split on the **first** `:` so
  Ollama tags survive (`ollama:llama3.1:8b`); provider is case-insensitive with aliases
  (`openai_compat`, `lm-studio`). `None` reads `AGENTLENS_MODEL` (empty counts as unset).
  Providers and env (all read at call time; `overrides` go to the constructor and win):
  `openai` (`OPENAI_BASE_URL`, key via the adapter's `OPENAI_API_KEY` fallback), `anthropic`
  (`ANTHROPIC_BASE_URL`, `ANTHROPIC_API_KEY`), `ollama` (`OLLAMA_BASE_URL`, default
  `http://localhost:11434/v1`), `vllm` (`VLLM_BASE_URL` `:8000/v1`, optional `VLLM_API_KEY`),
  `lmstudio` (`LMSTUDIO_BASE_URL` `:1234/v1`), `openai-compat` (`OPENAI_COMPAT_BASE_URL`
  required, optional `OPENAI_COMPAT_API_KEY`) — local providers pass `api_key=""`, so
  `OPENAI_API_KEY` is never sent to a local or third-party server, and set `provider` to their
  name (recorded by `wrap_llm`). `mock` / `mock:<name>`: `MockLLM(model=<name> or "mock")`
  echoing the last user message (useful as a zero-config smoke-test client); `script=`,
  `cycle=`, `default=` overrides make it a scripted mock. Bad specs, unknown providers, a
  missing spec/env or a missing required base URL raise `LLMConfigError` (an `LLMError` and a
  `ValueError`) listing the known providers. The provider registry is the public `PROVIDERS`
  dict of builders `(model, overrides) -> LLMClient`.
- `pricing.py` + packaged `pricing.yaml` (shipped in the wheel: hatchling includes non-Python
  files under `src/agentlens`; read with `importlib.resources`): `{updated, free_providers,
  models: {id: {input, output}}, aliases: {id: id}}`, USD per 1M tokens, **approximate list
  prices dated in the file**. `Pricing.lookup(model)`: case-insensitive exact key, then alias,
  then the **longest key that is a prefix followed by `-`, `:` or `@`** — dated/tagged ids
  (`gpt-4o-mini-2024-07-18`, `claude-3-5-haiku-latest`, `claude-sonnet-4-5-20250929`) get the
  base price, while `gpt-4omni` does not match `gpt-4o`. `Pricing.cost(model, in, out,
  provider=)` is `0.0` for `free_providers` (`ollama`, `vllm`, `lmstudio`, `mock`) and `None`
  for unknown models (unknown ≠ free). `Pricing.load(path=None)` = packaged table with `path`
  (else `$AGENTLENS_PRICING`) **merged over** it — overriding a few prices or adding a model
  needs no copy of the whole table; a missing/invalid override file fails loudly.
- Cost is filled by a `PricedClient(client, pricing=None, provider=None)` wrapper rather than
  inside each adapter: adapters stay pure protocol translators (and testable without a table),
  any third-party `LLMClient` can be priced, and the table is loaded once per client. It only
  fills `usage.cost_usd` when usage exists and the provider did not report a cost; model =
  response `model` (the one that really answered), else request, else client default;
  attributes delegate to the wrapped client, so `wrap_llm(get_client(...))` still records
  `provider`/`model` and the priced usage. `get_client` wraps every HTTP client; the mock is
  returned bare (it already reports `cost_usd=0`, and tests use its `.requests`).
- PyYAML (`pyyaml`) is a runtime dependency (with `types-PyYAML` for mypy): the pricing table
  is YAML, and so will be suites and gates (§6–7); `yaml.safe_load` only.

## 4. Replay
- **Cassette**: during recording, each LLM/tool call is stored with
  `key = (call_index, sha256(normalised request))`. During playback the wrapped client returns
  the stored response. Modes: `strict` (hash must match), `lenient` (index only), `fallthrough`
  (miss → call real client).
- **Divergence**: first call whose request hash differs is reported with a diff of the request.
- **Live replay**: re-run case inputs with a different `LLMClient`/config, recording a new trace.

Implementation notes (`src/agentlens/replay/`, #13):
- `Cassette(entries=(), *, mode="record", record_new=False, metadata=None)` holds
  `CassetteEntry(index, kind, key, name, request, response)` items. `cassette.wrap_llm(client=None,
  *, model=, provider=)` returns a `CassetteLLM` (an `LLMClient`, sync + async);
  `cassette.wrap_tool(fn, name=)` (also as decorator, sync + async) does the same for tools.
  Wrap **inside** the tracing wrappers — `agentlens.wrap_llm(cassette.wrap_llm(client))`,
  `agentlens.wrap_tool(cassette.wrap_tool(fn))` — so playback produces the same spans as the
  live run (tested: a recorded MockLLM + tool run replayed against a failing client and a
  different tool gives an identical trace, ids/timestamps aside). `cassette.wrap_tool` of an
  already traced tool unwraps it and re-applies tracing outside the cassette.
- Key: `(index, key)`. `index` counts calls **per kind** (`llm`, `tool`) through one cassette,
  so adding/removing a tool call does not shift every later LLM index. LLM key =
  `LLMRequest.fingerprint()` (same value as the `request_hash` span attribute); tool key =
  `tool_key(name, arguments)` = sha256 of canonical JSON `{"name", "arguments"}`, arguments
  bound by parameter name as in `tool_call` spans. Volatile values are not an issue for the
  keys: ids inside later requests (tool-call ids) come from played-back responses, so they are
  the recorded ones.
- Modes (`cassette.mode`, switchable at any time): `record` always calls through and stores
  (replacing an entry at the same index); `strict` needs entry at index with equal key and tool
  name; `lenient` takes the entry at index whatever the request; `fallthrough` plays hits and
  sends misses to the real client/tool, storing them (replacing the mismatching entry or
  appending) only with `record_new=True`. A miss in `strict`/`lenient` raises
  `CassetteMissError(kind, index, mode, expected, actual_request, actual_key, actual_name)`
  with `expected_request` — #14 builds the request diff on these fields. A miss does not
  consume the index. Pure playback needs no real client; recording or a fallthrough miss
  without one raises `CassetteError`.
- Errors of the real client/tool are not recorded and give the index back, so the cassette
  holds the successful calls in order (a retried call is replayed once). Calls made **inside**
  a real tool execution (a tool that itself calls the LLM) bypass the cassette (a per-cassette
  `ContextVar`): playing the tool back skips its body, so recording those calls would shift
  later indexes. Concurrent calls are thread-safe (one lock) but indexed in arrival order.
- Played-back tool results are the recorded **JSON** (a tuple comes back as a list, a model as a
  dict) — enough for agents that feed results back to an LLM; typed decoding can come later.
- `CassetteLLM.model`/`provider` (read by `wrap_llm` for span names/attributes): explicit
  argument, else the real client's, else `cassette.metadata` — recording stores the client's
  values there, so playback without a client keeps span names identical.
- File format (JSONL, written atomically like `store.jsonl`): line 1 is a header
  `{"format": "agentlens-cassette", "version": 1, "metadata": {...}}`, then one entry per line
  in recording order; LLM `request` is the `LLMRequest` JSON without defaults (as span input),
  `response` the full `LLMResponse` JSON (usage and raw included). Chosen over plain entries
  without header so the format can be versioned and carry the model/provider defaults; the
  header is optional on load (hand-written fixtures). `Cassette.load(path, mode="strict")`
  validates every line (`CassetteFormatError(path:line)`, also for duplicate `(kind, index)`).
- `Cassette.from_trace(trace)` rebuilds a cassette from `llm_call`/`tool_call` spans in start
  order (failed, unfinished and tool-nested spans skipped, response `usage` taken from the
  span; `raw` only if traced with `include_raw=True`), so any traced run can be replayed.

Implementation notes (divergence, #14, FR-R3):
- `ReplayDivergence` (pydantic, `agentlens.replay.divergence`): `kind`, `index`, `reason`
  (`request_changed` | `tool_changed` | `extra_call` | `missing_call`), `mode` (`None` for trace
  comparisons), expected/actual `name`, `key` and `request`, `diff`, `diff_truncated`, `span_id`;
  `create(...)` derives reason and diff, `summary()` gives one line, `to_dict(include_requests=)`
  JSON for reports, `str()` = summary + diff. Chosen as a schema model (not a dataclass) so it
  validates and round-trips like the rest of the data model.
- Diff = `difflib.unified_diff` over pretty-printed canonical JSON (`sort_keys`, indent 2,
  unicode kept) labelled `recorded/replayed <kind> call #<i>`; tool calls diff
  `{"name", "arguments"}` so a renamed tool shows up. Bounded for errors/traces/reports:
  `DIFF_MAX_LINES=200` lines (then `... (N more diff lines)`) of `DIFF_MAX_LINE_CHARS=500`
  chars each (`request_diff(...)` exposes both limits). A side without a call diffs as empty.
- Detection is in `Cassette._begin`: any playback call whose recorded entry is missing or has a
  different key/tool name is a divergence, in every playback mode. `strict` (and `lenient` past
  the end) raise `CassetteMissError`, whose message now ends with the diff and whose
  `.divergence` holds it; `lenient`/`fallthrough` continue. All are kept in
  `cassette.divergences` / `cassette.first_divergence` (cleared by `rewind()`). `record` never
  diverges. The diff is built outside the cassette lock.
- Reports (#26-#28) are not built yet, so divergences are put on the trace: the current span, when
  it is the matching `llm_call`/`tool_call` (cassette wrapped inside the tracing wrappers), gets
  attributes `replay_diverged=True` and `replay_divergence=<summary>` (also on the failed span in
  strict mode); the trace metadata `replay_divergence` holds the **first** divergence's
  `to_dict(include_requests=False)` (the replayed request is already the span input; `span_id`
  links them). Plain snake_case keys like the other span attributes. Marking never raises.
- `find_divergence(recorded, replayed, *, kind="llm")` compares two traces: both are turned
  into cassette entries with `Cassette.from_trace` (same span selection and keys), then compared
  index by index; first mismatch, or `None`. Works for live replays against another model (FR-R2).
  "Missing calls" (the replay stopped early) are only visible this way or via `cassette.unplayed`.
- `CassetteMode`/`EntryKind` moved to `replay/_types.py` (re-exported unchanged) to avoid an
  import cycle between `cassette`, `errors` and `divergence`.

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

Implementation notes (`src/agentlens/evals/`, #15, FR-E3):
- `Evaluator` is a `runtime_checkable` Protocol whose `name` is a read-only property (so a
  plain attribute, a dataclass field or a `@property` all satisfy it). `AsyncEvaluator` adds
  `aevaluate(trace, case) -> Awaitable[EvalResult]` for judges (#18). `run_evaluator` /
  `arun_evaluator` are the only way the runner should call evaluators: they **never raise** —
  any `Exception` (or a non-`EvalResult` return) becomes `error_result(name, exc)`: score 0,
  `passed=False`, `reason="evaluator error: <Type>: <msg>"`, `details["error"]={type, message}`
  so reports can tell a crash from a genuine failure. `KeyboardInterrupt`/`SystemExit` propagate.
  `arun_evaluator` awaits `aevaluate` when present, else runs `evaluate` via `asyncio.to_thread`.
- Registry (`EvaluatorRegistry`, process-wide `default_registry`; module functions `register`,
  `unregister`, `get_evaluator`, `available_evaluators`): name → factory, `factory(**params) ->
  Evaluator` (an evaluator class is a factory). `get_evaluator(spec, **overrides)` accepts an
  `EvaluatorSpec` or its YAML spellings. **Scalar shorthand rule**: params exactly `{"value": x}`
  go to the factory's *primary* parameter — `register(..., primary=...)`, else the first
  keyword-passable parameter of its signature; a factory that declares `value` or only `**kwargs`
  gets `value=x` unchanged. Params are checked with `inspect.Signature.bind` before calling, so a
  typo is an `EvaluatorConfigError` naming the signature; `TypeError`/`ValueError` from the factory
  and non-evaluator returns are config errors too. Config errors (`UnknownEvaluatorError` with
  difflib suggestions, `EvaluatorConfigError`, `DuplicateEvaluatorError`, all `EvaluatorError`)
  are raised eagerly — a broken suite fails at load, not mid-run.
- Duplicates raise unless `replace=True`, except when the new factory has the same origin
  (`module.qualname`) as the registered one: re-running a notebook cell or reloading a module
  just replaces it.
- `@evaluator` (bare or `@evaluator(name=, threshold=0.5, register=True, replace=, registry=)`)
  returns a `FunctionEvaluator`: still callable as the original function, `evaluate` normalises
  the return value with `to_eval_result` (`EvalResult` as is; `bool` → 1/0; number in `[0, 1]` →
  `passed = score >= threshold`; `(value, reason)`; mapping with `score`/`passed`) and catches
  exceptions itself, so it is safe to call directly too. The function must take `(trace, case)`
  positionally; later parameters become keyword params of the registered factory
  (`FunctionEvaluator.bind`). `threshold` is always accepted as a spec param (pass threshold) and
  forwarded to the function only if it declares it; for a function without extra params it is
  the primary, so `{my_eval: 0.8}` sets the threshold. `async def` evaluators work: `aevaluate`
  awaits them, `evaluate` runs them with `asyncio.run` (in a worker thread if a loop is running).
- `evaluate_case(trace, case, evaluators=None, *, registry=None)` (default `case.evaluators`,
  objects and specs mixed) resolves everything first, then runs each in order;
  `aevaluate_case` runs them concurrently with `asyncio.gather`, keeping order. Merging suite
  defaults with case evaluators is the runner's job (#20).
- Built-ins (#16-#18) will register themselves at import of `agentlens.evals` (imported from its
  `__init__`), so `available_evaluators()` lists them without extra setup. Top-level re-exports:
  `agentlens.evaluator`, `agentlens.Evaluator`.

Implementation notes (`src/agentlens/evals/builtin/`, #16, FR-E1 part 1):
- Built-ins are classes (the class is the registry factory) in `evals/builtin/output.py`
  (`ExactMatch`, `Contains`, `Regex`, `JsonSchema`) and `evals/builtin/trace_checks.py`
  (`NoError`; #17's trace checks go there too). `evals/builtin/__init__.py` registers them in
  `default_registry` at import of `agentlens.evals` (`register_builtins(registry=None)` fills a
  fresh registry; `BUILTIN_EVALUATORS` maps name → (class, primary)). Primaries for the scalar
  shorthand: `exact_match` → `value`, `contains` → `text`, `regex` → `pattern`, `json_schema` →
  `schema`, `no_error` → `include_spans`. All results score 1/0 except `contains` (fraction).
- **What output is checked.** `output_value(trace, path=None)` returns `trace.output` or the part
  a `path` names: dotted string `"items.0.text"` or a list `["key.with.dots", 0]` (list indexes,
  negative ok). With a path, a string output is parsed as JSON first. A missing path raises
  `OutputPathError` (a `LookupError`), which evaluators turn into a *failed* result (the agent's
  output lacks the field), not a config error. `to_text(value)` / `output_text(trace, path=)`:
  a string as is, `None` stays `None`, anything else canonical JSON (`sort_keys`, default
  `", "`/`": "` separators, `ensure_ascii=False`) — what `contains`/`regex` search.
- Strings are always NFC-normalised before comparing (composed and decomposed accents are equal);
  `case_sensitive=False` uses `str.casefold` (`"Straße"` = `"STRASSE"`); `normalise_whitespace`
  collapses whitespace runs to one space. Flag params must be real booleans (YAML `"no"` is a
  config error).
- `exact_match(value=UNSET, case_sensitive=True, strip=True, normalise_whitespace=False,
  path=None)`: **precedence** — the `value` param, else `case.expected`, else `evaluate` raises
  `EvaluatorConfigError` (reported by `run_evaluator` as an evaluator error, not an agent
  failure). `UNSET` sentinel so `{exact_match: {value: null}}` can expect a `null` output
  (`{exact_match: null}` means "no params"). A string expectation compares text (non-string
  output as canonical JSON); any other expectation compares structurally after parsing a string
  output as JSON: key order ignored, `1 == 1.0`, but JSON types are tagged so `true != 1`; the
  text options apply to nested strings.
- `contains(text=None, mode="all"|"any", case_sensitive=True, normalise_whitespace=False,
  path=None)`: `text` is a string or list (fallback `case.expected`); empty strings/lists are
  rejected (they would always match). Score = fraction found in both modes; `all` passes only at
  1.0, `any` with at least one. `None` output contains nothing. Details: `matched`, `missing`.
- `regex(pattern, mode="search"|"match"|"fullmatch", ignore_case, multiline, dotall, path)`:
  compiled in `__init__`, so an invalid pattern is an `EvaluatorConfigError` at suite load.
  Details: `match`, `span`, `groups`, `named_groups`.
- `json_schema(schema=None, parse_json=True, max_errors=5, path=None)`: schema param, else
  `case.expected["schema"]`; checked with `validator_for(schema).check_schema` (draft from
  `$schema`, default 2020-12). A string output is parsed as JSON (one surrounding Markdown code
  fence tolerated — LLMs add them); `parse_json=False` validates it as a JSON string; `None` is
  validated as `null`. All errors are collected (`iter_errors`), sorted by `json_path`; the reason
  lists the first `max_errors` as `$.path: message`, `details["errors"]` = `{path, message,
  validator}` + `error_count`. **Dependency**: `jsonschema` (runtime, + `types-jsonschema` dev):
  full draft 4–2020-12 support, `$ref` resolution and precise error paths are too much to
  hand-roll; it is the reference implementation (MIT, wheels everywhere; pulls `attrs`,
  `referencing`, `jsonschema-specifications`, `rpds-py`).
- `no_error(include_spans=True)`: fails on `trace.error` and, by default, on any span with
  `error` — a tool/LLM call that raised even if the agent caught it, because a swallowed tool
  error is usually a regression worth seeing (adapter HTTP retries happen inside one span, so
  they do not count). `{no_error: {include_spans: false}}` checks only the run. Details list
  the first 10 failed spans in start order.

Implementation notes (`src/agentlens/evals/builtin/trace_checks.py`, #17, FR-E1 part 2):
- Trajectory checks read only the trace's derived metrics, so they work on live, stored and
  replayed traces alike. Classes/primaries: `ToolSequence` (`tool_sequence` → `tools`),
  `MaxSteps` (`max_steps` → `value`), `CostBudget` (`cost_budget` → `max_usd`),
  `LatencyBudget` (`latency_budget` → `max_ms`). Limits must be finite numbers `>= 0`
  (`max_steps` an integer; bools/strings/NaN/inf are `EvaluatorConfigError` at build).
- `tool_sequence(tools=None, mode="ordered", allow_extra=True, forbidden=None)` compares
  `trace.tool_calls` (tool span names, start order) with `tools`, else `case.expected["tools"]`
  (§6), else `evaluate` raises `EvaluatorConfigError`. **Default mode `ordered`** — the usual
  intent ("look up, then refund") without failing on an extra search. **Repeats count in every
  mode (multiset semantics)**: `[search, search]` needs two `search` calls. Scores (LCS =
  longest common subsequence, an `O(n·m)` DP; ties keep the earlier expected tools, so the
  later ones are reported missing):
  - `strict`: pass iff `actual == expected`; score `LCS / max(len(expected), len(actual))`, so
    missing, extra and reordered calls all lower it and only an exact match scores 1.
  - `ordered`: pass iff expected is a subsequence of actual; score `LCS / len(expected)`.
  - `subset`: pass iff the expected multiset is included in the calls (any order); score
    `matched / len(expected)`.
  - An empty expected list passes `ordered`/`subset`; in `strict` it means "no tool calls".
  - `allow_extra=False` (`ordered`/`subset`): calls to tools *not in* the expected list fail the
    check and are added to the denominator; retries of expected tools are not extras (use
    `strict` for "exactly these calls"). `forbidden`: any call to one of these tools fails with
    score 0 (a safety-style constraint, not partial credit).
  - Reason: verdict + `(mode): expected [...], actual [...]` (lists capped at 20 names); details
    `expected`, `actual`, `missing`, `unexpected`, `forbidden_called`, `mode`, `source`.
- **Budget scoring** (shared): pass iff `actual <= limit`; score 1 within, else `limit / actual`
  (10 % over → ≈0.91), so reports can rank how far over a run went while the gate still fails.
- `max_steps(value)`: `trace.n_steps` (`agent_step` spans if any, else LLM calls); details say
  which were counted.
- `cost_budget(max_usd, on_unknown="fail")`: `trace.total_cost`. **Unknown cost** (no span with
  a cost) and **partial cost** (some spans with `usage` but `cost_usd=None`) cannot be verified:
  if the known part already exceeds the budget it fails regardless (a lower bound); otherwise
  `on_unknown` decides — default `"fail"` (a budget you cannot check is not met, and an unpriced
  model should be noticed; `MockLLM` reports cost 0 so offline suites are unaffected), `"pass"`
  scores 1 with a reason saying the cost was not verified. Details: `cost_usd`, `max_usd`,
  `unpriced_spans`, `on_unknown`.
- `latency_budget(max_ms)`: `trace.latency_ms`; a trace that has not ended (`None`) fails with
  score 0.
- Helpers `check_choice`, `check_number`, `params_repr` moved to `builtin/_common.py`.

Implementation notes (`src/agentlens/evals/judge/`, `evals/builtin/judges.py`, `evals/prompts/`, #18, FR-E2/FR-E4):
- **Reusable machinery** (`agentlens.evals.judge`): `LLMJudge[Verdict]` is an abstract evaluator; a
  concrete judge implements `build_prompt(trace, case) -> JudgePrompt(system, user, output)`,
  `parse_reply(json_object) -> Verdict` (raises `JudgeParseError`) and `make_result(verdict, run)`.
  The base class owns the flow, identical in `evaluate` and `aevaluate` (`client.acomplete`):
  cache lookup → call → parse → retry → cache write → `EvalResult`. (Since #19 the non-evaluator
  part — client, cache, `run_judge`/`arun_judge`, `parse_reply` — is `BaseJudge`, which
  `LLMJudge` extends with `build_prompt`/`make_result`/`evaluate`.)
- **Request**: `temperature=0.0` always (`JUDGE_TEMPERATURE`), `response_format={"type":
  "json_object"}` (OpenAI JSON mode; the Anthropic adapter emulates it with a system instruction),
  optional `max_tokens`, `model=None` (the client's own). Prompts say "JSON" (required by OpenAI's
  JSON mode).
- **Prompts** are packaged `string.Template` text files in `src/agentlens/evals/prompts/`
  (`judge_system`, `rubric`, `pass_fail`, `reference`), loaded with `importlib.resources`
  (`load_template`). `string.Template` instead of `str.format`/Jinja: no dependency, the `{}` of
  the JSON answer format need no escaping, and a `$` in agent output is never re-expanded
  (substitution is single-pass). Input/output/reference are wrapped in `<input>`/`<output>`/
  `<reference>` tags and the system prompt says tagged content is data, not instructions (a
  cheap prompt-injection guard). Values are shown as text: strings as is, `None` as `(none)`,
  anything else as sorted-key JSON (`value_text`), so cache keys do not depend on dict order.
  The judge is asked for `reasoning` **before** the verdict (explain-then-score).
  `template=` overrides the packaged prompt (checked at build: valid `$` syntax, only the
  placeholders `$criteria $input $output $reference`).
- **Parsing** (`extract_json_object`): the content of Markdown code fences first, then the whole
  reply, then every `{` as a candidate start (`JSONDecoder.raw_decode`), so "Sure! {...} Hope
  this helps" works. `parse_score` accepts numbers and numeric strings (`"4"`, `"4/5"`), keys
  case-insensitive (`score`/`rating`), fractional values kept, but out-of-range (`7`, `0`),
  booleans and NaN are errors — clamping would silently turn a 0-10 scale answer into a 5.
  `parse_verdict` accepts `pass`/`fail` (also yes/no, true/false, booleans; keys `verdict`/
  `passed`/`pass`/`result`). `parse_reasoning` reads `reasoning`/`reason`/`explanation`/`rationale`.
- **Retry**: on a parse error the judge is asked again up to `retries` times (default 1, max 5),
  with its bad reply and the error appended to the conversation. Still unparseable → a *failed*
  result (score 0) with `details["error"] = {type: "JudgeParseError", message}` (the
  `error_result` shape, so reports count it as an evaluator problem) and `details["raw_reply"]`
  (clipped to 4000 chars). LLM errors (network, auth, no model configured) propagate and become
  evaluator-error results through `run_evaluator`.
- **Cache** (`JudgeCache` protocol: `get(key) -> dict | None`, `set(key, dict)`): key =
  `judge_cache_key(judge_model, prompt, output)` = sha256 of the JSON list `[judge model, system +
  user prompt, output text]` (JSON-encoded so concatenations cannot collide). The prompt already
  contains criteria, input and reference, so changing any of them, the template or the judge
  model misses the cache. The **raw reply** is cached (`{"reply", "judge_model", "model"}`), not
  the parsed verdict, and only once it parsed: a parser change re-reads old replies, a cached reply
  that no longer parses is ignored and re-asked, failures are never cached. A hit makes no LLM
  call; `details` say `cached`, `attempts` (0 on a hit), `cache_key`, `judge_model` and
  `judge_usage` (summed over attempts; absent on a hit).
  Implementations: `MemoryJudgeCache` (dict + lock) and `DiskJudgeCache(directory)` — one JSON file
  per key at `<dir>/<key[:2]>/<key>.json`, written atomically (temp file + `os.replace`), corrupt
  files are misses. **Default: `DiskJudgeCache(".agentlens/judge_cache")`** (next to the SQLite
  store, already git-ignored): FR-E4 asks for reproducible judge scores across runs, which an
  in-memory cache cannot give, and plain files are easy to inspect, prune and save/restore as a CI
  cache. `cache=` param / `$AGENTLENS_JUDGE_CACHE`: `False`/`"off"` disables, `"memory"` is a
  process-wide `MemoryJudgeCache`, a string/path is a directory. The test suite sets
  `AGENTLENS_JUDGE_CACHE=off`; tests that script a `MockLLM` judge should pass `cache=False` or a
  fresh `MemoryJudgeCache`, since two tests with the same prompt would otherwise share a verdict.
- **Judge model** (`model` spec via `get_client`, or an injected `client`, not both): default
  `$AGENTLENS_JUDGE_MODEL`, else `$AGENTLENS_MODEL`, resolved on first use (`LLMJudge.client`), so
  building judges from a suite needs no environment; the spec is syntax-checked at build. The
  cache-key label is `"<provider>:<model>"` for a spec, the client's `provider`/`model`
  attributes (else class name) for an injected client.
- `llm_rubric(criteria, min_score=4, include_expected=True, template=None, model, client, cache,
  retries=1, max_tokens=None)`: 1-5 score with anchored scale in the prompt; `score = (s - 1) / 4`
  (4 → 0.75), `passed = s >= min_score`. **Default `min_score` 4** = "meets the criteria with only
  minor issues"; 3 ("partially meets") is not a pass. Reason `judge score 4/5 (>= 4): <reasoning>`;
  details `rubric_score`, `min_score`, `reasoning`, `criteria`.
- `llm_pass_fail(criteria, include_expected=True, template=None, model, client, cache, retries,
  max_tokens)`: verdict pass → score 1, fail → 0; the prompt says to fail when any criterion is not
  met *or it cannot tell*. Details `verdict`, `reasoning`, `criteria`.
- Both show the judge the case input (`trace.input`, else `case.input`), the output and — unless
  `include_expected=False` — `case.expected` as a *reference* ("does not need to match word for
  word"). Scalar shorthand: `{llm_rubric: "Polite"}` sets `criteria`.
- Bias and calibration guidance for users: `docs/judges.md`.

Implementation notes (`src/agentlens/evals/judge/pairwise.py`, `evals/prompts/pairwise.txt`, #19, FR-E2):
- **Own API, not an evaluator**: comparing needs two traces, so `PairwiseJudge(criteria, *,
  include_expected=True, template=None, model, client, cache, retries=1, max_tokens=None)` exposes
  `compare(trace_a, trace_b, case=None) -> PairwiseResult` and `acompare` (the two orders run
  concurrently with `asyncio.gather`). It extends `BaseJudge` (extracted from `LLMJudge`, which keeps
  its API), so model resolution, temperature 0, JSON mode, parse + retry and caching are shared.
  Not registered in the evaluator registry: a registry entry would have to fake
  `evaluate(trace, case)`. The comparator (#22) will build one `PairwiseJudge` from the suite/CLI
  options and call it per case on (baseline trace, candidate trace), storing
  `result.to_eval_result(candidate="b")` (evaluator name `pairwise`, overridable with `name=`).
- **Prompt**: `output_1`/`output_2` (positions, never "A"/"B", so labels carry no bias), the case
  input, the optional reference, and an instruction that order/length are not evidence; reply
  `{"reasoning", "winner": "1"|"2"|"tie"}`. `parse_position` also accepts `1`/`2` numbers,
  "output 1", "first"/"second", "draw"/"equal"/"neither" and keys `choice`/`better`/`preference`/
  `verdict`; "A"/"B" are rejected (ambiguous after swapping).
- **Position-bias mitigation**: order `ab` shows A first, `ba` shows B first; each positional answer
  is mapped back to A/B, converted to a preference for A (1 / 0.5 / 0) and the two are averaged.
  `winner` = a if > 0.5, b if < 0.5, else tie; `consistent` = both mapped winners equal;
  `confidence = 1 - |pref_ab - pref_ba|` (1 consistent, 0.5 winner vs tie, 0 opposite → tie).
  Alternatives rejected: a single random order (bias only averaged over many cases, not per case),
  asking for a score per output (absolute scales are less reliable than a choice).
- **Cache**: each order is a normal judge call keyed by `judge_cache_key(judge model, prompt,
  json([output 1, output 2]))` — the presentation order is in the key, so `ab` and `ba` never share
  an entry, while the same ordered pair reuses its verdict (also for `compare(B, A)`; identical
  outputs make one call).
- **Failures**: if either order is still unparseable after retries, `PairwiseResult.error` is set
  ("order ab: ..."), the result is a tie with confidence 0, and `to_eval_result` fails (score 0,
  `details["error"]` in the `error_result` shape). No decision from a single order.
- **`PairwiseResult`** (frozen dataclass): `winner`, `preference`, `consistent`, `confidence`,
  `orders` (two `OrderVerdict`: order, raw `position`, mapped `winner`, reasoning, call metadata,
  error/raw reply), `criteria`, `error`; `reason`, `reasons`, `details()`,
  `to_eval_result(candidate="b")` → score = the candidate's preference, passed when the candidate
  wins or ties.

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

Implementation notes (`src/agentlens/suite/`, #20, FR-S1):
- **Model**: `Suite` (pydantic, `extra="forbid"`): `name`, `agent` (`"pkg.mod:callable"` or, for
  Python suites, the callable itself; serialised as `module:qualname`), `description`, `model`,
  `repetitions` (>= 1), `concurrency` (>= 1), `timeout_s`, `imports` (modules imported before
  checking evaluators, e.g. custom `@agentlens.evaluator` definitions), `metadata`, `defaults`
  (`SuiteDefaults`: `evaluators`, `tags`), `cases` (>= 1, unique ids). `cases` keep what was
  declared; `resolved_cases()` returns copies with the defaults merged in, which is what the
  runner (#21) executes.
- **Defaults merge** (`merge_evaluators`): defaults first, then the case's own; a case evaluator
  *replaces* every default with the same name (so `{max_steps: 4}` tightens a default
  `{max_steps: 8}` instead of running both). Tags: defaults then own, de-duplicated.
- **Errors with line context**: the YAML is composed into a node tree with `yaml.SafeLoader`, every
  node's start mark recorded by path in a `SourceMap`, then constructed from the same tree (one
  parse). Pydantic error locations map back to `file:line:column` (longest recorded prefix; the
  key position for unknown fields, with a "did you mean" hint). All problems (schema, duplicate
  keys/ids, unknown evaluators with suggestions, bad parameters, failing `imports`) are collected
  into one `SuiteError` whose `issues` are `SuiteIssue(message, loc, file, line, column)`, sorted
  by position. Evaluator specs are validated by building them from the registry (built once per
  distinct spec, then discarded); `check=False` skips that.
- **Datasets**: `cases: path` (relative to the suite file) or `load_cases(path)` read `.jsonl`
  (row positions), `.yaml`/`.yml`, `.json` (read as YAML to keep positions); the dataset's source
  map is mounted under `cases`, so errors point into the dataset file.
- **Agent import** (`resolve_agent`, `Suite.load_agent`): entry-point syntax, attribute paths
  (`mod:Agent.run`); the cwd and the suite's folder are *appended* to `sys.path` (installed code
  wins on a clash, a stray `json.py` cannot shadow the stdlib). Failures → `AgentImportError`
  (a `SuiteError`) distinguishing missing module, module raising on import, missing attribute,
  not callable. The agent is imported lazily, not by `load_suite`, so suites can be validated
  without importing the code under test.
- **For the runner (#21)**: `is_async_agent(fn)` / `Suite.agent_is_async()` (async functions,
  partials, objects with `async def __call__`); `Suite.content_hash()` = sha256 of the canonical
  JSON (sorted keys) of the suite with resolved cases, excluding `description`, `concurrency`
  and the `defaults` spelling, so it changes exactly when what runs changes (including an
  external dataset's contents) — recorded in the experiment config as the suite hash.

Implementation notes — runner and experiments (`suite/runner.py`, `suite/experiment.py`, #21,
FR-S2/FR-S3):
- **API**: `run_suite(suite | path, *, store=None, name=None, model=None, repetitions=None,
  concurrency=None, timeout_s=None, config=None, registry=None, redactor=DEFAULT_REDACTOR,
  progress=None) -> ExperimentRun` is a blocking wrapper (`asyncio.run`) over `arun_suite`.
  Overrides win over the suite's settings. Evaluators are built for every case *before* the
  experiment is created, so a bad spec fails before anything runs or is stored.
- **Execution**: one task per (case, repetition), all gated by one `asyncio.Semaphore(concurrency)`;
  a slot covers the agent call and its evaluation (so LLM judges are bounded too). `async def`
  agents are awaited; sync agents go through `asyncio.to_thread`, which copies the context, so
  `wrap_llm`/`wrap_tool` spans land in the right trace in both cases. Each run is
  `agentlens.run(suite.name, input=case.input, metadata={case_id, repetition, suite, experiment})`;
  the output is converted with the recorder's JSON conversion.
- **Agent call convention**: `agent(case.input)`, plus `model="provider:name"` when the agent has a
  parameter named `model` and a model is known (`model=` > `suite.model` > `$AGENTLENS_MODEL`), so
  the agent can do `get_client(model)`. No env mutation (it would leak across concurrent runs).
  Alternative considered: always passing a run context object — more API to freeze for little gain.
- **Failure isolation**: agent exceptions are recorded on the trace by `agentlens.run` and
  suppressed; the run is still evaluated (e.g. `no_error` fails) and saved. `timeout_s` wraps the
  agent call in `asyncio.wait_for` → `RunTimeoutError` recorded as `trace.error`; async agents are
  cancelled, sync threads cannot be killed and finish in the background (their late spans are
  dropped with a warning). Store errors are *not* suppressed: they abort the suite (remaining tasks
  cancelled), since a silently incomplete experiment would be worse.
- **What is evaluated and stored**: the redacted copy handed to the trace sink, so stored traces
  and stored scores agree and re-evaluating from the store reproduces them. If the redactor fails,
  a content-free trace (ids, timings, metadata) with `error = REDACTION_FAILED` is stored instead.
- **Results**: `ExperimentRun(experiment, suite, cases: list[CaseResult])` with `runs`, `total`,
  `passed`, `errors`, `pass_rate`, `case(id)`; `CaseResult(case, runs)` with `passed`, `pass_rate`,
  `errors`, `mean_scores()` per evaluator; `RunResult(case_id, repetition, trace, eval_results)`
  with `passed` (no error and all evaluators passed), `error`, `timed_out`. Ordered by case then
  repetition regardless of completion order. `progress(result, completed, total)` after each saved
  run; its exceptions are logged.
- **Experiment config** (`experiment_config`): `git_sha` (`git rev-parse HEAD` in the suite's
  folder), `git_dirty` (tracked changes only; both `None` without git/repo, 5 s timeout), `model`,
  `suite_hash`, `agent`, effective `repetitions`/`concurrency`/`timeout_s`, `agentlens_version`,
  `suite_metadata`, then user `config` (wins on clash). Default name `<suite>-YYYYmmdd-HHMMSS` (UTC).
- **Baselines**: `set_baseline(store, id | name | Experiment)` unmarks every other baseline of the
  same suite *first*, then marks the target (an interruption leaves zero baselines, never two);
  `unset_baseline`, `get_baseline(store, suite name | Suite)` (most recent if several were marked
  by hand), `find_experiment` (id, else most recent with that name) → `ExperimentNotFoundError`.
  Built on `save_experiment` upserts, so any `Store` backend works.

Implementation notes — pytest plugin (`integrations/pytest_plugin.py`, #25, FR-S6):
- **Registration**: entry point `pytest11: agentlens = "agentlens.integrations.pytest_plugin"`, so
  pytest loads it in every project that installs AgentLens. It must be inert: it only adds
  options and the `agentlens_run` fixture; the suite runner and store are imported lazily inside
  the hooks (a test checks that loading the plugin does not import them).
- **Collection**: `--agentlens SUITE` (repeatable, `action="append"`) resolves each file against
  the invocation dir (missing → `UsageError`) and appends it to `config.args` in
  `pytest_configure`, unless a positional argument already names it (`suite.yaml::case_id` keeps
  narrowing). `pytest_collect_file` claims exactly those files (`AgentlensSuiteFile`), so suites
  sit next to normal tests and `-k`/`-m`/node ids/`--collect-only`/`--junitxml` work unchanged. A
  `SuiteError` is a collection error rendered as its `file:line:column` issues (no traceback).
  Alternative considered: claiming every `*.yaml` under the test paths — too greedy for projects
  with unrelated YAML; creating items in `pytest_collection_modifyitems` — loses collection-error
  reporting and node-id selection.
- **Granularity**: one item (`AgentlensCase`) per case, node id `<suite file>::<case_id>`, report
  header `agentlens <suite>::<case_id>` (`reportinfo`); suite name and case tags are extra
  keywords for `-k`. A case passes when every repetition passed (repetitions measure flakiness,
  so a flaky case should fail). Alternative: one item per case × repetition — noisier output and
  inflated test counts for the same information, which the report already lists per repetition.
- **Execution**: the first selected case of a suite runs **all selected cases** of that file with
  `run_suite` (a copy of the suite with the deselected cases removed), so a pytest session gives
  one experiment per suite with the suite's concurrency, and deselected cases never run. Errors
  that stop a suite (agent import, store) become `SuiteRunError` on every case of that suite.
  `--agentlens-repetitions` overrides the suites' repetitions; `--agentlens-db PATH` opens one
  `SQLiteStore` for the session (closed in `pytest_unconfigure`). Per-session state lives in
  `config.stash`, never in module globals (pytester and xdist run several sessions per process).
- **Report**: `format_case_result(result, suite=)` — case input/expected, then per failing
  repetition the agent error (or output) and each failed `EvalResult` as `name: score S: reason`;
  values are JSON, truncated at 200 characters. Used by `AgentlensCase.repr_failure` and by
  `assert_passed(result)`, which raises `CaseFailedError` (an `AssertionError`). Junit properties:
  `agentlens_experiment`, `agentlens_pass_rate`, `agentlens_score.<evaluator>`.
- **Python tests**: fixture `agentlens_run(agent, case, *, evaluators=(), repetitions=None,
  model=None, timeout_s=None) -> CaseResult` runs a one-case `Suite` named after the test
  (`evaluators` are its defaults, so a case evaluator with the same name wins). Synchronous
  (`run_suite` starts its own loop). A `@pytest.mark.agentlens(suite=...)` marker was not added:
  the fixture covers the Python side with less API.
- **This repository's own run** uses `-p no:agentlens` (pyproject `addopts`): an auto-loaded
  plugin imports `agentlens` before pytest-cov starts measuring, which would hide every
  module-level line. The plugin's tests load it in `pytester` runs. Because the distribution
  now has a `pytest11` entry point, pytest marks the `agentlens` package for assertion
  rewriting; `runpy` cannot run modules through that import hook, so the `python -m agentlens`
  test removes it from `sys.meta_path` (no effect outside pytest).

Implementation notes — example support agent (`examples/support_agent/`, #29, FR-D1):
- **Agent**: `run_agent(question, model=None, *, llm=None, prompt_version="v1", max_iterations=8)`
  is a plain loop over any `LLMClient` (`LLMRequest(messages, tools=TOOL_SPECS, temperature=0)`;
  tool calls are executed and appended as `tool` messages; the first answer without tool calls is
  returned; after `max_iterations` model calls it returns a fixed apology instead of raising, so
  `max_steps` reports the runaway loop). The client is `llm=`, else `build_llm(model)`: `model=`
  (passed by the runner) > `$AGENTLENS_MODEL` > `mock:support`; any `mock` spec builds the scripted
  policy, other specs go through `get_client`. Wrapped with `wrap_llm`; tools with `wrap_tool`.
- **Tools** (`lookup_order`, `issue_refund`, `track_order`, `update_address`, `escalate_to_human`,
  `search_faq`) are closures over a per-run deep copy of a fixed in-memory database (`TODAY` is a
  constant), so runs are isolated under concurrency and deterministic. They enforce only data
  invariants and return `{"error": ...}` for business errors; `call_tool` turns an unknown tool,
  bad arguments or an exception into an error result (the exception stays on the span).
- **Policy in the prompt** (`prompts/system_<PROMPT_VERSION>.txt`): refund window and $200 limit,
  escalation. The tools deliberately do not enforce them, so a prompt edit can regress behaviour
  (FR-D2, #30).
- **Mock policy** (`mock_policy.support_policy`, a `MockLLM` callable script): keyword intents per
  clause (order number carried forward; escalation wins), each intent a generator state machine
  replayed over the tool results already in the conversation; the first step without a result is
  the next `ToolCall` (`id=call_<n>`), otherwise the final answer is one templated sentence per
  intent. Rules (`up to $N`, `within N days of delivery`) are parsed from the system prompt, so the
  mock reacts to prompt changes like a model would. Alternative rejected: per-case list scripts
  (deterministic but they would not react to prompt or code changes, defeating regression testing).
- **Dataset**: 27 cases in `cases.yaml`, shared by `suite.yaml` (deterministic: `no_error`,
  `max_steps: 6`, `tool_sequence` in `ordered`/`subset` mode with `forbidden` tools for refusals,
  case-insensitive `contains`/`regex` on facts, so real models can pass too) and
  `suite.judges.yaml` (adds `llm_rubric`; needs a judge model, so not run in CI). Agent path
  `examples.support_agent:run_agent`, resolved from the cwd: run from the repository root
  (`examples/__init__.py` makes it a regular package). Tests: `tests/examples/test_support_agent.py`
  (100% offline pass, exact tool lists per case, concurrency isolation, prompt-change regression);
  a `live` test runs the dataset with `$AGENTLENS_LIVE_SUPPORT_MODEL` (>= 70% pass).

## 7. Comparison & gates
- Metrics per case: pass rate, mean score per evaluator, cost, latency, steps.
- Aggregate delta with **paired bootstrap** (10k resamples over cases) → 95% CI.
  Regression = CI entirely below 0 for "higher is better" metrics.
- Gates YAML: `success_rate: {min: 0.9}`, `cost_usd: {max_delta_pct: 20}`, `regressions: {max: 0}`
  (see the gates implementation notes below).

Implementation notes — comparator (`src/agentlens/compare/`, #22, FR-S4, ADR 0005):
- **API**: `compare_experiments(store, baseline, candidate, *, resamples=10_000, seed=0,
  confidence=0.95) -> Comparison` (each side an id, a name — most recent with that name, via
  `find_experiment` — or an `Experiment`); `compare_runs(baseline, candidate, ...)` does the same on
  in-memory data (`ExperimentRun` or any iterable of `RunResult`), so it is testable without a
  store; `experiment_runs(store, experiment)` rebuilds `RunResult`s from stored traces (`case_id`
  and `repetition` metadata written by the runner) plus `list_eval_results`. Traces without a
  string `case_id` are skipped with a warning.
- **Metrics** (names are what gates will reference): `pass_rate` (a run passes iff no error and all
  its eval results passed — the same rule as `RunResult.passed`), `score.<evaluator>` (mean score),
  `cost_usd`, `latency_ms`, `steps` (`Trace.n_steps`). Per case each is the mean over the case's
  repetitions; `cost_usd`/`latency_ms` average only runs where the value is known (unknown cost is
  not free), and a case where no run has one has no value for that metric.
- **Pairing**: by case id over cases present in both experiments (others in `added_cases` /
  `removed_cases`, never in a delta). For each metric the pairs are the shared cases where *both*
  sides have a value; `n` says how many. `MetricDelta.baseline`/`candidate` are means over those
  pairs, so `delta = candidate - baseline` is exactly the mean paired difference the CI is about;
  `delta_pct = delta / |baseline| * 100` (`None` when the baseline is 0). No pairs → `no_data`.
- **Bootstrap** (`paired_bootstrap(diffs, *, resamples, confidence, seed)`): percentile method over
  resampled means, quantiles interpolated like NumPy's default `linear`. Stdlib only:
  `random.Random(seed).choices(diffs, k=n)` per resample (10k × 100 cases ≈ 60 ms); no NumPy
  dependency for one statistic. The indices drawn depend only on the seed, `n` and `resamples`,
  never on the values, so the same seed is reproducible and shifting all diffs shifts the CI by
  the same amount (property-tested with Hypothesis together with "zero diffs → [0, 0]", "CI
  contains the observed mean", "CI within the data range"). Constant diffs short-circuit to a
  point interval, and the interval is widened to contain the observed mean (only ever a
  floating-point-rounding adjustment). Every metric uses the same seed. Alternatives rejected:
  BCa intervals (more accurate on skewed data but harder to explain in a PR comment, and the
  jackknife step is extra code), a t-test (assumes normal diffs; pass rates per case are bounded
  and lumpy).
- **Classification** (`classify(direction, ci_low, ci_high)`): higher-is-better metrics
  (`pass_rate`, scores) regress when `ci_high < 0`; lower-is-better (`cost_usd`, `latency_ms`,
  `steps`) when `ci_low > 0`; the mirror case is an improvement; a CI touching 0 is `unchanged`.
  One case gives a point CI, so it can be flagged: the ~20-case guidance of ADR 0005 is the
  user's (and gates', via `n`) responsibility.
- **Per case** (`CaseComparison`): `metrics[name] = CaseMetric(baseline, candidate, delta)`,
  `baseline_runs`/`candidate_runs`, `status` from the case's pass rate (any drop = `regression`,
  e.g. passed in the baseline and fails in the candidate; any rise = `improvement`) and
  `regressed_evaluators` (evaluators whose pass rate on that case dropped). This is a "where to
  look" list; whether a change is beyond noise is what the aggregate CIs decide.
- **`Comparison`** (pydantic, round-trips through `model_dump_json`/`model_validate_json`):
  `baseline`/`candidate` experiments, `metrics` (`pass_rate`, `score.*` sorted, `cost_usd`,
  `latency_ms`, `steps`), `cases` (baseline order), `added_cases`, `removed_cases`, `resamples`,
  `confidence`, `seed`; helpers `metric(name)`, `case(id)`, `regressions`, `improvements`,
  `case_regressions`, `case_improvements`, `has_regressions`. Gates (#23) and reporters (#26-#28)
  consume it.
- **Pairwise judge** (§5): not wired in yet. It needs the case and an LLM, so it belongs to the
  CLI/gates layer (#23/#24), which can call `PairwiseJudge.compare` per shared case on one
  baseline and one candidate trace (`experiment_runs` gives both) and add the results as a
  `score.pairwise` metric.

Implementation notes — gates (`src/agentlens/compare/gates.py`, #23, FR-S5):
- **File format**: a flat mapping `metric: {rule: value}` plus an optional reserved `options:`
  key (`min_cases`, default 20; `on_missing`, default `fail`). Metrics are the comparison's
  (`pass_rate`, `score.<evaluator>`, `cost_usd`, `latency_ms`, `steps`) with the aliases
  `success_rate`, `cost`, `latency`, plus two counts that need a comparison: `regressions`
  (aggregate metrics with `status == regression`, i.e. CI entirely on the bad side) and
  `case_regressions` (cases whose pass rate dropped). Two names for the same metric is an error.
- **Rules** (all inclusive, compared with a 1e-9 relative tolerance so `27/30` meets `min: 0.9`):
  `min`/`max` on the candidate value; `min_delta`/`max_delta` on `candidate - baseline`;
  `min_delta_pct`/`max_delta_pct` on `delta_pct`. Delta rules use the point estimate, not the CI:
  "cost may grow at most 20 %" is a budget, while "beyond noise" is what `regressions` checks.
  Counts accept only `min`/`max`. Each bound gives its own `GateResult`.
- **Where values come from**: `evaluate_gates(gates, comparison=None, run=None)`. With `run`
  (an `ExperimentRun` or runs), `min`/`max` use `run_metrics(run)` — the candidate over *all*
  its cases, averaged like the comparator (per case over repetitions, then over cases). Without
  it they use the comparison's candidate means over the paired cases. Neither → `ValueError`.
- **Missing data** (metric absent, `no_data`, percentage change with a 0 baseline and a non-zero
  delta): the gate **fails** with the reason by default — a gate that silently stops checking
  (e.g. a renamed evaluator, costs no longer recorded) is worse than a red build — and
  `on_missing: skip` (per gate or in `options`) skips it. Not *applicable* is different: delta
  and count gates without a comparison are skipped and a note says how many, so
  `agentlens run --gates` can gate absolute values before any baseline exists.
- **Report**: `GateResult(name, metric, rule, threshold, actual, passed, skipped, reason)`,
  `GateReport(passed, results, notes)` with `exit_code` (`EXIT_OK` 0 / `EXIT_GATE_FAILED` 1;
  the CLI (#24) owns any other code, e.g. for usage/config errors), `failures`, `skipped`,
  `summary()`. Pydantic, so it round-trips through JSON for reporters (#26-#28). Notes never fail:
  fewer than `min_cases` cases (ADR 0005's ~20), cases only in one experiment.
- **Loading**: `load_gates(path)` reuses the suite loader's position-tracking YAML parser
  (`suite.source.load_yaml`), so every problem (unknown metric/rule with "did you mean", bad
  value, `min > max`, delta rules on counts, duplicate keys, YAML syntax) is collected into one
  `GatesError` (a `SuiteError` subclass, same `SuiteIssue` formatting) at its `file:line:column`.
  `Gates.from_dict` does the same without positions; `Gates(rules=...)` validates in Python.

## 8. CLI
`agentlens run`, `agentlens compare`, `agentlens report`, `agentlens baseline set`,
`agentlens traces ls/show`, `agentlens export/import`, `agentlens ui`.

Implementation notes (`src/agentlens/cli/`, `src/agentlens/report/`, #24):
- **Typer** (runtime dependency): commands are typed functions whose annotations are the
  argument parsing, help texts come from the same place, and `typer.testing.CliRunner` tests them
  in-process. Recent Typer vendors Click and brings `rich` (help rendering; the rich terminal
  report of #26 reuses it) and `shellingham`. Alternatives: `argparse` (no new dependency, but
  hand-written subcommand plumbing and help) and plain Click (decorators repeat what the types
  already say). Entry point `agentlens = "agentlens.cli:app"`, plus `python -m agentlens`.
  Rich's pretty tracebacks are disabled: they can print local variables, e.g. API keys.
- **Commands**: `run SUITE` (overrides `--name/--model/--repetitions/--concurrency/--timeout`,
  progress per run on stderr unless `-q`, `--gates` checks absolute rules via
  `evaluate_gates(run=...)`, `--set-baseline` marks the experiment unless a gate failed: a failing
  run must not become the reference); `compare --candidate X [--baseline Y]` where `Y` is an id,
  a name or the keyword `baseline` (default) = the marked baseline of the *candidate's* suite (an
  experiment literally named "baseline" is reachable by id); runs are loaded once and passed to
  `compare_runs` and, as `run=`, to `evaluate_gates`, so absolute gates see every candidate case
  exactly as `run --gates` does, while delta/count gates use the paired cases; `--json PATH`
  writes a `ComparisonReport` (`-` = JSON on stdout instead of text). `report TARGET` takes an
  experiment id/name or a JSON file (comparison or experiment report, re-rendered).
  `baseline set|unset|show`, `experiments ls`, `traces ls|show`, `export`/`import` (JSONL) are
  thin wrappers over the store and suite APIs. `experiments ls` was added (not in the list above)
  because names and ids have to be discoverable for `compare`/`report`.
- **Global options**: `--db PATH` / `$AGENTLENS_DB`, default `.agentlens/agentlens.db`; commands
  that only read require the file to exist (a typo in the path is an error, not a new empty
  database); `run` and `import` create it. `--version`.
- **Exit codes**: `0` ok, `1` a gate failed (`GateReport.exit_code`), `2` usage/configuration
  error — Click's own code for bad arguments, reused for `SuiteError`/`GatesError`/
  `AgentImportError` (printed with every `file:line:column` issue), unknown experiment/trace,
  `StoreError`, `JSONLFormatError`, unreadable/unwritable files, a corrupt database, an
  unavailable report format. Errors are one `error: ...` block on stderr; anything else is a bug
  and keeps its traceback. Suite and gates files are loaded before the database is opened, so a
  bad config runs nothing and creates nothing.
- **Reports** (`agentlens.report`): models `ExperimentReport.from_runs(experiment, runs, gates=)`
  (totals, `run_metrics` means, `CaseSummary` per case with failed evaluators) and
  `ComparisonReport(comparison, gates)`; `load_report(path)` reads either (or a bare
  `Comparison`). `render(report, fmt)` dispatches on a registry keyed by `ReportFormat`
  (`text`, `rich`, `json`, `markdown`, `html`); plain text has no colours (same in a terminal, a
  CI log or a file); the rich terminal report is #26 (below). #27 (Markdown) and #28 (HTML) plug in with
  `register_renderer`, so the CLI's `--format` needs no change; asking for a format with no
  renderer is exit 2 naming the issue that brings it.
Implementation notes — rich terminal report (`src/agentlens/report/terminal.py`, #26, FR-C1):
- **Format `rich`** registered like the others (`ReportFormat.RICH`, `RichRenderer`), so
  `render(report, "rich")` returns the report with ANSI colours; `RichRenderer(width=, color=)`
  (default width: the terminal's, else 100). The building blocks are public too:
  `comparison_renderable(report)` / `experiment_renderable(report)` return rich renderables to
  print on any `rich.console.Console` (e.g. one that records, for an export).
- **Comparison view**: header (baseline/candidate, paired/added/removed cases, bootstrap method);
  a metrics table with `n`, baseline, candidate, delta (absolute, with `↑`/`↓` for the way the
  value moved) and delta %, the CI and the status; a one-line verdict; case regression and
  improvement tables (pass rate `1 → 0`, runs, regressed evaluators); added/removed cases; a
  gates panel (green/red border, `PASS`/`FAIL`/`SKIP` per bound with rule, actual value and
  reason, then the notes). **Colour follows the metric's `Direction`**, the arrow follows the
  value: cost `↑` is red, pass rate `↑` green, so "up" never has to be read as "good". The
  metric name carries `↑`/`↓` for the direction that is better (explained in the caption).
  Colour never carries meaning alone (a requirement for colour-blind users and `NO_COLOR`):
  regressions also say `REGRESSION` (uppercase, as in plain text), improvements `improvement`,
  missing values `-`.
- **Experiment view**: header, metrics (pass rate as a percentage, mean score per evaluator,
  cost, latency, steps — the `run_metrics` means), a *failures* table with only the cases that
  failed a repetition (passed/runs, errors, failed evaluators) or "all N cases passed", and the
  gates panel. The plain-text report still lists every case.
- **When the CLI uses it**: `run`, `compare` and `report` print `rich` when stdout is an
  interactive terminal (`isatty`) and plain `text` when it is piped/redirected, `$NO_COLOR` is
  non-empty (no-color.org) or `$TERM=dumb` (`cli._common.auto_format`). `$FORCE_COLOR` is
  deliberately *not* a reason to switch: CI sets it, and CI logs and scripts parsing the output
  should keep the stable plain text. `report --format rich|text|json` overrides; `report -o FILE`
  without `--format` writes plain text.
- **Trailing padding** (rich pads every line to the width) is stripped from the string output,
  so copied or diffed reports have no trailing spaces.
- **Snapshot tests** (`tests/report/test_terminal.py`, `tests/report/snapshots/*.txt`): fixed
  experiments and runs, rendered on `Console(record=True, width=100, color_system=None)` and
  exported as text (trailing spaces stripped), compared with the committed files.
  `AGENTLENS_UPDATE_SNAPSHOTS=1 uv run pytest tests/report/test_terminal.py` rewrites them
  (review the diff); a missing snapshot is written and the test fails once. Colour rules are
  tested on the cells' styles and on the ANSI output. No new test dependency: `syrupy` would
  add snapshot management we need for two files.
- **Dependency**: `rich` is now declared (`rich>=13.0`) since `agentlens.report` imports it
  directly; it was already installed through Typer, so nothing new is pulled in, but relying on
  a transitive dependency would break if Typer dropped or loosened it.
- **Replay divergence** (`replay_divergence` trace metadata) is not shown: neither report model
  carries traces. It can be added to `CaseComparison`/`CaseSummary` when replay-based comparisons
  land in the CLI.

Implementation notes — Markdown report and GitHub Action (`report/markdown.py`,
`integrations/github.py`, `cli/github.py`, `action.yml`, #27, FR-C3, ADR 0006):
- **Format `markdown`** (`MarkdownRenderer(marker=, max_cases=)`, registered like the others):
  GitHub-flavoured and compact for a PR comment. Comparison: a hidden marker line, a verdict
  headline (`### ❌ AgentLens: gates failed · 3 metrics regressed · 1 improved · 3 cases
  regressed`; ✅/❌ always next to words, so it reads in notifications and screen readers), one
  context line (candidate vs baseline, suite, paired/added/removed cases), the metrics table
  (baseline, candidate, Δ, Δ%, CI, status; `↑`/`↓` for the better direction), the gates table
  (pass/fail/skip, rule, actual, reason) with the notes as a quoted list, case regressions and
  improvements each in a collapsed `<details>` block, added/removed case ids, and a `<sub>`
  footer with experiment names, ids, git shas and the bootstrap method. Experiment reports have
  the same shape (metrics, gates, failing cases collapsed). Headline verdict: the gates' when
  there are gates, else "no metric regressed" (comparison) / "every run passed" (experiment).
- **Escaping**: user text (names, case ids, reasons) has `&`, `<`, `>`, `|`, `\`, `*`, `[`, `]` and backticks escaped and newlines
  flattened, so it cannot break a table or inject HTML; ids go in code spans (backticks replaced,
  `|` escaped, which GitHub honours inside table code spans). Case tables and id lists stop at
  `max_cases` (50) with "…and N more" (a GitHub comment holds 65 536 characters; the poster
  truncates as a last resort).
- **Marker**: `marker_comment(key)` = `<!-- key -->`, key restricted to `[A-Za-z0-9_.:/@+=-]` so it
  cannot close the comment early. Default `agentlens-report`; `report --format markdown --marker
  KEY` and the action's `marker` input set it (one per suite when a PR runs several). Matching is
  on the exact `<!-- key -->` string, so `agentlens-report` does not match `agentlens-report:x`.
- **Comment upsert** (`upsert_pr_comment(body, *, context, marker, client=None)`): lists the PR's
  issue comments (`per_page=100`, following `Link: rel="next"`), PATCHes the oldest one containing
  the marker (skips the request when the body is identical: `unchanged`), else POSTs; returns
  `CommentResult(action, comment_id, url)`. Plain `httpx` (already a dependency) with the REST
  headers (`Bearer` token, `application/vnd.github+json`, `X-GitHub-Api-Version: 2022-11-28`);
  errors become `GitHubError` with GitHub's `message` and, for 401/403/404, a hint about
  `pull-requests: write` and read-only fork tokens; the token never appears in errors or `repr`.
  `GitHubContext.from_env()` reads `GITHUB_TOKEN` (or `GH_TOKEN`), `GITHUB_REPOSITORY`,
  `GITHUB_EVENT_PATH` (`pull_request.number`, `issue.number` or `number`) and `GITHUB_API_URL`
  (Enterprise). No retries: one comment per job, and a failed comment must not fail the check.
- **CLI**: `agentlens github comment --file F [--marker --pr --repo]` (exit 2 on any failure);
  `run --json PATH|-` writes the `ExperimentReport` with its gate verdict (status lines move to
  stderr with `-`), so a run without a baseline can be rendered with its gates like a comparison.
- **Action** (`action.yml`, composite): setup-uv → a venv in `$RUNNER_TEMP` with AgentLens (the
  action's own source by default, so the action ref pins the version; `agentlens-version` for
  PyPI) and the `install` arguments → baseline → candidate → report → comment → exit. Baseline:
  `baseline-ref` (default the PR base sha) is fetched (`--depth=1`) and checked out in a
  temporary `git worktree`, the suite runs there as `baseline-<sha12>`, then on the PR as
  `pr-<n>`, and `compare --gates --json` decides the exit code; when the suite does not exist at
  the base (a new suite) or, with `baseline-ref: ""`, there is no database, the candidate is run
  with `--gates --json` (absolute gates only, delta gates skipped and noted). Exit 2 from any
  step fails it immediately (misconfiguration); the gate code (0/1) is saved as an output, the
  Markdown goes to `$GITHUB_STEP_SUMMARY` and to the PR comment (a failed comment is a warning:
  fork PRs), and only then does the last step exit with it. `NO_COLOR` keeps the log plain.
- **This repo's CI** (`.github/workflows/agentlens.yml`, `pull_request`, `pull-requests: write`)
  runs `uses: ./` on `examples/ci_demo` (a `MockLLM` FAQ agent, 12 cases, gates on pass rate,
  pass-rate delta, case regressions and steps; latency is measured and therefore not gated).
  `tests/examples/test_ci_demo.py` checks the demo passes its gates and that the action's inputs,
  outputs and the workflow are wired consistently (the action itself only runs on GitHub; its
  script was exercised locally against throwaway repositories for the baseline, new-suite and
  database modes).

Implementation notes — HTML report (`src/agentlens/report/html.py`, `report/templates/`, #28, FR-C2):
- **Format `html`** (`HtmlRenderer(runs=, baseline_runs=, title=)`, registered without runs;
  `render_html(report, *, runs, baseline_runs, title)`). Report models carry aggregates only, so
  the traces come from the runs passed alongside (`experiment_runs(store, exp)` or a `run_suite`
  result): the renderer keeps the models unchanged (JSON reports stay small, no trace data in PR
  comments) and the page degrades to aggregates when no runs are given. Alternative considered:
  adding traces to `ComparisonReport` — it would bloat every `--json` file and the Markdown input.
- **CLI**: `report EXPERIMENT --format html [-o FILE]` loads the experiment's runs;
  `report FILE.json --format html` looks the report's experiments up by id in the database when
  it exists (a note on stderr says which traces are missing: a CI artifact read elsewhere);
  `compare ... --html FILE` writes it next to the usual output, with both sides' traces.
- **Jinja2** (new runtime dependency, `jinja2>=3.1`, brings `markupsafe`): templates as package
  data (`report.html.j2`, `_macros.html.j2`), `autoescape=True` and `StrictUndefined`, so every
  value from users is HTML-escaped by default and a missing variable is an error, not an empty
  string. Alternatives: `string.Template`/f-strings (no autoescaping: every interpolation would
  need a manual `escape()`, easy to forget once and get an XSS) and building the DOM in Python.
  `report.css`/`report.js` are packaged files inlined verbatim (not parsed as templates).
- **Self-contained**: one file, inline `<style>`/`<script>`, no `src`/`href` to the network, no
  web fonts (system font stack), no CDN — it opens offline, as a CI artifact or from email.
  Light/dark via CSS variables and `prefers-color-scheme`.
- **Page**: verdict (gates, else regressions), experiment metadata (suite, model, git sha), summary
  cards (pass rate, cost, latency, steps), metrics table (n, baseline, candidate, Δ, Δ%, CI text
  and a small CI-vs-0 bar, status), gates (pass/fail/skip, rule, actual, reason, notes), cases as
  `<details>` (regressions first, then improvements, unchanged, added, removed; filter chips and
  a search box) opening to baseline/candidate run cards side by side (pass/fail, latency, cost,
  steps, tokens, each evaluator's score and reason, output preview, error). Per-case latency is
  shown but not coloured (machine noise). Experiment reports list failing cases first.
- **Trace viewer**: a modal `<dialog>` built by ~300 lines of vanilla JS from the embedded JSON,
  only when a trace is opened (large experiments do not create thousands of DOM nodes up front).
  Tree = the run plus its spans nested by `parent_id` (start-time order, computed in Python with
  offsets/durations in ms relative to the earliest start, so no date parsing in JS); kind
  badges; waterfall bar per span; collapse/expand per node and for all; clicking a span shows
  timing, tokens, cost, error, and input/output/attributes pretty-printed with light syntax
  colouring; errors red (also on the row). Unfinished spans run to the end. Deep links:
  `report.html#trace-<id>` opens a trace.
- **Escaping**: Jinja autoescaping for the markup; the traces are JSON in
  `<script type="application/json">` with `<`, `>`, `&`, U+2028/9 written as `\uXXXX`
  (`script_json`), so `</script>` or `<!--` in a trace cannot leave the block; non-finite floats
  become `null`. The JS writes only through `textContent`/`createTextNode`. Tests render
  `</script><script>alert(1)</script>` in every user field; it was also checked in Chromium (no
  dialog, text shown literally).
- **Tests** (`tests/report/test_html.py`): key content, a structural check with `html.parser`
  (every element closed in order, exactly two scripts), self-containment (no `src`, only `#`
  hrefs, no `http(s):`, no `url(`/`@import`), the payload tree (nesting, order, offsets,
  unfinished spans, usage), escaping. The JS is not run in CI (no browser dependency); the
  screenshots in `docs/images/` are regenerated with `scripts/html_report_demo.py` +
  `scripts/screenshot_html_report.py` (Playwright, not a project dependency).

Implementation notes — local web UI (`src/agentlens/ui/`, `cli/ui.py`, #31, FR-C4):
- **Command**: `agentlens ui [--host 127.0.0.1] [--port 8765] [--db PATH] [--gates FILE]
  [--open/--no-open]` (`--db` on the command overrides the global one). Serves until Ctrl+C; a
  busy port or an invalid gates file exits 2. A missing database is *not* created: pages answer
  503 explaining how to create one, and the UI picks the file up as soon as it exists.
- **Stack: standard library server + server-rendered Jinja2, no new dependency.** The issue
  suggested FastAPI; for a local, single-user, read-only browser over one SQLite file, a router of
  ten regexes over `http.server.ThreadingHTTPServer` covers it, while FastAPI would add
  FastAPI + Starlette + uvicorn (+ anyio, h11, ...) to every install of a testing library (or an
  extra). Alternatives considered: FastAPI/Starlette (async, OpenAPI docs: not needed here), a
  SPA (React/Vite; a build step and a JS toolchain in a Python repo), loading JS from a CDN
  (breaks offline use and the CSP). Trade-off accepted: no async, no auto-generated API docs.
- **Split**: `UIApp.handle(method, target) -> Response` is pure (no sockets: tested directly,
  fast), `UIServer`/`make_server`/`serve` adapt it to HTTP. `make_server(app, port=0)` gives an
  ephemeral port, so tests and the CLI test run a real server in a thread without blocking.
- **Routes** (GET/HEAD only; anything else 405 with `Allow`, and `Connection: close` since the
  body is not read): `/traces` (filters `name`, `experiment` id-or-name, `case_id`,
  `meta=key=value` repeatable — the value is read as JSON when it is a JSON scalar, so
  `repetition=0` matches the number —, `errors=1|0`, `limit` 1-500 default 50, `offset`; one extra
  row is fetched to know whether there is a next page, no COUNT), `/traces/{id}`, `/experiments`
  (filters `name`, `suite`, `baseline`), `/experiments/{id-or-name}`, `/compare?baseline=&candidate=`
  (`baseline` defaults to the candidate's suite baseline, like the CLI; `resamples`, `seed`,
  `confidence` override; `--gates` applied when given; no parameters → redirect to the picker on
  `/experiments`), and `/api/traces`, `/api/traces/{id}`, `/api/experiments`,
  `/api/experiments/{id}`, `/api/compare` (the latter two are the same JSON as `report/compare
  --json`). Bad parameters → 400, unknown ids → 404 (HTML page or `{"error": ...}` under `/api`).
- **Reuse of the HTML report**: the experiment and comparison pages *are* `HtmlRenderer` pages
  (metrics with CI bars, gates, per-case side-by-side runs, modal trace viewer) through a subclass
  overriding the new `_chrome()` hook (navigation + footer). The UI's own pages extend a layout
  that inlines `report.css` + a small `ui.css` and import the report's macros (Jinja
  `ChoiceLoader`). The trace page embeds the same JSON payload and `report.js` with the dialog
  marked `data-inline` (rendered in the page, not modal). Trace list status: `error` (run or
  span), `fail` (an evaluator failed, names shown), `pass`, `ok` (not evaluated).
- **Security**: binds `127.0.0.1` by default. On a loopback bind, requests whose `Host` is not
  `127.0.0.1`/`localhost`/`[::1]` (with or without the port) get 400 — defence against DNS
  rebinding, where a web page's script reaches the local port under its own host name. Every
  response: `Content-Security-Policy: default-src 'none'; style-src 'unsafe-inline';
  script-src 'unsafe-inline'; img-src data:; form-action 'self'; base-uri 'none';
  frame-ancestors 'none'`, `X-Content-Type-Options: nosniff`, `Referrer-Policy: no-referrer`,
  `Cache-Control: no-store`. Escaping as in the report (autoescape, `script_json`, `textContent`);
  tests put `</script><script>alert(1)</script>` in trace names, inputs, metadata, case ids,
  experiment names and URLs. The UI never writes to the database (opening it applies pending
  migrations, as every command does).
- **Tests**: `tests/ui/test_app.py` (routing, filters, pagination, statuses, missing/corrupt DB,
  escaping), `tests/ui/test_server.py` (a real server on an ephemeral port with httpx: headers,
  HEAD, 405, Host check, keep-alive), `tests/cli/test_ui.py` (`serve_forever` patched to answer
  one request from a thread, so the command's wiring is exercised end to end without blocking).
  Screenshots `docs/images/web-ui{,-trace}.png` via `scripts/screenshot_web_ui.py` (Playwright).

- **Not yet**: the pairwise judge in `compare` (§7 note), user `config`
  values on `run` (Python API only).

## 8b. Documentation site (#33)
Implementation notes (`mkdocs.yml`, `docs/`, `scripts/make_demo_gif.py`, `scripts/mkdocs_hooks.py`):
- **mkdocs-material** + **mkdocstrings[python]** in a uv dependency group `docs` (not a runtime
  dependency or extra; not in `dev` either, so `uv sync` and the test jobs stay lean). CI builds
  it with `uv run --only-group docs mkdocs build --strict` (job `docs` in `ci.yml`): any warning
  (broken link or anchor, a page missing from the nav, a docstring parameter that is not in the
  signature) fails the PR. `docs.yml` builds again on `main` and deploys with
  `actions/upload-pages-artifact` + `actions/deploy-pages` (Pages source: GitHub Actions); site
  URL `https://ggbifulco.github.io/agentlens/`.
- **README vs site**: the README keeps the pitch, badges, a quickstart that runs offline from a
  fresh clone, a "three files" example, the architecture diagram, the GIF and screenshots, and
  links; the long sections moved (unchanged) to pages: `concepts/` (tracing, LLM clients, store,
  replay, evaluators, judges, suites, comparison & gates, reports), `cli.md`, `github-action.md`,
  `pytest-plugin.md`, `web-ui.md`, `opentelemetry.md`, plus `getting-started.md` and `index.md`.
  Links from `docs/` to files outside it (examples, workflows, tests) are absolute GitHub URLs,
  since mkdocs can only resolve pages inside `docs/`.
- **API reference**: one page per package (`::: agentlens.<pkg>`, members from `__all__`, Google
  docstring style). Docstrings keep their Sphinx roles (they read well in `help()`); the hook
  `scripts/mkdocs_hooks.py` strips `:class:`-style roles, `~pkg.` prefixes and `::` from the
  rendered API pages only. Four docstrings that documented several parameters on one line were
  split so griffe can match them to the signature.
- **Demo GIF**: no terminal recorder is needed. `make_demo_gif.py` runs the real commands in a
  pseudo-terminal (so the CLI picks its `rich` output), parses the ANSI text with `rich`
  (`Text.from_ansi`) and draws each frame with Pillow (typing, streaming progress, pauses;
  box-drawing characters drawn as lines so borders join). One fixed palette (theme colours blended
  over the background at 6 levels: what anti-aliased text is made of) keeps colours stable and
  lets Pillow store only each frame's changed area: ~225 frames in ~350 KB. Pillow is only
  pulled in ad hoc (`uv run --with pillow`), not a project dependency. Latency in the recording is
  real wall-clock time, so that row differs between regenerations.
- **Tests** (`tests/docs/test_docs.py`): every nav page exists and every `docs/**/*.md` is in the
  nav, the README's images exist and the GIF is < 2 MB, the quickstart names real files, the
  README's "three files" example is copied verbatim into a temp dir and run through the CLI (gates
  pass; then fail after a regression), and the hook rewrites API pages only.

## 9. Testing strategy
- Unit tests per module; property tests (hypothesis) for schema round-trips and bootstrap.
- Integration tests run the example agent end-to-end with `MockLLM` (no keys, deterministic).
- Optional live tests marked `@pytest.mark.live`, skipped without env vars.
