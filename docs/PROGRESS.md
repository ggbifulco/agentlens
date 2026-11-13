# Progress log

Append-only log of development sessions (newest at the bottom).

## 2026-11-11 — Project setup
- Wrote requirements, architecture, detailed design and ADRs 0001–0005.
- Created labels, milestones M0–M7 and 34 backlog issues (#1–#34).
- Added `AGENTS.md` (session procedure) and the Project board sync workflow.
- Next: #1 Project scaffolding.

## 2026-11-11 — #1 Project scaffolding
- Added `pyproject.toml` (hatchling backend, version read from `agentlens.__version__`,
  dev dependency group: ruff, mypy, pytest, pytest-cov, hypothesis, pre-commit) and `uv.lock`.
- `src/agentlens/__init__.py` with `__version__`, PEP 561 `py.typed` marker.
- Ruff lint/format config (markdown excluded: design docs hold pseudo-code), `mypy --strict`
  on `src/`, pytest with coverage gate 85% and a `live` marker skipped unless `AGENTLENS_LIVE_TESTS=1`.
- `.pre-commit-config.yaml`, `.gitignore` (and untracked a stray `.pyc`), `CONTRIBUTING.md`.
- Made `.github/scripts/project_sync.py` ruff-clean (line length only, no behaviour change).
- Note: no PR CI exists yet (it arrives with #2); verified locally.
- Next: #2 CI workflow.

## 2026-11-11 — #2 CI workflow
- Added `.github/workflows/ci.yml`: `lint` job (ruff check with GitHub annotations + ruff format
  check) and a `test` matrix on Python 3.11/3.12/3.13 (mypy --strict + pytest with the 85% gate
  from `pyproject.toml`). Triggers: push to `main`, every PR, manual dispatch.
- uv via `astral-sh/setup-uv` with cache; `UV_LOCKED=1` so a stale `uv.lock` fails CI.
- Coverage `coverage.xml` + HTML uploaded as `coverage-py3.X` artifacts; least-privilege
  `permissions`, `concurrency` cancel of outdated runs, job timeouts.
- CI badge in README, CI section in CONTRIBUTING.
- Unblocked #1: opened #36 stacked on `chore/1-project-scaffolding` so CI validated #1's code,
  then squash-merged #35 and retargeted #36 to `main`.
- Next: #3 Core schema models.

## 2026-11-12 — #3 Core schema models
- Added `agentlens.schema` (PR #37): `SpanKind`, `Usage`, `Span`, `Trace`, `EvaluatorSpec`, `Case`,
  `EvalResult`, `Experiment` on a shared `SchemaModel` base (`extra="forbid"`,
  `validate_assignment=True`); datetimes must be tz-aware and are normalised to UTC.
- `Trace` validates span/trace ids, unique span ids and parent links; derived metrics
  (`total_cost`, `total_tokens`, `latency_ms`, `tool_calls`, `n_steps`) are non-serialised
  properties. Added `error`, `start_time`, `end_time` to `Trace` (design doc updated).
- `EvaluatorSpec` parses/serialises the suite YAML shorthands (`no_error`, `{max_steps: 8}`, ...).
- `schema.ids.new_id()`: RFC 9562 UUIDv7 with a monotonic 12-bit counter; `id_datetime()`.
- New runtime dependency: `pydantic>=2.9` (+ `pydantic.mypy` plugin).
- Tests: unit tests + Hypothesis JSON round-trip for every model; reusable strategies in
  `tests/schema/strategies.py`. 64 tests, 100% coverage.
- Next: #4 Tracing context and recorder.

## 2026-11-12 — #4 Tracing context and recorder
- Added `agentlens.tracing` (PR #38): `run(name, input=, metadata=, sink=)` context manager
  yielding the live `Trace`; `@trace(kind=, name=)` (also bare) for sync and async functions;
  `span()` context manager; `current_trace()` / `current_span()`. All re-exported from `agentlens`.
- `Recorder` buffers spans on the trace (thread-safe append) and calls the sink once on finish;
  a sink is any `Callable[[Trace], None]` until the `Store` lands in #5.
- Context in two `ContextVar`s → correct nesting across `asyncio.gather`, `asyncio.to_thread` and
  `copy_context().run`; plain threads are intentionally not recorded. Nested runs are independent.
- FR-T5: recorder/sink failures are logged on `agentlens.tracing`, never raised; user exceptions
  (incl. `CancelledError`) are stored as `"Type: message"` and re-raised. Non-JSON values fall
  back to `repr()`. Generator functions are rejected by `@trace` (not supported yet).
- Benchmark test: ~40 µs/span overhead (budget 5 ms). 101 tests, 100% coverage.
- Next: #5 Store protocol and SQLiteStore.

## 2026-11-13 — #5 Store protocol and SQLiteStore
- Housekeeping: squash-merged PR #38 (#4 tracing, CI green, no maintainer feedback) per
  `MERGE_POLICY: auto`; the previous session's guard had blocked the self-merge.
- Added `agentlens.store`: `Store` protocol (runtime-checkable), `SQLiteStore`, `StoreError`,
  `SchemaVersionError`. Traces, experiments and eval results stored as JSON documents plus
  indexed filter columns; upsert semantics; `save_trace` works directly as a tracing sink.
- Filters: traces by name / experiment / metadata scalar equality (`json_each`, type-aware),
  newest first with `limit`; experiments by name / suite / baseline flag.
- Eval results: per-trace ordered list, replaced on save; `list_eval_results(experiment_id)`
  groups them by trace for the comparator.
- Migrations versioned in code (`MIGRATIONS` + `PRAGMA user_version`), one transaction each;
  newer databases are refused. Thread-safe single connection (RLock).
- No new dependencies (stdlib `sqlite3`). Tests on `tmp_path` databases + Hypothesis
  round-trip; 136 tests, 100% coverage.
- Next: #6 JSONL export/import of traces.

## 2026-11-13 — #6 JSONL export/import of traces
- Added `agentlens.store.jsonl`: `export_jsonl(store, path, name=, experiment_id=, metadata=,
  limit=)`, `import_jsonl(store, path, experiment_id=)`, plus store-free `write_jsonl` /
  `read_jsonl` and `JSONLFormatError` (a `StoreError` carrying `path` and 1-based `line`).
- Format: one `Trace` JSON document per line, oldest first; blank lines ignored. Only traces are
  exported (no experiment link / eval results) — documented in `docs/design.md` §2c.
- Atomic write (temp file + `os.replace`); import validates the whole file before saving
  (all-or-nothing on format errors) and is idempotent thanks to upserts.
- Works against the `Store` protocol, not `SQLiteStore`. No new dependencies.
- Tests: round-trip across two SQLite stores preserving every field, filters, ordering,
  atomicity, error reporting, Hypothesis write/read round-trip. 152 tests, 100% coverage.
- Next: #7 Redaction hook for secrets in traces.

## 2026-11-13 — #7 Redaction hook for secrets in traces
- Added `agentlens.tracing.redaction`: `Redactor` (= `Callable[[JsonValue], JsonValue]`),
  `PatternRedactor(patterns=, keys=, replacement=)`, `DEFAULT_REDACTOR`, `DEFAULT_PATTERNS`,
  `DEFAULT_SENSITIVE_KEYS`, `REDACTED` and `redact_trace(trace, redactor=None)`.
  `PatternRedactor`, `Redactor` and `redact_trace` are also re-exported from `agentlens`.
- `run(..., redactor=DEFAULT_REDACTOR)`: `Recorder.finish` hands the sink a redacted *copy*
  (trace input/output/metadata/error, span input/output/attributes/error); the live trace is
  untouched. `redactor=None` disables redaction.
- Default masks sensitive dict keys (`api_key`, `authorization`, `password`, `token`, ...,
  `*_api_key`/`*_secret`/`*_password`) and string patterns: `sk-` (OpenAI, Anthropic `sk-ant-`),
  `Bearer`, AWS `AKIA`/`ASIA`, GitHub `ghp_`/`github_pat_`, Google `AIza`, Slack, JWT, PEM
  private keys, `api_key=...` assignments.
- Fail closed + FR-T5: a failing redactor is logged and the trace is not persisted; user code
  never sees the error. Redaction runs once per trace, so per-span overhead is unchanged.
- No new dependencies (stdlib `re`). Tests with nested payloads incl. an end-to-end SQLite check;
  186 tests, 100% coverage.
- Next: #8 LLMClient protocol, request/response models and MockLLM.

## 2026-11-13 — #8 LLMClient protocol, request/response models and MockLLM
- Added `agentlens.llm`: `Message` (roles system/user/assistant/tool, `tool_calls` on assistant,
  `tool_call_id` on tool; `Message.system/user/assistant/tool` constructors), `ToolCall(id, name,
  arguments)`, `ToolSpec(name, description, parameters)`, `LLMRequest(messages, model,
  temperature, max_tokens, tools, response_format)`, `LLMResponse(content, tool_calls, usage,
  raw, model)` and the `runtime_checkable` `LLMClient` protocol (`complete` + async `acomplete`).
- `LLMRequest.canonical_json()` / `fingerprint()` (sha256) give the normalised request key that
  cassettes (#13) will hash; `LLMResponse.to_message()` feeds agent loops. Usage reuses
  `schema.Usage` (`None` = not reported).
- `MockLLM(script, model=, cycle=, default=)`: list (in order; `MockExhaustedError` when
  exhausted unless `cycle=True`/`default`), callable `(request) -> item`, or regex map matched
  against the last user message (`MockNoMatchError` without `default`). Items may be
  `LLMResponse`, text, `ToolCall` or `list[ToolCall]`. Deterministic usage from whitespace word
  counts at `cost_usd=0`; requests recorded in `.requests`; thread-safe.
- No new dependencies (async tested with `asyncio.run`, no pytest-asyncio). 243 tests,
  100% coverage.
- Next: #9 wrap_llm and wrap_tool auto-tracing.

## 2026-11-13 — #9 wrap_llm and wrap_tool auto-tracing
- Added `agentlens.tracing.wrappers`, re-exported as `agentlens.wrap_llm` / `agentlens.wrap_tool`
  (plus `TracedLLM`, `TOOL_NAME_ATTR` in `agentlens.tracing`).
- `wrap_llm(client, name=, provider=, include_raw=False) -> TracedLLM` (an `LLMClient`):
  `complete`/`acomplete` record `llm_call` spans — input = request JSON (defaults omitted),
  output = response JSON without `usage` (on `span.usage`) and `raw` (opt-in), attributes
  `client`, `provider`, `model`, `temperature`, `max_tokens`, `n_messages`, `n_tools`,
  `tool_names`, `response_format`, `request_hash`, `response_model`, `n_tool_calls`.
  Unknown attributes delegate to the wrapped client; re-wrapping is idempotent.
- `wrap_tool(fn, name=)` / `@wrap_tool` / `@wrap_tool(name=...)`: `tool_call` spans named after
  the tool, sync and async, any callable; never nests two spans when re-wrapped.
- Both are passthroughs outside `run`; errors recorded and re-raised; attribute/output building
  behind the FR-T5 guards (`_close(output_factory=)`, `Recorder.end_span(usage=, attributes=)`).
  ≈ 65 µs per `llm_call` span. `request_hash` + span order prepare cassettes (#13).
- No new dependencies. 276 tests, 100% coverage.
- Next: #10 OpenAI-compatible adapter (OpenAI, Ollama, vLLM, LM Studio).

## 2026-11-13 — #10 OpenAI-compatible adapter (OpenAI, Ollama, vLLM, LM Studio)
- Added `agentlens.llm.OpenAICompatClient(model=None, *, base_url=, api_key=, timeout=,
  max_retries=, retry_policy=, headers=, provider=, http_client=, async_http_client=)`:
  `complete`/`acomplete` on `{base_url}/chat/completions`; tools (`ToolSpec` → function tools,
  `tool_calls` decoded into `ToolCall` dict arguments, tool messages keep `tool_call_id`),
  `response_format` passthrough (JSON mode / JSON schema), `usage` → `schema.Usage`
  (`cost_usd=None` until pricing #12), `raw` = response body. Key from argument or
  `OPENAI_API_KEY`, none for local servers; never logged, in `repr`, `raw` or errors.
- Error hierarchy in `agentlens.llm.errors`: `LLMError` > `LLMHTTPError(status_code, body)` >
  `LLMRateLimitError`; `LLMConnectionError` > `LLMTimeoutError`; `LLMResponseError`.
- Shared private `agentlens.llm._http.JSONHTTPClient` (for #11 too) + public `RetryPolicy`:
  retries 408/429/5xx and transport errors, exponential backoff capped at `max_delay`,
  `Retry-After`/`retry-after-ms` honoured (capped), injectable `jitter`/`sleep`/`async_sleep`.
- New deps: `httpx` (runtime, both adapters, no vendor SDK per ADR 0002), `respx` (dev, mocked
  HTTP). Optional live test: `AGENTLENS_LIVE_TESTS=1 AGENTLENS_LIVE_OPENAI_MODEL=...`
  (+ `OPENAI_API_KEY` or `AGENTLENS_LIVE_OPENAI_BASE_URL`). 381 tests, 100% coverage.
- Next: #11 Anthropic adapter (reuse `agentlens.llm._http`).

## 2026-11-13 — #11 Anthropic adapter
- Added `agentlens.llm.AnthropicClient(model=None, *, base_url="https://api.anthropic.com",
  api_key=, max_tokens=1024, timeout=, max_retries=, retry_policy=, headers=,
  provider="anthropic", http_client=, async_http_client=)`: `complete`/`acomplete` on
  `{base_url}/v1/messages` via the shared `_http.JSONHTTPClient` (retries incl. 529);
  headers `x-api-key` (argument or `ANTHROPIC_API_KEY`) and `anthropic-version: 2023-06-01`.
  Also exported: `ANTHROPIC_VERSION`, `DEFAULT_ANTHROPIC_BASE_URL`, `DEFAULT_MAX_TOKENS`.
- Mapping: system messages → top-level `system`; `ToolSpec` → `name`/`description`/
  `input_schema`; assistant `tool_calls` → `tool_use` blocks; tool messages → `tool_result`
  blocks, consecutive results (and a following user message) merged into one user turn.
  Response `text` blocks → `content`, `tool_use` → `ToolCall`; usage `input_tokens` (+ cache
  tokens) / `output_tokens`, `cost_usd=None` until pricing (#12).
- `max_tokens` (required by Anthropic) falls back to the client default 1024. JSON
  `response_format` is emulated with a system-prompt instruction (schema included for
  `json_schema`); `text` ignored, other types raise `LLMError`.
- No new dependencies. Optional live test: `AGENTLENS_LIVE_TESTS=1
  AGENTLENS_LIVE_ANTHROPIC_MODEL=... ANTHROPIC_API_KEY=...`. 431 tests, 100% coverage.
- Next: #12 Client factory from config/env + pricing table.
