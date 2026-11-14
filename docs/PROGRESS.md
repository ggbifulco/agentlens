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

## 2026-11-13 — #12 Client factory from config/env + pricing table
- Added `agentlens.get_client(spec=None, *, pricing=None, **overrides)` (also in
  `agentlens.llm`, with `parse_spec`, `PROVIDERS`, `MODEL_ENV`): `"<provider>:<model>"` specs
  for `openai`, `anthropic`, `ollama`, `vllm`, `lmstudio`, `openai-compat` and `mock`
  (echo `MockLLM`, scriptable via `script=`); `None` reads `AGENTLENS_MODEL`. Env:
  `OPENAI_API_KEY`/`OPENAI_BASE_URL`, `ANTHROPIC_API_KEY`/`ANTHROPIC_BASE_URL`,
  `OLLAMA_BASE_URL` (default `http://localhost:11434/v1`), `VLLM_BASE_URL`/`VLLM_API_KEY`,
  `LMSTUDIO_BASE_URL`, `OPENAI_COMPAT_BASE_URL`/`OPENAI_COMPAT_API_KEY`; overrides win; local
  servers never receive `OPENAI_API_KEY`. Errors: new `LLMConfigError(LLMError, ValueError)`.
- Packaged `agentlens/llm/pricing.yaml` (approximate USD per 1M tokens, dated 2026-11-13,
  OpenAI + Anthropic; `free_providers` ollama/vllm/lmstudio/mock cost 0) and
  `Pricing`/`ModelPrice`: exact → alias → longest `-`/`:`/`@`-bounded prefix match, unknown
  model → `None`; `Pricing.load()` merges `$AGENTLENS_PRICING` (or a path) over the packaged
  table. `PricedClient` wrapper fills `usage.cost_usd` when the provider did not; `get_client`
  wraps every HTTP client, so `wrap_llm(get_client(...))` traces carry costs.
- New deps: `pyyaml` (runtime: pricing table now, suites/gates YAML in #20) and
  `types-PyYAML` (dev, mypy). 519 tests, 100% coverage; wheel verified to ship `pricing.yaml`.
- Next: #13 Cassette record/playback (strict / lenient / fallthrough).

## 2026-11-13 — #13 Cassette record/playback (strict / lenient / fallthrough)
- Added `agentlens.replay`: `Cassette(entries=(), *, mode="record", record_new=False,
  metadata=None)` with `wrap_llm(client=None, *, model=, provider=)` → `CassetteLLM` (an
  `LLMClient`, `complete`/`acomplete`) and `wrap_tool(fn, name=)` (wrapper or decorator, sync
  and async); `save(path)`, `Cassette.load(path, mode="strict", record_new=False)`,
  `Cassette.from_trace(trace)`, `entries`, `entries_of(kind)`, `unplayed`, `rewind()`.
  Also `CassetteEntry`, `CassetteHeader`, `tool_key`, `CassetteMode`, `EntryKind`.
- Calls keyed by `(index, key)`: index per kind (`llm`/`tool`), key = `LLMRequest.fingerprint()`
  (= span `request_hash`) or sha256 of canonical `{"name", "arguments"}` for tools. Modes:
  `record`, `strict` (key + tool name must match), `lenient` (index only), `fallthrough`
  (miss → real client/tool, stored only with `record_new=True`).
- Errors: `CassetteError` > `CassetteMissError(kind, index, mode, expected, actual_request,
  actual_key, actual_name)` (+ `expected_request`, base for #14's `ReplayDivergence`) and
  `CassetteFormatError(path:line)` (also a `ValueError`). Real-call errors are not recorded and
  free the index; calls inside a real tool bypass the cassette (they vanish on playback).
- JSONL format: header `{"format": "agentlens-cassette", "version": 1, "metadata": {...}}`
  (optional on load; carries the client's `model`/`provider` so playback without a client keeps
  span names), then one entry per line in recording order; atomic writes.
- Acceptance test: a MockLLM + tool run recorded, saved, loaded and replayed (strict, lenient,
  fallthrough, async, from_trace) against a failing client gives an identical trace.
- No new dependencies. 566 tests, 100% coverage.
- Next: #14 Replay divergence detection and request diff.

## 2026-11-13 — #14 Replay divergence detection and request diff
- Added `agentlens.replay.ReplayDivergence` (pydantic): `kind`, `index`, `reason`
  (`request_changed`/`tool_changed`/`extra_call`/`missing_call`), `mode`, expected/actual
  `name`/`key`/`request`, `diff`, `diff_truncated`, `span_id`; `create(...)`, `summary()`,
  `to_dict(include_requests=True)`, `str()` = summary + diff. `request_diff(expected, actual)`:
  `difflib` unified diff of pretty canonical JSON (sorted keys), bounded by `DIFF_MAX_LINES=200`
  and `DIFF_MAX_LINE_CHARS=500`.
- Strict mode: `CassetteMissError` message now ends with the diff; `exc.divergence`. Every
  playback mode records divergences in `cassette.divergences` / `cassette.first_divergence`
  (cleared by `rewind()`), lenient/fallthrough without failing.
- Report-ready trace marking: the diverging `llm_call`/`tool_call` span gets
  `replay_diverged=True` + `replay_divergence=<summary>` (`DIVERGED_ATTR`/`DIVERGENCE_ATTR`), the
  trace metadata `replay_divergence` = first divergence `to_dict(include_requests=False)`.
- `find_divergence(recorded_trace, replayed_trace, *, kind="llm")` compares two traces via
  `Cassette.from_trace` (same selection and keys). `CassetteMode`/`EntryKind` moved to
  `replay/_types.py` (re-exported) to break an import cycle.
- No new dependencies. 595 tests, 100% coverage; README example verified.
- Next: #15 Evaluator protocol, registry and @evaluator decorator (M3).

## 2026-11-13 — #15 Evaluator protocol, registry and @evaluator decorator
- Added `agentlens.evals`: `Evaluator` (runtime-checkable Protocol, read-only `name` +
  `evaluate(trace, case) -> EvalResult`) and `AsyncEvaluator` (`aevaluate`, for #18 judges).
  `to_eval_result(name, value, threshold=0.5)` normalises `EvalResult` / `bool` (1/0) / score in
  `[0, 1]` (`passed = score >= threshold`) / `(value, reason)` / mapping with `score`/`passed`.
- Safe runners: `run_evaluator`, `arun_evaluator` (awaits `aevaluate` or runs `evaluate` in a
  thread), `evaluate_case(trace, case, evaluators=None)` (default `case.evaluators`, objects and
  specs mixed, all resolved first), `aevaluate_case` (concurrent, ordered). Any evaluator
  exception becomes `error_result`: score 0, failed, `reason="evaluator error: Type: msg"`,
  `details["error"]`.
- Registry: `EvaluatorRegistry` + `default_registry`, `register(name, factory, primary=,
  replace=)`, `unregister`, `get_evaluator(spec, **overrides)`, `available_evaluators()`. Scalar
  shorthand `{max_steps: 8}` (`{"value": 8}`) goes to the factory's primary param (explicit, else
  first keyword-passable one). Params checked with `Signature.bind`; config errors raised eagerly
  (`UnknownEvaluatorError` with suggestions, `EvaluatorConfigError`, `DuplicateEvaluatorError`;
  same-origin re-registration replaces silently).
- `@agentlens.evaluator` (bare or `name=`, `threshold=`, `register=`, `replace=`, `registry=`)
  → `FunctionEvaluator` (still callable; `bind(**params)` is the registry factory; extra params
  after `(trace, case)` come from the spec, `threshold` always accepted; `async def` supported).
  Top-level `agentlens.evaluator`, `agentlens.Evaluator`. README "Custom evaluators" section.
- No new dependencies. 669 tests, 100% coverage; README example verified.
- Next: #16 Deterministic output evaluators (exact, contains, regex, json_schema, no_error).

## 2026-11-13 — #16 Deterministic output evaluators (exact, contains, regex, json_schema, no_error)
- Added `agentlens.evals.builtin` (registered in `default_registry` on import of
  `agentlens.evals`; `register_builtins(registry)`, `BUILTIN_EVALUATORS`): `ExactMatch`
  (`exact_match`), `Contains` (`contains`), `Regex` (`regex`), `JsonSchema` (`json_schema`),
  `NoError` (`no_error`). Shorthand primaries: `value`, `text`, `pattern`, `schema`,
  `include_spans`. Each returns score + reason + details; bad params (invalid regex/schema,
  unknown mode, non-bool flags) are `EvaluatorConfigError` at build time.
- Output selection: `output_value(trace, path=)` (dotted path or list; a string output is
  parsed as JSON when a path is given; missing path → `OutputPathError` → failed result),
  `to_text`/`output_text` (str as is, `None` stays `None`, else canonical JSON). Strings are
  NFC-normalised; `case_sensitive=False` = casefold; `normalise_whitespace` option.
- `exact_match`: `value` param > `case.expected` > config error (`UNSET` sentinel so `null`
  can be expected); string expectation → text compare, otherwise structural JSON compare
  (`true != 1`). `contains`: str/list, `mode` all/any, score = fraction found. `regex`:
  `mode` search/match/fullmatch + `ignore_case`/`multiline`/`dotall`. `json_schema`: schema or
  `case.expected.schema`, string outputs parsed as JSON (code fence tolerated), first
  `max_errors` errors in reason/details. `no_error`: `include_spans=True` by default.
- New runtime dependency `jsonschema` (+ `types-jsonschema` dev). 861 tests, 100% coverage;
  README example verified.
- Next: #17 Trajectory evaluators (tool_sequence, max_steps, cost/latency budgets).

## 2026-11-13 — #17 Trajectory evaluators (tool_sequence, max_steps, cost/latency budgets)
- Added to `agentlens.evals.builtin.trace_checks` (registered on import, re-exported from
  `agentlens.evals`): `ToolSequence` (`tool_sequence`, primary `tools`), `MaxSteps`
  (`max_steps`, `value`), `CostBudget` (`cost_budget`, `max_usd`), `LatencyBudget`
  (`latency_budget`, `max_ms`). Bad limits/modes are `EvaluatorConfigError` at build time.
- `tool_sequence(tools=None, mode="ordered"|"strict"|"subset", allow_extra=True, forbidden=None)`:
  expected tools from `tools`, else `case.expected["tools"]`; repeats count (multiset). Scores:
  strict `LCS / max(len)` (1 only on exact match), ordered `LCS / len(expected)` (subsequence),
  subset `matched / len(expected)` (any order). `allow_extra=False` fails on calls to tools
  outside the list; `forbidden` calls score 0. Reason shows expected vs actual; details list
  `missing`, `unexpected`, `forbidden_called`.
- Budgets: pass iff `actual <= limit`, score 1 within else `limit / actual`. `max_steps` uses
  `trace.n_steps`; `latency_budget` fails a running trace (`latency_ms=None`); `cost_budget`
  fails on unknown/partial cost by default (`on_unknown: pass` to accept; a known part already
  over budget fails regardless).
- Shared helpers `check_choice`, `check_number`, `params_repr` moved to `builtin/_common.py`.
  Tests use a synthetic-trace builder (`tests/evals/builtin/conftest.py::build_trace`).
- No new dependencies. 954 tests, 100% coverage; README example verified.
- Next: #18 LLM-as-judge evaluators (rubric 1-5, pass/fail) with caching.

## 2026-11-13 — #18 LLM-as-judge evaluators (rubric 1-5, pass/fail) with caching
- Added `agentlens.evals.judge` (reusable for #19): `LLMJudge[Verdict]` abstract evaluator
  (subclasses define `build_prompt`, `parse_reply`, `make_result`; the base does cache lookup →
  call at `temperature=0` with `response_format={"type": "json_object"}` → parse → retry
  (`retries=1`, bad reply fed back) → cache write, in both `evaluate` and `aevaluate`),
  `JudgePrompt`, `JudgeRun`, `value_text`; parsing helpers `extract_json_object` (code fences,
  prose, first `{`), `parse_score` (numeric strings, `"4/5"`, range-checked, no clamping),
  `parse_verdict`, `parse_reasoning`, `JudgeParseError`; prompts `load_template`/`render_template`/
  `check_template` over packaged `string.Template` files in `evals/prompts/` (`judge_system`,
  `rubric`, `pass_fail`, `reference`).
- Cache: `JudgeCache` protocol, `MemoryJudgeCache`, `DiskJudgeCache` (JSON file per key, atomic
  writes), `judge_cache_key` = sha256 of `[judge model, prompt, output]`, `resolve_cache`. Default
  on disk at `.agentlens/judge_cache/`; `cache=`/`$AGENTLENS_JUDGE_CACHE` = off | memory | path.
  The raw reply is cached only after it parsed.
- Evaluators (registered, re-exported from `agentlens.evals`): `LLMRubric` (`llm_rubric`, primary
  `criteria`, `min_score=4`, score `(s-1)/4`) and `LLMPassFail` (`llm_pass_fail`), both with
  `include_expected`, `template`, `model`/`client`, `cache`, `retries`, `max_tokens`. Judge model
  from `model`, else `$AGENTLENS_JUDGE_MODEL`, else `$AGENTLENS_MODEL`, resolved lazily.
  Unparseable reply → failed result with `details.raw_reply` and `details.error`.
- Docs: design §5 implementation notes, new `docs/judges.md` (usage, caching, position/verbosity/
  self-preference bias, calibration against human labels), README "LLM judges" example (verified).
- No new dependencies. 1131 tests, 100% coverage.
- Next: #19 Pairwise A/B judge.

## 2026-11-13 — #19 Pairwise A/B judge
- Added `agentlens.evals.judge.pairwise` (re-exported from `agentlens.evals.judge` and
  `agentlens.evals`): `PairwiseJudge(criteria, include_expected=True, template=None, model|client,
  cache, retries, max_tokens)` with `compare(trace_a, trace_b, case=None)` / async `acompare`
  (orders judged concurrently) → `PairwiseResult` (`winner` a/b/tie, `preference` for A in [0, 1],
  `consistent`, `confidence`, `orders` = two `OrderVerdict`, `reason`, `reasons`, `details()`,
  `to_eval_result(candidate="b")` → passed when the candidate wins or ties). New packaged prompt
  `prompts/pairwise.txt` (`$output_1`/`$output_2`, reply `{"reasoning", "winner": "1"|"2"|"tie"}`),
  tolerant `parse_position`.
- Position-bias mitigation: every comparison is judged as (A, B) and (B, A); the swapped answer is
  mapped back to A/B and the preferences (1 / 0.5 / 0) averaged. Opposite answers → tie with
  confidence 0; winner vs tie → that winner with confidence 0.5; all recorded in `details["orders"]`.
  An unparseable order → `error`, tie, failed `EvalResult` (never decided on one order).
- Refactor: the client/cache/call flow of `LLMJudge` moved to a new `BaseJudge` base (LLMJudge API
  unchanged). Cache key per order = `judge_cache_key(model, prompt, json([output 1, output 2]))`.
- Not in the evaluator registry (needs two traces); the comparator (#22) will call it per case on
  (baseline, candidate). Docs: `docs/judges.md` (pairwise usage, position-bias section), design §5
  notes, README example (verified).
- No new dependencies. 1174 tests, 100% coverage.
- Next: #20 YAML suite and dataset loader (M4).

## 2026-11-14 — #20 YAML suite and dataset loader
- Added `agentlens.suite`: `Suite` model (`name`, `agent` import path or callable, `model`,
  `repetitions`, `concurrency`, `timeout_s`, `imports`, `metadata`, `defaults` = `SuiteDefaults`
  (`evaluators`, `tags`), `cases`), `load_suite(path, registry=None, check=True)`,
  `load_cases(path)` for `.jsonl`/`.yaml`/`.yml`/`.json` datasets (also `cases: path` in a suite),
  `resolve_agent`/`Suite.load_agent()` (`pkg.mod:callable`, `mod:Agent.run`; cwd and suite folder
  appended to `sys.path`), `merge_evaluators`, `SuiteError`/`SuiteIssue`/`AgentImportError`.
- Validation errors with line context: YAML composed once into nodes whose marks are recorded by
  path, so every problem (schema, duplicate keys/ids, unknown evaluator with "did you mean",
  bad params, failing `imports`) is reported together as `file:line:col: cases[0].evaluators[1]:
  message`, also inside external datasets.
- Defaults merged into cases by `Suite.resolved_cases()`: defaults first; a case evaluator replaces
  defaults of the same name; tags de-duplicated.
- For #21: `Suite.content_hash()` (sha256 of canonical JSON of what runs, resolved cases),
  `is_async_agent` / `Suite.agent_is_async()`.
- Docs: README "Suites" section (verified), design §6 implementation notes.
- No new dependencies (PyYAML already required). 1295 tests, 100% coverage.
- Next: #21 Suite runner (repetitions, bounded asyncio concurrency, experiment persistence,
  baseline mark).

## 2026-11-14 — #21 Suite runner (repetitions, concurrency, experiment persistence)
- Added `agentlens.suite.runner`: `run_suite(suite | path, *, store, name, model, repetitions,
  concurrency, timeout_s, config, registry, redactor, progress)` (blocking, `asyncio.run`) and
  async `arun_suite`. One task per (case, repetition) behind an `asyncio.Semaphore(concurrency)`
  (slot = agent call + evaluation); async agents awaited, sync agents via `asyncio.to_thread`
  (context copied, so `wrap_llm`/`wrap_tool` spans land in the run's trace). Each run is
  `agentlens.run(suite.name, metadata={case_id, repetition, suite, experiment})`.
- Agent convention: `agent(case.input)`, plus `model=` when the agent has a `model` parameter
  (`model=` > `suite.model` > `$AGENTLENS_MODEL`), ready for `get_client(model)`.
- Failure isolation: agent exceptions and `RunTimeoutError` (per-run `timeout_s`, `wait_for`)
  recorded as `trace.error`, run still evaluated and saved; store errors abort the suite. The
  redacted trace is what gets evaluated and stored; a redactor failure stores a content-free
  error trace.
- Results: `ExperimentRun` (`experiment`, `cases`, `runs`, `total`, `passed`, `errors`,
  `pass_rate`, `case(id)`), `CaseResult` (`pass_rate`, `errors`, `mean_scores()`), `RunResult`
  (`trace`, `eval_results`, `passed`, `error`, `timed_out`); `progress(result, done, total)`.
- `agentlens.suite.experiment`: `experiment_config` (git sha + dirty flag, model, suite hash,
  agent, repetitions/concurrency/timeout, agentlens version, suite metadata, user overrides),
  `git_info`, `resolve_model`; baselines `set_baseline` (one per suite, others unmarked first),
  `unset_baseline`, `get_baseline`, `find_experiment` (id or most recent name),
  `ExperimentNotFoundError`.
- Docs: README "Running a suite" (verified), design §6 implementation notes.
- No new dependencies. 1342 tests, 100% coverage.
- Next: #22 Comparator with paired bootstrap confidence intervals.

## 2026-11-14 — #22 Comparator with paired bootstrap confidence intervals
- Added `agentlens.compare`: `compare_experiments(store, baseline, candidate, *, resamples=10_000,
  seed=0, confidence=0.95) -> Comparison` (id, name or `Experiment`), `compare_runs` on in-memory
  data (`ExperimentRun` or `RunResult`s), `experiment_runs(store, experiment)` (runs rebuilt from
  `case_id`/`repetition` trace metadata + stored eval results).
- Metrics per case (averaged over repetitions) and in aggregate: `pass_rate` (same rule as
  `RunResult.passed`), `score.<evaluator>`, `cost_usd` (None-aware), `latency_ms`, `steps`.
  Paired by case id over shared cases; `added_cases`/`removed_cases` reported.
- `paired_bootstrap(diffs, *, resamples, confidence, seed)`: percentile CI, stdlib
  `random.Random(seed)` (no NumPy; ~60 ms for 10k × 100 cases). `classify` by metric direction:
  CI entirely on the bad side → regression, good side → improvement, else unchanged (`no_data`
  without pairs).
- Pydantic models `MetricDelta`, `CaseMetric`, `CaseComparison` (pass-rate status,
  `regressed_evaluators`), `Comparison` (`metric()`, `case()`, `regressions`, `improvements`,
  `case_regressions`, `case_improvements`, `has_regressions`), JSON round-trip for gates/reports.
- Hypothesis properties: CI contains the observed mean, zero diffs → [0, 0], shift invariance,
  same seed → same CI, CI within the data range.
- Docs: README "Comparing experiments" (verified), design §7 implementation notes. Pairwise judge
  integration left to the CLI/gates layer (noted in design §7).
- No new dependencies. 1400 tests, 100% coverage.
- Next: #23 Gates config and exit codes (YAML thresholds on top of `Comparison`).
