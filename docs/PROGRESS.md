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
