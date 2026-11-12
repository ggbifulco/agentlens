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
