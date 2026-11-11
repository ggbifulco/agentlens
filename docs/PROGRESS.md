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
