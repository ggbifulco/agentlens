# Contributing to AgentLens

Thanks for your interest! This project uses [uv](https://docs.astral.sh/uv/) for environments,
dependencies and builds (see [ADR 0001](docs/adr/0001-python-tooling.md)).

## Setup

```bash
git clone https://github.com/ggbifulco/agentlens
cd agentlens
uv sync                       # creates .venv with the package (editable) + dev tools
uv run pre-commit install     # optional: run the checks on every commit
```

## Dev commands

| Task | Command |
|---|---|
| Lint | `uv run ruff check .` |
| Auto-fix lint | `uv run ruff check --fix .` |
| Format | `uv run ruff format .` |
| Check formatting | `uv run ruff format --check .` |
| Type-check (strict) | `uv run mypy src` |
| Tests + coverage | `uv run pytest` |
| Live tests (real LLMs, need API keys) | `AGENTLENS_LIVE_TESTS=1 uv run pytest -m live` |
| All pre-commit hooks | `uv run pre-commit run --all-files` |
| Build sdist + wheel | `uv build` |

The full local gate, identical to what CI runs:

```bash
uv run ruff check . && uv run ruff format --check . && uv run mypy src && uv run pytest
```

## Conventions

- **Layout**: `src/agentlens/<package>/...`, tests mirror it under `tests/`.
- **Typing**: full type hints; `mypy --strict` must pass on `src/`.
- **Coverage**: `pytest` fails below 85% (configured in `pyproject.toml`).
- **No API keys in CI**: use `MockLLM` / mocked HTTP; real-provider tests use `@pytest.mark.live`
  and are skipped unless `AGENTLENS_LIVE_TESTS=1`.
- **Commits**: [Conventional Commits](https://www.conventionalcommits.org/)
  (`feat(tracing): ...`, `fix(store): ...`, `chore: ...`).
- **Branches**: `feat/<issue>-<slug>`, `chore/...`, `docs/...`, `test/...`.
- **Decisions**: significant design decisions get an ADR in `docs/adr/`.

The end-to-end workflow followed by the AI developer (and a good template for humans) is in
[AGENTS.md](AGENTS.md).
