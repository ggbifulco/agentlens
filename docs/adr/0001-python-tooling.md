# ADR 0001: Python toolchain

- Status: Accepted
- Date: 2026-11-11

## Context
We need a modern, fast, reproducible Python setup that reviewers recognise.

## Decision
Python >= 3.11, `uv` for env/deps/build, `ruff` (lint+format), `mypy --strict` on `src/`, `pytest` + `pytest-cov` + `hypothesis`, `pre-commit`. src-layout package, `pyproject.toml` only.

## Consequences
Fast CI, single config file. uv is newer than pip/poetry, but widely adopted in 2025-26.
