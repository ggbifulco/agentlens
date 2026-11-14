"""Fixtures for suite tests: isolated ``sys.path``/``sys.modules`` and file helpers."""

from __future__ import annotations

import sys
import textwrap
from collections.abc import Callable, Iterator
from pathlib import Path

import pytest

from agentlens.evals import EvaluatorRegistry, NoError, register_builtins

WriteFile = Callable[[str, str], Path]


@pytest.fixture(autouse=True)
def _isolated_imports(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Undo ``sys.path`` additions and forget modules imported by a test."""
    monkeypatch.setattr(sys, "path", list(sys.path))
    before = set(sys.modules)
    yield
    for name in set(sys.modules) - before:
        del sys.modules[name]


@pytest.fixture
def write(tmp_path: Path) -> WriteFile:
    """``write("suite.yaml", text)`` -> path of a dedented file under ``tmp_path``."""

    def _write(name: str, text: str) -> Path:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
        return path

    return _write


@pytest.fixture
def registry() -> EvaluatorRegistry:
    """A fresh registry with the built-ins only."""
    reg = EvaluatorRegistry()
    register_builtins(reg)
    return reg


@pytest.fixture
def tiny_registry() -> EvaluatorRegistry:
    reg = EvaluatorRegistry()
    reg.register("no_error", NoError)
    return reg
