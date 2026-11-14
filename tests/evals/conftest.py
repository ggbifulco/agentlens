"""Shared fixtures for evaluator tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest

from agentlens.evals import EvaluatorRegistry, default_registry
from agentlens.schema import Case, Trace


@pytest.fixture
def trace() -> Trace:
    return Trace(name="agent", input="refund order 1234", output="Refund issued for order 1234")


@pytest.fixture
def case() -> Case:
    return Case(id="refund", input="refund order 1234", expected="order 1234")


@pytest.fixture
def registry() -> EvaluatorRegistry:
    return EvaluatorRegistry()


@pytest.fixture
def clean_default_registry() -> Iterator[EvaluatorRegistry]:
    """Restore the process-wide registry after a test that registers into it."""
    saved = dict(default_registry._entries)
    try:
        yield default_registry
    finally:
        default_registry._entries.clear()
        default_registry._entries.update(saved)


@pytest.fixture(autouse=True)
def _isolated_judge_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """No judge model from the developer's shell, and no default on-disk judge cache."""
    monkeypatch.delenv("AGENTLENS_JUDGE_MODEL", raising=False)
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)
    monkeypatch.setenv("AGENTLENS_JUDGE_CACHE", "off")
