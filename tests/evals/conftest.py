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
