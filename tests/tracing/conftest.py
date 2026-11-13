"""Fixtures for tracing tests."""

from __future__ import annotations

import pytest

from agentlens.schema import Trace


@pytest.fixture
def sink() -> list[Trace]:
    """A list used as trace sink: ``run(..., sink=sink.append)``."""
    return []
