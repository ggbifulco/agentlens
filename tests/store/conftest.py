"""Fixtures for store tests: every test gets its own database file under ``tmp_path``."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue

from agentlens.schema import Trace
from agentlens.store import SQLiteStore

T0 = datetime(2026, 11, 13, 9, 0, tzinfo=UTC)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteStore]:
    with SQLiteStore(tmp_path / "agentlens.db") as s:
        yield s


def make_trace(
    name: str = "agent",
    *,
    minute: int = 0,
    metadata: dict[str, JsonValue] | None = None,
) -> Trace:
    """A small finished trace starting ``minute`` minutes after :data:`T0`."""
    start = T0 + timedelta(minutes=minute)
    return Trace(
        name=name,
        input="hi",
        output="hello",
        start_time=start,
        end_time=start + timedelta(seconds=1),
        metadata=metadata or {},
    )
