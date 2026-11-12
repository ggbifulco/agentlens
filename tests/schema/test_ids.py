"""Tests for the UUIDv7 id helper."""

from __future__ import annotations

import threading
import uuid
from datetime import UTC, datetime, timedelta

import pytest

from agentlens.schema import ids
from agentlens.schema.ids import id_datetime, new_id


def test_new_id_is_a_uuid_version_7() -> None:
    parsed = uuid.UUID(new_id())
    assert parsed.version == 7
    assert parsed.variant == uuid.RFC_4122


def test_ids_are_strictly_increasing_within_a_process() -> None:
    generated = [new_id() for _ in range(10_000)]
    assert generated == sorted(generated)
    assert len(set(generated)) == len(generated)


def test_ids_are_unique_across_threads() -> None:
    results: list[str] = []
    lock = threading.Lock()

    def worker() -> None:
        local = [new_id() for _ in range(2_000)]
        with lock:
            results.extend(local)

    threads = [threading.Thread(target=worker) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(set(results)) == len(results) == 16_000


def test_id_datetime_recovers_creation_time() -> None:
    before = datetime.now(UTC) - timedelta(milliseconds=1)
    created = id_datetime(new_id())
    after = datetime.now(UTC) + timedelta(milliseconds=1)
    assert before <= created <= after


def test_id_datetime_rejects_other_uuid_versions() -> None:
    with pytest.raises(ValueError, match="not a UUIDv7"):
        id_datetime(str(uuid.uuid4()))


@pytest.fixture
def isolated_clock(monkeypatch: pytest.MonkeyPatch) -> pytest.MonkeyPatch:
    """Reset the generator state; monkeypatch restores it after the test."""
    monkeypatch.setattr(ids, "_last_ms", -1)
    monkeypatch.setattr(ids, "_counter", 0)
    return monkeypatch


def test_counter_overflow_borrows_next_millisecond(isolated_clock: pytest.MonkeyPatch) -> None:
    frozen_ms = 1_700_000_000_000
    isolated_clock.setattr(ids, "_clock_ns", lambda: frozen_ms * 1_000_000)
    generated = [new_id() for _ in range(ids._COUNTER_MAX + 10)]
    assert generated == sorted(generated)
    assert len(set(generated)) == len(generated)
    assert id_datetime(generated[-1]) > id_datetime(generated[0])


def test_clock_going_backwards_keeps_ids_monotonic(isolated_clock: pytest.MonkeyPatch) -> None:
    clock = iter([1_700_000_000_000, 1_699_999_999_000])
    isolated_clock.setattr(ids, "_clock_ns", lambda: next(clock) * 1_000_000)
    first, second = new_id(), new_id()
    assert first < second
