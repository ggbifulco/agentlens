"""Database lifecycle: migrations, persistence across connections, threads, round-trips."""

from __future__ import annotations

import sqlite3
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from hypothesis import given, settings

from agentlens.schema import EvalResult, Experiment, Trace
from agentlens.store import SCHEMA_VERSION, SchemaVersionError, SQLiteStore
from agentlens.store import migrations as m
from tests.schema import strategies as s
from tests.store.conftest import make_trace


def test_new_database_is_at_latest_version(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "db.sqlite"
    with SQLiteStore(path):
        pass
    assert path.exists()
    conn = sqlite3.connect(path)
    assert m.schema_version(conn) == SCHEMA_VERSION == len(m.MIGRATIONS)
    conn.close()


def test_data_survives_reopening(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    trace = make_trace()
    with SQLiteStore(path) as store:
        store.save_trace(trace)
    with SQLiteStore(path) as store:
        assert store.get_trace(trace.id) == trace


def test_migrations_are_applied_once(tmp_path: Path) -> None:
    conn = sqlite3.connect(tmp_path / "db.sqlite", isolation_level=None)
    assert m.apply_migrations(conn) == SCHEMA_VERSION
    assert m.apply_migrations(conn) == 0
    conn.close()


def test_newer_database_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "db.sqlite"
    conn = sqlite3.connect(path)
    conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 1}")
    conn.close()
    with pytest.raises(SchemaVersionError, match="newer"):
        SQLiteStore(path)


def test_failed_migration_rolls_back(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    broken = (*m.MIGRATIONS, "CREATE TABLE extra (x INTEGER); THIS IS NOT SQL")
    monkeypatch.setattr(m, "MIGRATIONS", broken)
    monkeypatch.setattr(m, "SCHEMA_VERSION", len(broken))
    conn = sqlite3.connect(tmp_path / "db.sqlite", isolation_level=None)

    with pytest.raises(sqlite3.OperationalError):
        m.apply_migrations(conn)

    # Migration 1 is committed; the broken one left neither its table nor its version.
    assert m.schema_version(conn) == len(broken) - 1
    tables = {r[0] for r in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
    assert "traces" in tables
    assert "extra" not in tables
    conn.close()


def test_in_memory_store() -> None:
    with SQLiteStore(":memory:") as store:
        trace = make_trace()
        store.save_trace(trace)
        assert store.get_trace(trace.id) == trace
        assert repr(store) == "SQLiteStore(':memory:')"


def test_concurrent_writes_from_threads(store: SQLiteStore) -> None:
    traces = [make_trace(minute=i) for i in range(50)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(store.save_trace, traces))
    assert len(store.list_traces()) == 50


@settings(max_examples=50, deadline=None)
@given(trace=s.traces(), experiment=s.experiments, results=s.eval_results.map(lambda r: [r]))
def test_hypothesis_round_trip(
    trace: Trace, experiment: Experiment, results: list[EvalResult]
) -> None:
    with SQLiteStore(":memory:") as store:
        store.save_experiment(experiment)
        store.save_trace(trace, experiment_id=experiment.id)
        store.save_eval_results(trace.id, results)

        assert store.get_experiment(experiment.id) == experiment
        assert store.get_trace(trace.id) == trace
        assert store.list_traces(experiment_id=experiment.id) == [trace]
        assert store.list_eval_results(experiment.id) == {trace.id: results}
