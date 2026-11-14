"""The default :class:`~agentlens.store.Store` backend: one local SQLite file (ADR 0003)."""

from __future__ import annotations

import os
import sqlite3
import threading
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from types import TracebackType
from typing import Self

from agentlens.schema import EvalResult, Experiment, Trace
from agentlens.store.base import MetadataScalar, StoreError
from agentlens.store.migrations import apply_migrations

__all__ = ["DEFAULT_PATH", "SQLiteStore"]

DEFAULT_PATH = Path(".agentlens") / "agentlens.db"
"""Where the CLI keeps its database unless told otherwise (project-local)."""

_MEMORY = ":memory:"


def _timestamp(value: datetime) -> str:
    """Fixed-width UTC ISO-8601, so that string order equals time order.

    (``datetime.isoformat`` drops the microseconds when they are zero, which
    would break lexicographic sorting.)
    """
    return value.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%S.%fZ")


def _metadata_clause(key: str, value: MetadataScalar) -> tuple[str, list[object]]:
    """SQL condition matching traces whose ``metadata[key] == value``.

    ``json_each`` yields one row per metadata entry with its JSON ``type`` and
    SQL ``atom``. Comparing the type too keeps ``True`` from matching ``1`` and
    ``None`` from matching a missing key.
    """
    sql = "EXISTS (SELECT 1 FROM json_each(traces.data, '$.metadata') AS m WHERE m.key = ? AND "
    if value is None:
        return sql + "m.type = 'null')", [key]
    if isinstance(value, bool):
        return sql + "m.type = ?)", [key, "true" if value else "false"]
    if isinstance(value, int | float):
        return sql + "m.type IN ('integer', 'real') AND m.atom = ?)", [key, value]
    if isinstance(value, str):
        return sql + "m.type = 'text' AND m.atom = ?)", [key, value]
    raise TypeError(f"metadata filter values must be JSON scalars, got {type(value).__name__}")


_ERROR_CLAUSE = (
    "(json_extract(traces.data, '$.error') IS NOT NULL OR EXISTS ("
    "SELECT 1 FROM json_each(traces.data, '$.spans') AS s "
    "WHERE json_extract(s.value, '$.error') IS NOT NULL))"
)
"""SQL condition: the run or one of its spans recorded an error."""


class SQLiteStore:
    """A :class:`~agentlens.store.Store` backed by a SQLite database file.

    Objects are stored as JSON documents next to a few indexed columns used for
    filtering, so reading a trace back yields exactly the trace that was saved.
    The schema is created/upgraded on open (:mod:`agentlens.store.migrations`).

    One connection is shared behind a lock, so a store can be used from several
    threads (e.g. as the sink of runs executed in a thread pool). It is not
    meant for many concurrent writer *processes* (ADR 0003).

    Example::

        with SQLiteStore("runs.db") as store, agentlens.run("demo", sink=store.save_trace):
            ...
    """

    def __init__(self, path: str | os.PathLike[str] = DEFAULT_PATH) -> None:
        self.path = str(path)
        if self.path != _MEMORY:
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        # isolation_level=None: autocommit, transactions are explicit (see _write).
        self._conn = sqlite3.connect(self.path, isolation_level=None, check_same_thread=False)
        self._lock = threading.RLock()
        try:
            self._conn.execute("PRAGMA foreign_keys = ON")
            apply_migrations(self._conn)
        except BaseException:
            self._conn.close()
            raise

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the database connection. The store cannot be used afterwards."""
        with self._lock:
            self._conn.close()

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self.close()

    def __repr__(self) -> str:
        return f"SQLiteStore({self.path!r})"

    @contextmanager
    def _write(self) -> Iterator[sqlite3.Connection]:
        """One write transaction: commit on success, roll back on any error."""
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except sqlite3.IntegrityError as exc:
                self._conn.execute("ROLLBACK")
                raise StoreError(f"integrity error: {exc}") from exc
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            self._conn.execute("COMMIT")

    def _query(self, sql: str, params: Sequence[object] = ()) -> list[tuple[object, ...]]:
        with self._lock:
            return self._conn.execute(sql, params).fetchall()

    # -- traces --------------------------------------------------------------

    def save_trace(self, trace: Trace, *, experiment_id: str | None = None) -> None:
        """Persist ``trace`` (upsert), optionally linking it to an experiment.

        Raises :class:`StoreError` if ``experiment_id`` is not a stored experiment.
        """
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO traces (id, name, experiment_id, start_time, data)
                VALUES (?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name,
                    experiment_id = COALESCE(excluded.experiment_id, traces.experiment_id),
                    start_time = excluded.start_time,
                    data = excluded.data
                """,
                (
                    trace.id,
                    trace.name,
                    experiment_id,
                    _timestamp(trace.start_time),
                    trace.model_dump_json(),
                ),
            )

    def get_trace(self, trace_id: str) -> Trace | None:
        """The trace with id ``trace_id``, or ``None``."""
        rows = self._query("SELECT data FROM traces WHERE id = ?", (trace_id,))
        return Trace.model_validate_json(str(rows[0][0])) if rows else None

    def list_traces(
        self,
        *,
        name: str | None = None,
        experiment_id: str | None = None,
        metadata: Mapping[str, MetadataScalar] | None = None,
        has_error: bool | None = None,
        limit: int | None = None,
        offset: int = 0,
    ) -> list[Trace]:
        """Traces matching all given filters, most recent first.

        ``has_error=True`` keeps traces where the run or any of its spans
        recorded an error (``False``: neither did). ``offset`` skips that many
        matches first (pagination together with ``limit``).
        """
        where: list[str] = []
        params: list[object] = []
        if has_error is not None:
            where.append(_ERROR_CLAUSE if has_error else f"NOT {_ERROR_CLAUSE}")
        if name is not None:
            where.append("name = ?")
            params.append(name)
        if experiment_id is not None:
            where.append("experiment_id = ?")
            params.append(experiment_id)
        for key, value in (metadata or {}).items():
            clause, clause_params = _metadata_clause(key, value)
            where.append(clause)
            params.extend(clause_params)
        sql = "SELECT data FROM traces"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY start_time DESC, id DESC"
        if limit is not None:
            if limit < 0:
                raise ValueError("limit must be >= 0")
            sql += " LIMIT ?"
            params.append(limit)
        if offset < 0:
            raise ValueError("offset must be >= 0")
        if offset:
            if limit is None:
                sql += " LIMIT -1"  # SQLite needs a LIMIT before OFFSET; -1 = no limit
            sql += " OFFSET ?"
            params.append(offset)
        return [Trace.model_validate_json(str(data)) for (data,) in self._query(sql, params)]

    # -- experiments ---------------------------------------------------------

    def save_experiment(self, experiment: Experiment) -> None:
        """Persist ``experiment`` (upsert by id)."""
        with self._write() as conn:
            conn.execute(
                """
                INSERT INTO experiments (id, name, suite, created_at, is_baseline, data)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (id) DO UPDATE SET
                    name = excluded.name,
                    suite = excluded.suite,
                    created_at = excluded.created_at,
                    is_baseline = excluded.is_baseline,
                    data = excluded.data
                """,
                (
                    experiment.id,
                    experiment.name,
                    experiment.suite,
                    _timestamp(experiment.created_at),
                    int(experiment.is_baseline),
                    experiment.model_dump_json(),
                ),
            )

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """The experiment with id ``experiment_id``, or ``None``."""
        rows = self._query("SELECT data FROM experiments WHERE id = ?", (experiment_id,))
        return Experiment.model_validate_json(str(rows[0][0])) if rows else None

    def list_experiments(
        self,
        *,
        name: str | None = None,
        suite: str | None = None,
        is_baseline: bool | None = None,
    ) -> list[Experiment]:
        """Experiments matching all given filters, most recent first."""
        where: list[str] = []
        params: list[object] = []
        for column, value in (("name", name), ("suite", suite), ("is_baseline", is_baseline)):
            if value is not None:
                where.append(f"{column} = ?")
                params.append(int(value) if isinstance(value, bool) else value)
        sql = "SELECT data FROM experiments"
        if where:
            sql += " WHERE " + " AND ".join(where)
        sql += " ORDER BY created_at DESC, id DESC"
        return [Experiment.model_validate_json(str(data)) for (data,) in self._query(sql, params)]

    # -- eval results --------------------------------------------------------

    def save_eval_results(self, trace_id: str, results: Sequence[EvalResult]) -> None:
        """Replace the evaluation results of a stored trace.

        Raises :class:`StoreError` if the trace is not stored.
        """
        with self._write() as conn:
            if conn.execute("SELECT 1 FROM traces WHERE id = ?", (trace_id,)).fetchone() is None:
                raise StoreError(f"unknown trace {trace_id!r}")
            conn.execute("DELETE FROM eval_results WHERE trace_id = ?", (trace_id,))
            conn.executemany(
                "INSERT INTO eval_results (trace_id, position, evaluator, data)"
                " VALUES (?, ?, ?, ?)",
                [(trace_id, i, r.evaluator, r.model_dump_json()) for i, r in enumerate(results)],
            )

    def get_eval_results(self, trace_id: str) -> list[EvalResult]:
        """The evaluation results of one trace, in saved order (empty if none)."""
        rows = self._query(
            "SELECT data FROM eval_results WHERE trace_id = ? ORDER BY position", (trace_id,)
        )
        return [EvalResult.model_validate_json(str(data)) for (data,) in rows]

    def list_eval_results(self, experiment_id: str) -> dict[str, list[EvalResult]]:
        """Evaluation results of every evaluated trace of an experiment, by trace id."""
        rows = self._query(
            """
            SELECT r.trace_id, r.data FROM eval_results AS r
            JOIN traces AS t ON t.id = r.trace_id
            WHERE t.experiment_id = ?
            ORDER BY t.start_time, t.id, r.position
            """,
            (experiment_id,),
        )
        grouped: dict[str, list[EvalResult]] = {}
        for trace_id, data in rows:
            grouped.setdefault(str(trace_id), []).append(EvalResult.model_validate_json(str(data)))
        return grouped
