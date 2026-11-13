"""SQLite schema migrations, versioned in code.

The schema version lives in SQLite's built-in ``PRAGMA user_version`` (an
integer in the database header). Migration ``n`` (1-based) upgrades a database
from version ``n - 1`` to ``n``. Migrations are append-only: once released, a
migration is never edited; a schema change is a new entry at the end.
"""

from __future__ import annotations

import sqlite3

from agentlens.store.base import SchemaVersionError

__all__ = ["MIGRATIONS", "SCHEMA_VERSION", "apply_migrations", "schema_version"]

MIGRATIONS: tuple[str, ...] = (
    # 1 — initial schema. Full objects are stored as JSON documents (``data``);
    # the columns next to them are copies of the fields we filter and sort on.
    """
    CREATE TABLE experiments (
        id          TEXT PRIMARY KEY,
        name        TEXT NOT NULL,
        suite       TEXT NOT NULL,
        created_at  TEXT NOT NULL,
        is_baseline INTEGER NOT NULL,
        data        TEXT NOT NULL
    );
    CREATE INDEX experiments_name ON experiments (name, created_at);
    CREATE INDEX experiments_suite ON experiments (suite, created_at);

    CREATE TABLE traces (
        id            TEXT PRIMARY KEY,
        name          TEXT NOT NULL,
        experiment_id TEXT REFERENCES experiments (id) ON DELETE CASCADE,
        start_time    TEXT NOT NULL,
        data          TEXT NOT NULL
    );
    CREATE INDEX traces_name ON traces (name, start_time);
    CREATE INDEX traces_experiment ON traces (experiment_id, start_time);
    CREATE INDEX traces_start ON traces (start_time);

    CREATE TABLE eval_results (
        trace_id  TEXT NOT NULL REFERENCES traces (id) ON DELETE CASCADE,
        position  INTEGER NOT NULL,
        evaluator TEXT NOT NULL,
        data      TEXT NOT NULL,
        PRIMARY KEY (trace_id, position)
    );
    """,
)

SCHEMA_VERSION = len(MIGRATIONS)
"""The schema version this AgentLens writes."""


def schema_version(conn: sqlite3.Connection) -> int:
    """Current schema version of the database behind ``conn`` (0 = empty)."""
    (version,) = conn.execute("PRAGMA user_version").fetchone()
    return int(version)


def apply_migrations(conn: sqlite3.Connection) -> int:
    """Upgrade the database to :data:`SCHEMA_VERSION`; return the number applied.

    Each migration runs in its own transaction together with the version bump,
    so a failure leaves the database at the last fully applied version.
    ``conn`` must be in autocommit mode (``isolation_level=None``).
    """
    current = schema_version(conn)
    if current > SCHEMA_VERSION:
        raise SchemaVersionError(
            f"database schema version {current} is newer than this AgentLens "
            f"supports ({SCHEMA_VERSION}); upgrade agentlens"
        )
    for version in range(current + 1, SCHEMA_VERSION + 1):
        # executescript() would commit implicitly; run statements one by one
        # inside an explicit transaction instead (SQLite DDL is transactional).
        conn.execute("BEGIN IMMEDIATE")
        try:
            for statement in _split(MIGRATIONS[version - 1]):
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {version}")
        except BaseException:
            conn.execute("ROLLBACK")
            raise
        conn.execute("COMMIT")
    return SCHEMA_VERSION - current


def _split(script: str) -> list[str]:
    """Split a migration script into statements (our DDL has no ``;`` in literals)."""
    return [s.strip() for s in script.split(";") if s.strip()]
