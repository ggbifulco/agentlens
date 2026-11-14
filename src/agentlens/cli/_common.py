"""Shared CLI plumbing: global state, the store, error handling and exit codes."""

from __future__ import annotations

import os
import sqlite3
import sys
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import NoReturn

import typer

from agentlens.compare.gates import EXIT_GATE_FAILED, EXIT_OK
from agentlens.report import ReportError, ReportFormat, ReportFormatUnavailableError
from agentlens.store import JSONLFormatError, SQLiteStore, StoreError
from agentlens.suite import ExperimentNotFoundError, SuiteError

__all__ = [
    "DB_ENV",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "State",
    "auto_format",
    "db_exists",
    "fail",
    "open_store",
    "user_errors",
]

EXIT_USAGE = 2
"""Usage or configuration error: bad arguments, invalid suite/gates file, unknown
experiment, unreadable file. (Click uses the same code for bad arguments.)"""

DB_ENV = "AGENTLENS_DB"
"""Environment variable with the database path (``--db`` wins)."""

_MEMORY = ":memory:"


@dataclass
class State:
    """Options of the ``agentlens`` command shared by every subcommand."""

    db: Path


def fail(message: str) -> NoReturn:
    """Print ``error: message`` to stderr and exit with :data:`EXIT_USAGE`."""
    typer.echo(f"error: {message}", err=True)
    raise typer.Exit(EXIT_USAGE)


@contextmanager
def user_errors() -> Iterator[None]:
    """Turn expected failures (bad config, unknown ids, unreadable files) into exit 2.

    Bugs still raise with a traceback, so they are not mistaken for user errors.
    """
    try:
        yield
    except SuiteError as exc:  # includes GatesError and AgentImportError
        fail(str(exc))
    except (
        ExperimentNotFoundError,
        StoreError,
        JSONLFormatError,
        ReportError,
        ReportFormatUnavailableError,
        sqlite3.DatabaseError,
    ) as exc:
        fail(str(exc))
    except OSError as exc:
        fail(f"{exc.strerror or exc}: {exc.filename}" if exc.filename else str(exc))


def auto_format() -> ReportFormat:
    """The report format for stdout when none was asked for.

    ``rich`` (colours, tables) when stdout is an interactive terminal, plain
    ``text`` otherwise: piped, redirected to a file, in CI logs, or when
    ``$NO_COLOR`` is set or ``$TERM`` is ``dumb``. ``$FORCE_COLOR`` is not
    honoured here, so CI logs and scripts that parse the output keep the
    stable plain text; ``--format rich`` asks for the rich view explicitly.
    """
    if os.environ.get("NO_COLOR") or os.environ.get("TERM") == "dumb":
        return ReportFormat.TEXT
    try:
        interactive = sys.stdout.isatty()
    except (AttributeError, ValueError):  # replaced or closed stdout
        interactive = False
    return ReportFormat.RICH if interactive else ReportFormat.TEXT


def _state(ctx: typer.Context) -> State:
    state = ctx.find_object(State)
    if state is None:  # pragma: no cover - the root callback always sets it
        raise RuntimeError("agentlens CLI state is missing")
    return state


def db_exists(ctx: typer.Context) -> bool:
    """Whether the database chosen with ``--db`` / ``$AGENTLENS_DB`` exists (in-memory: no)."""
    db = _state(ctx).db
    return str(db) != _MEMORY and db.exists()


@contextmanager
def open_store(ctx: typer.Context, *, create: bool = False) -> Iterator[SQLiteStore]:
    """Open the database chosen with ``--db`` / ``$AGENTLENS_DB``.

    Commands that only read require it to exist (``create=False``), so a typo
    in the path is an error instead of an empty new database.
    """
    db = _state(ctx).db
    if not create and str(db) != _MEMORY and not db.exists():
        fail(f"no AgentLens database at {db} (run `agentlens run` first, or pass --db)")
    with user_errors(), SQLiteStore(db) as store:
        yield store
