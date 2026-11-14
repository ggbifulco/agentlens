"""UI fixtures: a database with two experiments, a standalone trace and hostile strings,
and a live server on an ephemeral port."""

from __future__ import annotations

import threading
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import httpx
import pytest

from agentlens.schema import Experiment, Span, SpanKind, Trace
from agentlens.store import SQLiteStore
from agentlens.suite import RunResult
from agentlens.ui import UIApp, UIServer, make_server
from tests.compare.helpers import T0, run

EVIL = "</script><script>alert(1)</script>"


@dataclass
class Data:
    """What the fixture database holds."""

    db: Path
    base: Experiment
    cand: Experiment
    base_runs: list[RunResult]
    cand_runs: list[RunResult]
    standalone: Trace
    evil: Trace


def _save(store: SQLiteStore, exp: Experiment, runs: list[RunResult], start: int) -> None:
    """Save ``runs`` under ``exp``, one minute apart from ``start`` (``runs`` is updated)."""
    store.save_experiment(exp)
    for i, result in enumerate(runs):
        shift = timedelta(minutes=start + i)
        old = result.trace
        assert old.end_time is not None
        trace = old.model_copy(
            update={
                "start_time": old.start_time + shift,
                "end_time": old.end_time + shift,
                "spans": [
                    span.model_copy(
                        update={
                            "start_time": span.start_time + shift,
                            "end_time": (span.end_time or span.start_time) + shift,
                        }
                    )
                    for span in old.spans
                ],
                "metadata": {**old.metadata, "experiment": exp.id},
            }
        )
        runs[i] = RunResult(result.case_id, result.repetition, trace, result.eval_results)
        store.save_trace(trace, experiment_id=exp.id)
        store.save_eval_results(trace.id, result.eval_results)


@pytest.fixture
def data(tmp_path: Path) -> Data:
    db = tmp_path / "ui.db"
    base = Experiment(name="main", suite="toy", is_baseline=True, config={"model": "mock"})
    cand = Experiment(name="pr-1", suite="toy", created_at=base.created_at + timedelta(hours=1))
    base_runs = [run(f"q{i}", cost=0.01) for i in range(6)]
    cand_runs = [run(f"q{i}", cost=0.01, passed=i >= 3) for i in range(5)]
    cand_runs.append(run("boom", error="ValueError: boom"))
    standalone = Trace(
        name="adhoc",
        input={"question": "hi"},
        output="hello",
        start_time=T0 + timedelta(days=1),
        end_time=T0 + timedelta(days=1, seconds=1),
    )
    standalone.spans = [
        Span(
            trace_id=standalone.id,
            kind=SpanKind.TOOL_CALL,
            name="lookup",
            start_time=standalone.start_time,
            end_time=standalone.end_time,
            error="KeyError: 'order'",
        )
    ]
    evil = Trace(
        name=EVIL,
        input=EVIL,
        output=EVIL,
        start_time=T0 + timedelta(days=2),
        end_time=T0 + timedelta(days=2, seconds=1),
        metadata={"case_id": EVIL, "note": EVIL},
    )
    with SQLiteStore(db) as store:
        _save(store, base, base_runs, 0)
        _save(store, cand, cand_runs, 10)
        store.save_trace(standalone)
        store.save_trace(evil)
    return Data(db, base, cand, base_runs, cand_runs, standalone, evil)


@pytest.fixture
def app(data: Data) -> Iterator[UIApp]:
    ui = UIApp(data.db, resamples=200)
    yield ui
    ui.close()


@dataclass
class Live:
    server: UIServer
    client: httpx.Client


@pytest.fixture
def live(app: UIApp) -> Iterator[Live]:
    """The app served on 127.0.0.1 with an ephemeral port, in a background thread."""
    server = make_server(app, port=0)
    thread = threading.Thread(target=server.serve_forever, args=(0.05,), daemon=True)
    thread.start()
    with httpx.Client(base_url=server.url, timeout=10) as client:
        yield Live(server, client)
    server.shutdown()
    server.server_close()
    thread.join(5)
