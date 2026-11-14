from __future__ import annotations

import json
from datetime import UTC, datetime, timedelta

import pytest

from agentlens.cli import EXIT_OK, EXIT_USAGE
from agentlens.schema import EvalResult, Experiment, Span, SpanKind, Trace, Usage
from agentlens.store import SQLiteStore
from agentlens.suite import get_baseline
from tests.cli.conftest import Toy

T0 = datetime(2026, 1, 1, tzinfo=UTC)

# -- baseline ------------------------------------------------------------------------


def test_baseline_show_all_and_by_suite(ran: Toy) -> None:
    result = ran("baseline", "show")
    assert result.exit_code == EXIT_OK
    lines = result.stdout.splitlines()
    assert lines[0].split() == ["id", "name", "suite", "created", "(UTC)", "model", "baseline"]
    assert len(lines) == 2
    assert " main " in lines[1]
    assert lines[1].endswith("yes")
    assert ran("baseline", "show", "toy").stdout == result.stdout


def test_baseline_set_and_unset(ran: Toy) -> None:
    result = ran("baseline", "set", "pr-1")
    assert result.exit_code == EXIT_OK
    assert result.stdout.startswith("baseline of suite 'toy': pr-1 (")
    with SQLiteStore(ran.db) as store:
        baseline = get_baseline(store, "toy")
        assert baseline is not None
        assert baseline.name == "pr-1"

    result = ran("baseline", "unset", "pr-1")
    assert result.exit_code == EXIT_OK
    assert "is not a baseline" in result.stdout
    assert ran("baseline", "show").stdout == "no baselines set\n"
    missing = ran("baseline", "show", "toy")
    assert missing.exit_code == EXIT_USAGE
    assert "suite 'toy' has no baseline" in missing.stderr


@pytest.mark.parametrize("action", ["set", "unset"])
def test_baseline_unknown_experiment(ran: Toy, action: str) -> None:
    result = ran("baseline", action, "nope")
    assert result.exit_code == EXIT_USAGE
    assert "no experiment with id or name 'nope'" in result.stderr


# -- experiments ---------------------------------------------------------------------


def test_experiments_ls(ran: Toy) -> None:
    result = ran("experiments", "ls")
    assert result.exit_code == EXIT_OK
    names = [line.split()[1] for line in result.stdout.splitlines()[1:]]
    assert names == ["pr-1", "main"]
    assert len(ran("experiments", "ls", "--limit", "1").stdout.splitlines()) == 2
    assert " main " in ran("experiments", "ls", "-n", "main").stdout
    assert ran("experiments", "ls", "--suite", "other").stdout == "no experiments\n"


def test_experiments_ls_shows_model(toy: Toy) -> None:
    assert toy("run", str(toy.suite), "-q", "-r", "1", "-m", "mock:x").exit_code == EXIT_OK
    assert "mock:x" in toy("experiments", "ls").stdout


# -- traces --------------------------------------------------------------------------


def test_traces_ls_filters(ran: Toy) -> None:
    result = ran("traces", "ls")
    assert result.exit_code == EXIT_OK
    lines = result.stdout.splitlines()
    assert lines[0].split() == [
        "id",
        "name",
        "started",
        "(UTC)",
        "case",
        "ms",
        "steps",
        "cost",
        "$",
        "status",
    ]
    assert len(lines) == 21  # header + default limit 20

    pr = ran("traces", "ls", "--experiment", "pr-1", "--limit", "50").stdout.splitlines()
    assert len(pr) == 13
    assert sum(line.endswith("error") for line in pr) == 2
    assert len(ran("traces", "ls", "-n", "toy", "-l", "3").stdout.splitlines()) == 4
    assert ran("traces", "ls", "--name", "other").stdout == "no traces\n"


def test_traces_ls_unknown_experiment(ran: Toy) -> None:
    result = ran("traces", "ls", "-e", "nope")
    assert result.exit_code == EXIT_USAGE


def _rich_trace() -> Trace:
    trace_id = "t-rich"
    root = Span(
        id="s1",
        trace_id=trace_id,
        kind=SpanKind.AGENT_STEP,
        name="plan",
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=30),
    )
    llm = Span(
        id="s2",
        trace_id=trace_id,
        parent_id="s1",
        kind=SpanKind.LLM_CALL,
        name="gpt",
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=20),
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.002),
    )
    tool = Span(
        id="s3",
        trace_id=trace_id,
        parent_id="s1",
        kind=SpanKind.TOOL_CALL,
        name="lookup",
        start_time=T0,
        usage=Usage(input_tokens=1),
        error="KeyError: 'order'",
    )
    return Trace(
        id=trace_id,
        name="support",
        input={"question": "where is " + "my order " * 20},
        output="It shipped.",
        error="RuntimeError: gave up",
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=42),
        spans=[root, llm, tool],
        metadata={"case_id": "c1"},
    )


@pytest.fixture
def stored(toy: Toy) -> Toy:
    with SQLiteStore(toy.db) as store:
        store.save_trace(_rich_trace())
        store.save_eval_results(
            "t-rich",
            [
                EvalResult(evaluator="no_error", score=0.0, passed=False, reason="trace error"),
                EvalResult(evaluator="max_steps", score=1.0, passed=True),
            ],
        )
        store.save_trace(Trace(id="t-plain", name="plain", start_time=T0))
    return toy


def test_traces_show_text(stored: Toy) -> None:
    result = stored("traces", "show", "t-rich")
    assert result.exit_code == EXIT_OK, result.output
    out = result.stdout
    assert "trace     t-rich" in out
    assert "started   2026-01-01 00:00:00 UTC  (42 ms)" in out
    assert "steps     1  tokens 16  cost $0.002" in out
    assert 'input     {"question": "where is my order' in out
    assert out.splitlines()[4].endswith("...")
    assert "output    It shipped." in out
    assert "error     RuntimeError: gave up" in out
    assert 'metadata  {"case_id": "c1"}' in out
    assert "  - agent_step plan  30 ms" in out
    assert "    - llm_call gpt  20 ms  10+5 tok  $0.002" in out
    assert "    - tool_call lookup  - ms  1+0 tok  error: KeyError: 'order'" in out
    assert "  - FAIL no_error 0: trace error" in out
    assert "  - pass max_steps 1\n" in out


def test_traces_show_minimal(stored: Toy) -> None:
    out = stored("traces", "show", "t-plain").stdout
    assert "started   2026-01-01 00:00:00 UTC  (- ms)" in out
    assert "cost $-" in out
    assert "input     null" in out
    for absent in ("error", "metadata", "spans", "evaluations"):
        assert f"\n{absent}" not in out


def test_traces_show_json(stored: Toy) -> None:
    result = stored("traces", "show", "t-rich", "--json")
    assert result.exit_code == EXIT_OK
    assert Trace.model_validate_json(result.stdout) == _rich_trace()


def test_traces_show_unknown(stored: Toy) -> None:
    result = stored("traces", "show", "nope")
    assert result.exit_code == EXIT_USAGE
    assert "no trace with id 'nope'" in result.stderr


# -- export / import -----------------------------------------------------------------


def test_export_and_import_round_trip(ran: Toy) -> None:
    out = ran.root / "export" / "pr.jsonl"
    result = ran("export", str(out), "--experiment", "pr-1")
    assert result.exit_code == EXIT_OK
    assert result.stdout == f"exported 12 traces to {out}\n"
    assert len(out.read_text().splitlines()) == 12

    other = ran.root / "other.db"
    imported = ran("--db", str(other), "import", str(out), db=False)
    assert imported.exit_code == EXIT_OK, imported.output
    assert imported.stdout == f"imported 12 traces from {out}\n"
    with SQLiteStore(other) as store:
        assert len(store.list_traces()) == 12


def test_export_filters(ran: Toy) -> None:
    out = ran.root / "one.jsonl"
    assert ran("export", str(out), "-n", "toy", "-l", "1").stdout == f"exported 1 trace to {out}\n"
    assert json.loads(out.read_text())["name"] == "toy"


def test_import_links_to_an_experiment(toy: Toy) -> None:
    with SQLiteStore(toy.db) as store:
        store.save_experiment(Experiment(id="e1", name="imported", suite="toy"))
    data = toy.write("t.jsonl", Trace(id="t1", name="x", start_time=T0).model_dump_json() + "\n")
    result = toy("import", str(data), "--experiment", "imported")
    assert result.exit_code == EXIT_OK
    assert result.stdout == f"imported 1 trace from {data}\n"
    with SQLiteStore(toy.db) as store:
        assert [t.id for t in store.list_traces(experiment_id="e1")] == ["t1"]


def test_import_creates_the_database(toy: Toy) -> None:
    data = toy.write("t.jsonl", Trace(name="x", start_time=T0).model_dump_json() + "\n")
    assert toy("import", str(data)).exit_code == EXIT_OK
    assert toy.db.exists()


@pytest.mark.parametrize(
    ("content", "message"),
    [(None, "No such file"), ('{"name": 1}\n', "t.jsonl:1")],
)
def test_import_errors(toy: Toy, content: str | None, message: str) -> None:
    path = toy.root / "t.jsonl"
    if content is not None:
        path.write_text(content)
    result = toy("import", str(path))
    assert result.exit_code == EXIT_USAGE
    assert message in result.stderr
