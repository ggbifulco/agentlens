"""JSONL export/import of traces (FR-T4)."""

from __future__ import annotations

import json
from collections.abc import Iterator
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlens.schema import Experiment, Span, SpanKind, Trace, Usage
from agentlens.store import (
    JSONLFormatError,
    SQLiteStore,
    StoreError,
    export_jsonl,
    import_jsonl,
    read_jsonl,
    write_jsonl,
)
from tests.schema import strategies as s
from tests.store.conftest import T0, make_trace


def rich_trace() -> Trace:
    """A trace exercising every field: spans, usage, attributes, error, unicode."""
    trace = make_trace(metadata={"case_id": "c1", "nested": {"a": [1, 2.5, None, True]}})
    trace.error = "ValueError: boom"
    trace.output = {"text": "ciao — 你好 \n newline"}
    llm = Span(
        trace_id=trace.id,
        kind=SpanKind.LLM_CALL,
        name="gpt",
        start_time=trace.start_time,
        end_time=trace.end_time,
        input=[{"role": "user", "content": "hi"}],
        output="hello",
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
        attributes={"temperature": 0, "model": "mock"},
    )
    tool = Span(
        trace_id=trace.id,
        parent_id=llm.id,
        kind=SpanKind.TOOL_CALL,
        name="lookup",
        start_time=trace.start_time,
        input={"id": 1},
        error="KeyError: 'id'",
    )
    trace.spans = [llm, tool]
    return trace


def test_round_trip_through_two_stores_preserves_all_fields(tmp_path: Path) -> None:
    traces = [rich_trace(), make_trace("other", minute=1)]
    path = tmp_path / "traces.jsonl"
    with SQLiteStore(tmp_path / "a.db") as source:
        for trace in traces:
            source.save_trace(trace)
        assert export_jsonl(source, path) == 2

    with SQLiteStore(tmp_path / "b.db") as target:
        assert import_jsonl(target, path) == 2
        for trace in traces:
            assert target.get_trace(trace.id) == trace
        assert target.list_traces() == source_order(traces)


def source_order(traces: list[Trace]) -> list[Trace]:
    return sorted(traces, key=lambda t: t.start_time, reverse=True)


def test_file_has_one_trace_document_per_line_oldest_first(
    store: SQLiteStore, tmp_path: Path
) -> None:
    old, new = make_trace(minute=0), make_trace(minute=5)
    store.save_trace(new)
    store.save_trace(old)
    path = tmp_path / "out.jsonl"

    export_jsonl(store, path)

    lines = path.read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [old.id, new.id]
    assert lines[0] == old.model_dump_json()


def test_export_applies_filters(store: SQLiteStore, tmp_path: Path) -> None:
    experiment = Experiment(name="exp", suite="s", created_at=T0)
    store.save_experiment(experiment)
    a = make_trace("a", minute=0, metadata={"case_id": "x"})
    b = make_trace("a", minute=1, metadata={"case_id": "y"})
    c = make_trace("b", minute=2, metadata={"case_id": "x"})
    store.save_trace(a, experiment_id=experiment.id)
    store.save_trace(b)
    store.save_trace(c, experiment_id=experiment.id)
    path = tmp_path / "out.jsonl"

    def exported(**filters: object) -> list[str]:
        export_jsonl(store, path, **filters)  # type: ignore[arg-type]
        return [t.id for t in read_jsonl(path)]

    assert exported(name="a") == [a.id, b.id]
    assert exported(experiment_id=experiment.id) == [a.id, c.id]
    assert exported(metadata={"case_id": "x"}) == [a.id, c.id]
    assert exported(name="a", metadata={"case_id": "x"}) == [a.id]
    # limit keeps the most recent traces, still written oldest first
    assert exported(limit=2) == [b.id, c.id]
    assert exported(name="missing") == []
    assert path.read_text(encoding="utf-8") == ""


def test_write_creates_parent_directories_and_leaves_no_temp_file(tmp_path: Path) -> None:
    path = tmp_path / "nested" / "dir" / "t.jsonl"
    assert write_jsonl([make_trace()], path) == 1
    assert [p.name for p in path.parent.iterdir()] == ["t.jsonl"]


def test_failed_write_keeps_existing_file_intact(tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    write_jsonl([make_trace()], path)
    before = path.read_bytes()

    def exploding() -> Iterator[Trace]:
        yield make_trace(minute=1)
        raise RuntimeError("serialisation failed")

    with pytest.raises(RuntimeError):
        write_jsonl(exploding(), path)

    assert path.read_bytes() == before
    assert [p.name for p in tmp_path.iterdir()] == ["t.jsonl"]


def test_read_skips_blank_lines(tmp_path: Path) -> None:
    trace = make_trace()
    path = tmp_path / "t.jsonl"
    path.write_text(f"\n{trace.model_dump_json()}\n   \n\n", encoding="utf-8")
    assert list(read_jsonl(path)) == [trace]


@pytest.mark.parametrize(
    ("line", "reason"),
    [
        ("not json", "Invalid JSON"),
        ('{"name": "x", "start_time": "yesterday"}', "valid datetime"),
        ("[1, 2]", "Input should be"),
    ],
)
def test_invalid_line_reports_file_and_line_number(tmp_path: Path, line: str, reason: str) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(f"{make_trace().model_dump_json()}\n\n{line}\n", encoding="utf-8")

    with pytest.raises(JSONLFormatError) as info:
        list(read_jsonl(path))

    error = info.value
    assert isinstance(error, StoreError)
    assert error.line == 3
    assert error.path == str(path)
    assert str(error).startswith(f"{path}:3: invalid trace: ")
    assert reason in str(error) or reason in str(error.__cause__)


def test_errors_are_summarised_with_location_and_count(tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text('{"start_time": "yesterday", "bogus": 1}\n', encoding="utf-8")
    with pytest.raises(JSONLFormatError, match=r":1: invalid trace: \w+: .+ \(\+\d+ more\)$"):
        list(read_jsonl(path))


def test_import_is_all_or_nothing_on_format_errors(store: SQLiteStore, tmp_path: Path) -> None:
    path = tmp_path / "bad.jsonl"
    path.write_text(f"{make_trace().model_dump_json()}\n{{broken\n", encoding="utf-8")

    with pytest.raises(JSONLFormatError):
        import_jsonl(store, path)

    assert store.list_traces() == []


def test_import_twice_is_idempotent(store: SQLiteStore, tmp_path: Path) -> None:
    path = tmp_path / "t.jsonl"
    write_jsonl([make_trace(minute=0), make_trace(minute=1)], path)

    assert import_jsonl(store, path) == 2
    assert import_jsonl(store, path) == 2
    assert len(store.list_traces()) == 2


def test_import_can_link_traces_to_an_experiment(store: SQLiteStore, tmp_path: Path) -> None:
    experiment = Experiment(name="exp", suite="s", created_at=T0)
    store.save_experiment(experiment)
    trace = make_trace()
    path = tmp_path / "t.jsonl"
    write_jsonl([trace], path)

    import_jsonl(store, path, experiment_id=experiment.id)

    assert store.list_traces(experiment_id=experiment.id) == [trace]


def test_import_into_unknown_experiment_raises_store_error(
    store: SQLiteStore, tmp_path: Path
) -> None:
    path = tmp_path / "t.jsonl"
    write_jsonl([make_trace()], path)
    with pytest.raises(StoreError):
        import_jsonl(store, path, experiment_id="missing")


def test_missing_file_raises_file_not_found(store: SQLiteStore, tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        import_jsonl(store, tmp_path / "nope.jsonl")


@settings(max_examples=50, deadline=None)
@given(traces=st.lists(s.traces(), max_size=4, unique_by=lambda t: t.id))
def test_hypothesis_write_read_round_trip(
    tmp_path_factory: pytest.TempPathFactory, traces: list[Trace]
) -> None:
    path = tmp_path_factory.mktemp("jsonl") / "t.jsonl"
    assert write_jsonl(traces, path) == len(traces)
    assert list(read_jsonl(path)) == traces
