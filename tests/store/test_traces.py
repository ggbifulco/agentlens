"""Saving, reading and filtering traces."""

from __future__ import annotations

import pytest

import agentlens
from agentlens.schema import Experiment, Span, SpanKind, Trace, Usage
from agentlens.store import SQLiteStore, Store, StoreError
from tests.store.conftest import make_trace


def test_sqlite_store_satisfies_protocol(store: SQLiteStore) -> None:
    assert isinstance(store, Store)


def test_save_and_get_preserves_everything(store: SQLiteStore) -> None:
    trace = make_trace(metadata={"case_id": "c1", "nested": {"a": [1, 2.5, None]}})
    llm = Span(
        trace_id=trace.id,
        kind=SpanKind.LLM_CALL,
        name="gpt",
        start_time=trace.start_time,
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.001),
        attributes={"temperature": 0},
    )
    tool = Span(
        trace_id=trace.id,
        parent_id=llm.id,
        kind=SpanKind.TOOL_CALL,
        name="lookup",
        start_time=trace.start_time,
        input={"id": 1},
    )
    trace.spans = [llm, tool]

    store.save_trace(trace)

    restored = store.get_trace(trace.id)
    assert restored == trace
    assert restored is not None
    assert restored.total_cost == pytest.approx(0.001)
    assert restored.tool_calls == ["lookup"]


def test_get_unknown_trace_returns_none(store: SQLiteStore) -> None:
    assert store.get_trace("missing") is None


def test_save_is_an_upsert(store: SQLiteStore) -> None:
    trace = make_trace()
    store.save_trace(trace)
    trace.output = "changed"
    store.save_trace(trace)

    assert store.list_traces() == [trace]
    restored = store.get_trace(trace.id)
    assert restored is not None
    assert restored.output == "changed"


def test_list_is_most_recent_first_with_limit(store: SQLiteStore) -> None:
    traces = [make_trace(minute=m) for m in (5, 0, 10)]
    for t in traces:
        store.save_trace(t)

    listed = store.list_traces()
    assert [t.start_time.minute for t in listed] == [10, 5, 0]
    assert store.list_traces(limit=2) == listed[:2]
    assert store.list_traces(limit=0) == []


def test_negative_limit_is_rejected(store: SQLiteStore) -> None:
    with pytest.raises(ValueError, match="limit"):
        store.list_traces(limit=-1)


def test_filter_by_name(store: SQLiteStore) -> None:
    a, b = make_trace("a"), make_trace("b")
    store.save_trace(a)
    store.save_trace(b)

    assert store.list_traces(name="a") == [a]
    assert store.list_traces(name="zzz") == []


def test_filter_by_experiment(store: SQLiteStore) -> None:
    exp = Experiment(name="pr-1", suite="support")
    store.save_experiment(exp)
    inside, outside = make_trace(minute=1), make_trace(minute=2)
    store.save_trace(inside, experiment_id=exp.id)
    store.save_trace(outside)

    assert store.list_traces(experiment_id=exp.id) == [inside]


def test_resave_without_experiment_keeps_link(store: SQLiteStore) -> None:
    exp = Experiment(name="pr-1", suite="support")
    store.save_experiment(exp)
    trace = make_trace()
    store.save_trace(trace, experiment_id=exp.id)
    store.save_trace(trace)

    assert store.list_traces(experiment_id=exp.id) == [trace]


def test_unknown_experiment_is_a_store_error(store: SQLiteStore) -> None:
    with pytest.raises(StoreError, match="integrity"):
        store.save_trace(make_trace(), experiment_id="nope")
    assert store.list_traces() == []  # the failed write was rolled back


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("refund", ["str"]),
        (3, ["int", "float"]),
        (3.0, ["int", "float"]),
        (True, ["true"]),
        (False, ["false"]),
        (None, ["null"]),
    ],
)
def test_filter_by_metadata_value(store: SQLiteStore, value: object, expected: list[str]) -> None:
    for label, v in [
        ("str", "refund"),
        ("int", 3),
        ("float", 3.0),
        ("true", True),
        ("false", False),
        ("null", None),
        ("other", "x"),
    ]:
        store.save_trace(make_trace(label, metadata={"k": v}))
    store.save_trace(make_trace("missing", metadata={"other": 1}))

    found = store.list_traces(metadata={"k": value})  # type: ignore[dict-item]
    assert sorted(t.name for t in found) == sorted(expected)


def test_metadata_filters_combine_with_and(store: SQLiteStore) -> None:
    match = make_trace(metadata={"case_id": "c1", "repetition": 0})
    store.save_trace(match)
    store.save_trace(make_trace(metadata={"case_id": "c1", "repetition": 1}))
    store.save_trace(make_trace(metadata={"case_id": "c2", "repetition": 0}))

    assert store.list_traces(metadata={"case_id": "c1", "repetition": 0}) == [match]


def test_metadata_keys_with_special_characters(store: SQLiteStore) -> None:
    trace = make_trace(metadata={'we.ird "key"': "v"})
    store.save_trace(trace)
    assert store.list_traces(metadata={'we.ird "key"': "v"}) == [trace]


def test_metadata_filter_rejects_non_scalars(store: SQLiteStore) -> None:
    with pytest.raises(TypeError, match="scalars"):
        store.list_traces(metadata={"k": [1]})  # type: ignore[dict-item]


def test_store_works_as_tracing_sink(store: SQLiteStore) -> None:
    with agentlens.run("demo", input="q", metadata={"case_id": "c1"}, sink=store.save_trace) as t:
        with agentlens.span("lookup", kind=SpanKind.TOOL_CALL):
            pass
        t.output = "a"

    [saved] = store.list_traces(metadata={"case_id": "c1"})
    assert saved == t
    assert isinstance(saved, Trace)
    assert saved.tool_calls == ["lookup"]
