"""Experiments and evaluation results."""

from __future__ import annotations

from datetime import timedelta

import pytest

from agentlens.schema import EvalResult, Experiment
from agentlens.store import SQLiteStore, StoreError
from tests.store.conftest import T0, make_trace


def _exp(
    name: str, *, suite: str = "support", hours: int = 0, baseline: bool = False
) -> Experiment:
    return Experiment(
        name=name,
        suite=suite,
        created_at=T0 + timedelta(hours=hours),
        config={"model": "mock"},
        is_baseline=baseline,
    )


def test_save_get_and_upsert_experiment(store: SQLiteStore) -> None:
    exp = _exp("main")
    store.save_experiment(exp)
    assert store.get_experiment(exp.id) == exp

    exp.is_baseline = True
    store.save_experiment(exp)
    assert store.get_experiment(exp.id) == exp
    assert store.list_experiments() == [exp]


def test_get_unknown_experiment_returns_none(store: SQLiteStore) -> None:
    assert store.get_experiment("missing") is None


def test_list_experiments_filters_and_order(store: SQLiteStore) -> None:
    main = _exp("main", hours=0, baseline=True)
    pr = _exp("pr-1", hours=1)
    other = _exp("main", suite="billing", hours=2)
    for e in (main, pr, other):
        store.save_experiment(e)

    assert store.list_experiments() == [other, pr, main]
    assert store.list_experiments(name="main") == [other, main]
    assert store.list_experiments(suite="support") == [pr, main]
    assert store.list_experiments(is_baseline=True) == [main]
    assert store.list_experiments(is_baseline=False) == [other, pr]
    assert store.list_experiments(name="main", suite="support") == [main]


def _result(name: str, score: float) -> EvalResult:
    return EvalResult(evaluator=name, score=score, passed=score >= 0.5, details={"n": 1})


def test_eval_results_round_trip_in_order(store: SQLiteStore) -> None:
    trace = make_trace()
    store.save_trace(trace)
    results = [_result("no_error", 1.0), _result("contains", 0.0), _result("contains", 1.0)]

    store.save_eval_results(trace.id, results)

    assert store.get_eval_results(trace.id) == results


def test_saving_eval_results_replaces_previous(store: SQLiteStore) -> None:
    trace = make_trace()
    store.save_trace(trace)
    store.save_eval_results(trace.id, [_result("a", 1.0), _result("b", 1.0)])
    store.save_eval_results(trace.id, [_result("c", 0.0)])

    assert store.get_eval_results(trace.id) == [_result("c", 0.0)]
    store.save_eval_results(trace.id, [])
    assert store.get_eval_results(trace.id) == []


def test_eval_results_for_unknown_trace_fail(store: SQLiteStore) -> None:
    with pytest.raises(StoreError, match="unknown trace"):
        store.save_eval_results("missing", [_result("a", 1.0)])
    assert store.get_eval_results("missing") == []


def test_list_eval_results_by_experiment(store: SQLiteStore) -> None:
    exp, other = _exp("pr-1"), _exp("main")
    store.save_experiment(exp)
    store.save_experiment(other)
    t1, t2, t3, t_other = (make_trace(minute=m) for m in (1, 2, 3, 4))
    for t in (t1, t2, t3):
        store.save_trace(t, experiment_id=exp.id)
    store.save_trace(t_other, experiment_id=other.id)
    store.save_eval_results(t1.id, [_result("a", 1.0)])
    store.save_eval_results(t2.id, [_result("a", 0.0), _result("b", 1.0)])
    store.save_eval_results(t_other.id, [_result("a", 1.0)])

    grouped = store.list_eval_results(exp.id)

    assert list(grouped) == [t1.id, t2.id]  # t3 has no results; trace start order
    assert grouped[t2.id] == [_result("a", 0.0), _result("b", 1.0)]
    assert store.list_eval_results("missing") == {}
