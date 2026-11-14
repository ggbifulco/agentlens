from __future__ import annotations

import pytest

from agentlens.compare import (
    COST,
    LATENCY,
    PASS_RATE,
    STEPS,
    CaseMetrics,
    Direction,
    collect_metrics,
    metric_direction,
    run_metrics,
    score_metric,
)
from agentlens.schema import Experiment, Trace
from agentlens.store import SQLiteStore
from agentlens.suite import ExperimentNotFoundError
from tests.compare.conftest import make_trace, metrics, result, run


def test_metric_direction() -> None:
    assert metric_direction(PASS_RATE) is Direction.HIGHER_IS_BETTER
    assert metric_direction(score_metric("exact_match")) is Direction.HIGHER_IS_BETTER
    for name in (COST, LATENCY, STEPS):
        assert metric_direction(name) is Direction.LOWER_IS_BETTER
    with pytest.raises(ValueError, match="unknown metric"):
        metric_direction("tokens")


def test_run_metrics_from_trace() -> None:
    trace = make_trace("c1", 2, cost=0.04, latency_ms=250, steps=4)
    m = run_metrics(trace, [result("a", 1.0), result("a", 0.5), result("b", 0.25)])
    assert m.case_id == "c1"
    assert m.repetition == 2
    assert m.passed is False  # b failed
    assert m.cost_usd == pytest.approx(0.04)
    assert m.latency_ms == pytest.approx(250)
    assert m.steps == 4
    assert m.scores == {"a": 0.75, "b": 0.25}


def test_run_metrics_error_fails_and_unknown_cost() -> None:
    m = run_metrics(make_trace("c1", error="ValueError: boom", cost=None), [])
    assert m.passed is False
    assert m.error == "ValueError: boom"
    assert m.cost_usd is None


def test_run_metrics_fallbacks_without_suite_metadata() -> None:
    trace = Trace(name="adhoc", metadata={"repetition": True})
    m = run_metrics(trace, [])
    assert (m.case_id, m.repetition) == ("adhoc", 0)
    assert m.latency_ms is None
    assert m.passed is True
    explicit = run_metrics(trace, [], case_id="x", repetition=3)
    assert (explicit.case_id, explicit.repetition) == ("x", 3)


def test_case_metrics_averages_runs() -> None:
    case = CaseMetrics.from_runs(
        "c",
        [
            run("c", passed=True, cost=0.02, latency_ms=100, steps=2, scores={"s": 1.0}),
            run("c", passed=False, cost=None, latency_ms=300, steps=4, scores={"s": 0.0}),
            run("c", passed=True, cost=0.04, latency_ms=200, steps=3),
        ],
    )
    assert case.runs == 3
    assert case.pass_rate == pytest.approx(2 / 3)
    assert case.values[COST] == pytest.approx(0.03)  # unknown cost skipped, not zero
    assert case.values[LATENCY] == pytest.approx(200)
    assert case.values[STEPS] == pytest.approx(3)
    assert case.values[score_metric("s")] == pytest.approx(0.5)


def test_case_metrics_omits_unknown_metrics_and_rejects_empty() -> None:
    case = CaseMetrics.from_runs("c", [run("c", cost=None, latency_ms=None)])
    assert COST not in case.values
    assert LATENCY not in case.values
    with pytest.raises(ValueError, match="no runs"):
        CaseMetrics.from_runs("c", [])


def test_experiment_metrics_groups_by_case_in_first_seen_order() -> None:
    exp = metrics("e", [run("b"), run("a", passed=False), run("b", passed=False)])
    assert list(exp.cases) == ["b", "a"]
    assert exp.cases["b"].runs == 2
    assert exp.mean(PASS_RATE) == pytest.approx((0.5 + 0.0) / 2)
    assert exp.mean("score:none") is None
    assert exp.metric_names == {PASS_RATE, COST, LATENCY, STEPS}


def test_collect_metrics_from_store() -> None:
    with SQLiteStore(":memory:") as store:
        exp = Experiment(name="main", suite="suite")
        store.save_experiment(exp)
        for i, (case_id, score) in enumerate([("b", 1.0), ("a", 0.0), ("b", 0.5)]):
            trace = make_trace(case_id, i, offset_s=i)
            store.save_trace(trace, experiment_id=exp.id)
            store.save_eval_results(trace.id, [result("judge", score)])
        # A trace of another experiment is ignored.
        store.save_trace(make_trace("zzz"))

        collected = collect_metrics(store, "main")
        assert collected.experiment.id == exp.id
        assert list(collected.cases) == ["b", "a"]
        assert collected.cases["b"].values[score_metric("judge")] == pytest.approx(0.75)
        assert collected.cases["b"].pass_rate == pytest.approx(1.0)
        assert collected.cases["a"].pass_rate == 0.0

        with pytest.raises(ExperimentNotFoundError):
            collect_metrics(store, "nope")
