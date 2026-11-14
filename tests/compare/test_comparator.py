from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path

import pytest

from agentlens.compare import (
    CaseComparison,
    Comparison,
    Direction,
    Status,
    classify,
    compare_experiments,
    compare_runs,
    experiment_runs,
    metric_direction,
)
from agentlens.schema import Case, Experiment, Trace
from agentlens.store import SQLiteStore
from agentlens.suite import CaseResult, ExperimentNotFoundError, ExperimentRun, RunResult, Suite
from tests.compare.helpers import T0, cases, run

# -- direction and classification ------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "direction"),
    [
        ("pass_rate", Direction.HIGHER_IS_BETTER),
        ("score.exact_match", Direction.HIGHER_IS_BETTER),
        ("cost_usd", Direction.LOWER_IS_BETTER),
        ("latency_ms", Direction.LOWER_IS_BETTER),
        ("steps", Direction.LOWER_IS_BETTER),
    ],
)
def test_metric_direction(name: str, direction: Direction) -> None:
    assert metric_direction(name) is direction


@pytest.mark.parametrize(
    ("direction", "ci", "status"),
    [
        (Direction.HIGHER_IS_BETTER, (-0.3, -0.1), Status.REGRESSION),
        (Direction.HIGHER_IS_BETTER, (0.1, 0.3), Status.IMPROVEMENT),
        (Direction.HIGHER_IS_BETTER, (-0.1, 0.1), Status.UNCHANGED),
        (Direction.HIGHER_IS_BETTER, (-0.1, 0.0), Status.UNCHANGED),
        (Direction.LOWER_IS_BETTER, (0.1, 0.3), Status.REGRESSION),
        (Direction.LOWER_IS_BETTER, (-0.3, -0.1), Status.IMPROVEMENT),
        (Direction.LOWER_IS_BETTER, (0.0, 0.0), Status.UNCHANGED),
    ],
)
def test_classify(direction: Direction, ci: tuple[float, float], status: Status) -> None:
    assert classify(direction, *ci) is status


# -- compare_runs ------------------------------------------------------------------------


def test_identical_runs_are_unchanged() -> None:
    runs = cases(5, cost=0.01, latency_ms=200.0, steps=2)
    comparison = compare_runs(runs, runs)
    assert [m.name for m in comparison.metrics] == [
        "pass_rate",
        "score.check",
        "cost_usd",
        "latency_ms",
        "steps",
    ]
    for metric in comparison.metrics:
        assert metric.status is Status.UNCHANGED
        assert metric.delta == 0.0
        assert (metric.ci_low, metric.ci_high) == (0.0, 0.0)
        assert metric.n == 5
    assert not comparison.has_regressions
    assert comparison.case_regressions == comparison.case_improvements == []


def test_pass_rate_drop_on_many_cases_is_a_regression() -> None:
    baseline = cases(20)
    candidate = [run(f"c{i}", passed=i % 4 != 0) for i in range(20)]  # 5 of 20 now fail
    comparison = compare_runs(baseline, candidate)
    pass_rate = comparison.metric("pass_rate")
    assert pass_rate.baseline == 1.0
    assert pass_rate.candidate == pytest.approx(0.75)
    assert pass_rate.delta == pytest.approx(-0.25)
    assert pass_rate.delta_pct == pytest.approx(-25.0)
    assert pass_rate.ci_high is not None
    assert pass_rate.ci_high < 0
    assert pass_rate.status is Status.REGRESSION
    assert pass_rate.is_regression
    assert not pass_rate.is_improvement
    assert pass_rate.direction is Direction.HIGHER_IS_BETTER
    assert comparison.has_regressions
    assert {m.name for m in comparison.regressions} == {"pass_rate", "score.check"}
    regressed = comparison.case_regressions
    assert [c.case_id for c in regressed] == ["c0", "c4", "c8", "c12", "c16"]
    assert regressed[0].regressed_evaluators == ["check"]
    assert regressed[0].pass_rate.delta == -1.0


def test_single_flip_is_not_an_aggregate_regression() -> None:
    baseline = cases(20)
    candidate = [run(f"c{i}", passed=i != 3) for i in range(20)]
    comparison = compare_runs(baseline, candidate)
    assert comparison.metric("pass_rate").status is Status.UNCHANGED
    assert [c.case_id for c in comparison.case_regressions] == ["c3"]


def test_lower_is_better_metrics() -> None:
    baseline = [run(f"c{i}", cost=0.010, latency_ms=100 + i, steps=3) for i in range(10)]
    candidate = [run(f"c{i}", cost=0.015, latency_ms=50 + i, steps=3) for i in range(10)]
    comparison = compare_runs(baseline, candidate)
    cost = comparison.metric("cost_usd")
    assert cost.status is Status.REGRESSION
    assert cost.direction is Direction.LOWER_IS_BETTER
    assert cost.delta_pct == pytest.approx(50.0)
    latency = comparison.metric("latency_ms")
    assert latency.status is Status.IMPROVEMENT
    assert latency.delta == pytest.approx(-50.0)
    assert comparison.metric("steps").status is Status.UNCHANGED
    assert [m.name for m in comparison.improvements] == ["latency_ms"]


def test_values_are_averaged_over_repetitions() -> None:
    baseline = [run("a", r, passed=True, latency_ms=100.0) for r in range(3)]
    candidate = [
        run("a", 0, passed=True, latency_ms=100.0),
        run("a", 1, passed=False, latency_ms=200.0),
        run("a", 2, passed=True, latency_ms=300.0),
    ]
    case = compare_runs(baseline, candidate).case("a")
    assert (case.baseline_runs, case.candidate_runs) == (3, 3)
    assert case.pass_rate.candidate == pytest.approx(2 / 3)
    assert case.metrics["latency_ms"].candidate == pytest.approx(200.0)
    assert case.metrics["latency_ms"].delta == pytest.approx(100.0)
    assert case.status is Status.REGRESSION


def test_error_runs_fail_even_when_evaluators_pass() -> None:
    comparison = compare_runs([run("a")], [run("a", error="ValueError: boom")])
    assert comparison.case("a").pass_rate.candidate == 0.0
    assert comparison.case("a").regressed_evaluators == []


def test_scores_per_evaluator() -> None:
    baseline = [
        run(f"c{i}", passed={"a": True, "b": True}, score={"a": 0.9, "b": 0.5}) for i in range(8)
    ]
    candidate = [
        run(f"c{i}", passed={"a": True, "b": True}, score={"a": 0.6, "b": 0.5}) for i in range(8)
    ]
    comparison = compare_runs(baseline, candidate)
    score_a = comparison.metric("score.a")
    assert score_a.evaluator == "a"
    assert score_a.delta == pytest.approx(-0.3)
    assert score_a.status is Status.REGRESSION
    assert comparison.metric("score.b").status is Status.UNCHANGED
    assert comparison.metric("pass_rate").status is Status.UNCHANGED


def test_cost_is_none_aware() -> None:
    baseline = [run("a", cost=0.01), run("b", cost=None), run("c", cost=0.02)]
    candidate = [run("a", cost=0.03), run("b", cost=0.05), run("c", cost=None)]
    comparison = compare_runs(baseline, candidate)
    cost = comparison.metric("cost_usd")
    assert cost.n == 1  # only case "a" has a cost on both sides
    assert cost.delta == pytest.approx(0.02)
    assert comparison.case("b").metrics["cost_usd"].baseline is None
    assert comparison.case("b").metrics["cost_usd"].delta is None


def test_no_cost_anywhere_is_no_data() -> None:
    comparison = compare_runs(cases(3), cases(3))
    cost = comparison.metric("cost_usd")
    assert cost.status is Status.NO_DATA
    assert cost.n == 0
    assert cost.delta is cost.ci_low is cost.baseline is None
    assert "cost_usd" not in comparison.case("c0").metrics


def test_cost_known_on_some_repetitions_only() -> None:
    baseline = [run("a", 0, cost=0.01), run("a", 1, cost=None), run("a", 2, cost=0.03)]
    case = compare_runs(baseline, baseline).case("a")
    assert case.metrics["cost_usd"].baseline == pytest.approx(0.02)


def test_zero_baseline_has_no_delta_pct() -> None:
    comparison = compare_runs(cases(4, passed=False), cases(4, passed=True))
    pass_rate = comparison.metric("pass_rate")
    assert pass_rate.delta == 1.0
    assert pass_rate.delta_pct is None
    assert pass_rate.status is Status.IMPROVEMENT
    assert len(comparison.case_improvements) == 4


def test_added_and_removed_cases_are_reported_and_not_paired() -> None:
    baseline = [run("keep"), run("gone", passed=False)]
    candidate = [run("new", passed=False), run("keep")]
    comparison = compare_runs(baseline, candidate)
    assert comparison.removed_cases == ["gone"]
    assert comparison.added_cases == ["new"]
    assert [c.case_id for c in comparison.cases] == ["keep"]
    assert comparison.metric("pass_rate").n == 1
    with pytest.raises(KeyError):
        comparison.case("gone")
    with pytest.raises(KeyError):
        comparison.metric("score.nope")


def test_disjoint_runs_have_no_data() -> None:
    comparison = compare_runs([run("a")], [run("b")])
    assert comparison.cases == []
    assert all(m.status is Status.NO_DATA for m in comparison.metrics)
    assert [m.name for m in comparison.metrics] == ["pass_rate", "cost_usd", "latency_ms", "steps"]


def test_evaluator_on_one_side_only() -> None:
    baseline = [run("a", passed={"x": True})]
    candidate = [run("a", passed={"x": True, "y": False})]
    comparison = compare_runs(baseline, candidate)
    assert comparison.metric("score.y").status is Status.NO_DATA
    assert comparison.case("a").metrics["score.y"].baseline is None
    assert comparison.case("a").regressed_evaluators == []


def test_same_seed_is_reproducible_and_seed_is_recorded() -> None:
    baseline = [run(f"c{i}", latency_ms=100 + 7 * i) for i in range(15)]
    candidate = [run(f"c{i}", latency_ms=100 + (i * 13) % 29) for i in range(15)]
    first = compare_runs(baseline, candidate, seed=42, resamples=2000, confidence=0.9)
    second = compare_runs(baseline, candidate, seed=42, resamples=2000, confidence=0.9)
    assert first == second
    assert (first.seed, first.resamples, first.confidence) == (42, 2000, 0.9)


@pytest.mark.parametrize(
    ("kwargs", "message"), [({"resamples": 0}, "resamples"), ({"confidence": 1.5}, "confidence")]
)
def test_rejects_bad_arguments(kwargs: dict[str, float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        compare_runs([run("a")], [run("a")], **kwargs)  # type: ignore[arg-type]


def test_accepts_experiment_runs() -> None:
    suite = Suite(name="toy", agent=lambda x: x, cases=[{"id": "a", "input": 1}])
    case = Case(id="a", input=1)
    base_exp = Experiment(name="main", suite="toy")
    cand_exp = Experiment(name="pr", suite="toy")
    baseline = ExperimentRun(base_exp, suite, [CaseResult(case, [run("a")])])
    candidate = ExperimentRun(cand_exp, suite, [CaseResult(case, [run("a", passed=False)])])
    comparison = compare_runs(baseline, candidate)
    assert comparison.baseline == base_exp
    assert comparison.candidate == cand_exp
    assert comparison.case_regressions[0].case_id == "a"


def test_comparison_round_trips_through_json() -> None:
    baseline = [run(f"c{i}", cost=0.01) for i in range(5)]
    candidate = [run(f"c{i}", cost=0.02, passed=i > 1) for i in range(5)] + [run("new")]
    comparison = compare_runs(baseline, candidate, resamples=500)
    restored = Comparison.model_validate_json(comparison.model_dump_json())
    assert restored == comparison
    assert isinstance(restored.cases[0], CaseComparison)
    assert restored.metric("cost_usd").status is Status.REGRESSION


# -- from a store -------------------------------------------------------------------------


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteStore]:
    with SQLiteStore(tmp_path / "runs.db") as db:
        yield db


def save(store: SQLiteStore, name: str, runs: list[RunResult]) -> Experiment:
    experiment = Experiment(name=name, suite="toy")
    store.save_experiment(experiment)
    for result in runs:
        store.save_trace(result.trace, experiment_id=experiment.id)
        store.save_eval_results(result.trace.id, result.eval_results)
    return experiment


def test_compare_experiments_from_store(store: SQLiteStore) -> None:
    baseline = [run(f"c{i}", r, cost=0.01) for i in range(10) for r in range(2)]
    candidate = [run(f"c{i}", r, cost=0.01, passed=i >= 5) for i in range(10) for r in range(2)]
    base_exp = save(store, "main", baseline)
    cand_exp = save(store, "pr-1", candidate)

    comparison = compare_experiments(store, "main", cand_exp.id, resamples=1000)
    assert comparison.baseline == base_exp
    assert comparison.candidate == cand_exp
    assert comparison.metric("pass_rate").delta == pytest.approx(-0.5)
    assert comparison.metric("pass_rate").status is Status.REGRESSION
    assert sorted(c.case_id for c in comparison.case_regressions) == [f"c{i}" for i in range(5)]
    assert comparison.case("c0").baseline_runs == 2
    assert comparison == compare_runs(
        baseline,
        candidate,
        resamples=1000,
        baseline_experiment=base_exp,
        candidate_experiment=cand_exp,
    ).model_copy(update={"cases": comparison.cases})


def test_experiment_runs_rebuilds_runs(
    store: SQLiteStore, caplog: pytest.LogCaptureFixture
) -> None:
    experiment = save(store, "main", [run("a", 0), run("a", 1, passed=False)])
    stray = Trace(name="manual", start_time=T0, end_time=T0)
    store.save_trace(stray, experiment_id=experiment.id)
    no_rep = Trace(name="toy", start_time=T0, end_time=T0, metadata={"case_id": "b"})
    store.save_trace(no_rep, experiment_id=experiment.id)

    with caplog.at_level(logging.WARNING, logger="agentlens.compare"):
        runs = experiment_runs(store, experiment)
    assert sorted((r.case_id, r.repetition, r.passed) for r in runs) == [
        ("a", 0, True),
        ("a", 1, False),
        ("b", 0, True),
    ]
    assert "skipped 1 trace" in caplog.text


def test_compare_experiments_unknown_experiment(store: SQLiteStore) -> None:
    save(store, "main", [run("a")])
    with pytest.raises(ExperimentNotFoundError):
        compare_experiments(store, "main", "nope")
