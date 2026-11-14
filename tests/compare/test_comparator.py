from __future__ import annotations

import random
from typing import Any

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

import agentlens
from agentlens.compare import (
    COST,
    LATENCY,
    PASS_RATE,
    STEPS,
    CaseStatus,
    Comparison,
    ExperimentMetrics,
    MetricStatus,
    RunMetrics,
    compare,
    compare_experiments,
    metric_order,
    metrics_from_run,
    score_metric,
)
from agentlens.llm import LLMRequest, Message, get_client
from agentlens.store import SQLiteStore
from agentlens.suite import ExperimentNotFoundError, Suite, run_suite
from tests.compare.conftest import metrics, run

FAST: dict[str, Any] = {"resamples": 500}


def experiment(name: str, pass_rates: list[float], **run_kwargs: Any) -> ExperimentMetrics:
    """One run per case; ``pass_rates`` are 0/1 per case."""
    return metrics(
        name,
        [run(f"c{i}", passed=bool(p), **run_kwargs) for i, p in enumerate(pass_rates)],
    )


# -- verdicts ----------------------------------------------------------------------


def test_clear_regression_is_flagged() -> None:
    base = experiment("base", [1] * 30)
    cand = experiment("cand", [1] * 20 + [0] * 10)
    result = compare(base, cand, **FAST)
    pr = result.metric(PASS_RATE)
    assert pr.baseline == 1.0
    assert pr.candidate == pytest.approx(2 / 3)
    assert pr.delta == pytest.approx(-1 / 3)
    assert pr.delta_pct == pytest.approx(-100 / 3)
    assert pr.ci.high < 0
    assert pr.status is MetricStatus.REGRESSED
    assert pr.regressed
    assert result.regressions == [pr]
    assert [c.case_id for c in result.regressed_cases] == [f"c{i}" for i in range(20, 30)]
    assert result.case("c25").pass_rate_delta == -1.0


def test_improvement_is_flagged() -> None:
    result = compare(experiment("b", [0] * 25 + [1] * 5), experiment("c", [1] * 30), **FAST)
    assert result.metric(PASS_RATE).status is MetricStatus.IMPROVED
    assert result.improvements == [result.metric(PASS_RATE)]
    assert len(result.improved_cases) == 25
    assert result.regressions == []


def test_noise_is_not_a_regression() -> None:
    """Two runs of the same flaky agent: some cases flip both ways."""
    rng = random.Random(3)
    base = experiment("b", [rng.random() < 0.7 for _ in range(40)])
    cand = experiment("c", [rng.random() < 0.7 for _ in range(40)])
    pr = compare(base, cand, **FAST).metric(PASS_RATE)
    assert pr.ci.low < 0 < pr.ci.high
    assert pr.status is MetricStatus.UNCHANGED


def test_lower_is_better_metrics() -> None:
    base = metrics("b", [run(f"c{i}", cost=0.010, latency_ms=100, steps=2) for i in range(10)])
    cand = metrics(
        "c",
        [run(f"c{i}", cost=0.012 + i * 1e-4, latency_ms=80, steps=2) for i in range(10)],
    )
    result = compare(base, cand, **FAST)
    assert result.metric(COST).status is MetricStatus.REGRESSED  # more expensive
    assert result.metric(COST).delta_pct == pytest.approx(24.5)
    assert result.metric(LATENCY).status is MetricStatus.IMPROVED  # faster
    assert result.metric(STEPS).status is MetricStatus.UNCHANGED
    assert result.metric(STEPS).ci.width == 0
    # Per-case status ignores cost/latency: nothing changed in quality.
    assert {c.status for c in result.cases} == {CaseStatus.UNCHANGED}


def test_scores_and_mixed_cases() -> None:
    base = metrics("b", [run("x", scores={"judge": 0.5, "exact": 1.0}), run("y")])
    cand = metrics("c", [run("x", scores={"judge": 0.75, "exact": 0.0}), run("y")])
    result = compare(base, cand, **FAST)
    x = result.case("x")
    assert x.deltas[score_metric("judge")] == pytest.approx(0.25)
    assert x.deltas[score_metric("exact")] == -1.0
    assert x.status is CaseStatus.MIXED
    assert result.case("y").status is CaseStatus.UNCHANGED
    # Score metrics exist for one paired case only -> no verdict.
    judge = result.metric(score_metric("judge"))
    assert judge.n == 1
    assert judge.status is MetricStatus.INSUFFICIENT_DATA
    assert list(x.deltas) == [PASS_RATE, "score:exact", "score:judge", COST, LATENCY, STEPS]


def test_metrics_are_ordered() -> None:
    base = metrics("b", [run("x", scores={"z": 1.0, "a": 1.0})])
    result = compare(base, base, **FAST)
    assert [m.metric for m in result] == [
        PASS_RATE,
        "score:a",
        "score:z",
        COST,
        LATENCY,
        STEPS,
    ]
    assert metric_order("other") > metric_order(STEPS)


def test_insufficient_data_and_min_cases() -> None:
    base = experiment("b", [1])
    cand = experiment("c", [0])
    assert compare(base, cand, **FAST).metric(PASS_RATE).status is MetricStatus.INSUFFICIENT_DATA
    forced = compare(base, cand, min_cases=1, **FAST).metric(PASS_RATE)
    assert forced.status is MetricStatus.REGRESSED


def test_added_and_removed_cases_are_not_compared() -> None:
    base = metrics("b", [run("keep"), run("old", passed=False)])
    cand = metrics("c", [run("new", passed=False), run("keep")])
    result = compare(base, cand, **FAST)
    assert [c.case_id for c in result.cases] == ["keep"]
    assert result.added_cases == ["new"]
    assert result.removed_cases == ["old"]
    assert result.metric(PASS_RATE).delta == 0
    with pytest.raises(KeyError):
        result.case("new")
    with pytest.raises(KeyError):
        result.metric("score:missing")


def test_metric_missing_on_one_side_is_paired_only_where_present() -> None:
    base = metrics("b", [run("x", cost=None), run("y", cost=0.01), run("z", cost=0.01)])
    cand = metrics("c", [run("x", cost=0.5), run("y", cost=0.02), run("z", cost=None)])
    cost = compare(base, cand, **FAST).metric(COST)
    assert cost.n == 1
    assert cost.delta == pytest.approx(0.01)


def test_no_paired_cases() -> None:
    result = compare(metrics("b", [run("a")]), metrics("c", [run("b")]), **FAST)
    assert result.metrics == []
    assert result.cases == []


def test_delta_pct_none_for_zero_baseline() -> None:
    result = compare(experiment("b", [0, 0]), experiment("c", [1, 1]), **FAST)
    assert result.metric(PASS_RATE).delta_pct is None


def test_result_records_settings_and_experiments() -> None:
    base, cand = experiment("b", [1, 1]), experiment("c", [1, 0])
    result = compare(base, cand, resamples=50, confidence=0.9, seed=7)
    assert (result.resamples, result.confidence, result.seed) == (50, 0.9, 7)
    assert result.baseline is base.experiment
    assert result.candidate is cand.experiment
    assert result.metric(PASS_RATE).ci.confidence == 0.9


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"min_cases": 0}, "min_cases"),
        ({"resamples": 0}, "resamples"),
        ({"confidence": 1.5}, "confidence"),
    ],
)
def test_invalid_arguments(kwargs: dict[str, Any], match: str) -> None:
    base = experiment("b", [1])
    with pytest.raises(ValueError, match=match):
        compare(base, base, **kwargs)


# -- properties --------------------------------------------------------------------

case_st = st.fixed_dictionaries(
    {
        "passed": st.booleans(),
        "score": st.floats(min_value=0, max_value=1),
        "cost": st.one_of(st.none(), st.floats(min_value=0, max_value=1)),
        "latency": st.floats(min_value=0, max_value=1e4),
        "steps": st.integers(min_value=0, max_value=20),
    }
)
pairs_st = st.lists(st.tuples(case_st, case_st), min_size=1, max_size=15)


def _exp(name: str, cases: list[dict[str, Any]]) -> ExperimentMetrics:
    return metrics(
        name,
        [
            RunMetrics(
                case_id=f"c{i}",
                repetition=0,
                passed=c["passed"],
                error=None,
                cost_usd=c["cost"],
                latency_ms=c["latency"],
                steps=c["steps"],
                scores={"s": c["score"]},
            )
            for i, c in enumerate(cases)
        ],
    )


@settings(max_examples=50, deadline=None)
@given(pairs_st)
def test_property_self_comparison_is_unchanged(pairs: list[tuple[dict[str, Any], ...]]) -> None:
    exp = _exp("e", [a for a, _ in pairs])
    result = compare(exp, exp, resamples=100)
    for m in result.metrics:
        assert m.delta == 0
        assert m.ci.low == m.ci.high == 0
        assert m.status in (MetricStatus.UNCHANGED, MetricStatus.INSUFFICIENT_DATA)
    assert all(c.status is CaseStatus.UNCHANGED for c in result.cases)


@settings(max_examples=50, deadline=None)
@given(pairs_st)
def test_property_swapping_sides_mirrors_the_result(
    pairs: list[tuple[dict[str, Any], ...]],
) -> None:
    a = _exp("a", [x for x, _ in pairs])
    b = _exp("b", [y for _, y in pairs])
    forward = compare(a, b, resamples=200)
    backward = compare(b, a, resamples=200)
    mirror = {
        MetricStatus.REGRESSED: MetricStatus.IMPROVED,
        MetricStatus.IMPROVED: MetricStatus.REGRESSED,
        MetricStatus.UNCHANGED: MetricStatus.UNCHANGED,
        MetricStatus.INSUFFICIENT_DATA: MetricStatus.INSUFFICIENT_DATA,
    }
    assert [m.metric for m in forward] == [m.metric for m in backward]
    for f, r in zip(forward, backward, strict=True):
        assert f.n == r.n
        assert r.delta == pytest.approx(-f.delta, abs=1e-9)
        assert r.ci.low == pytest.approx(-f.ci.high, abs=1e-6)
        assert r.ci.high == pytest.approx(-f.ci.low, abs=1e-6)
        assert r.status is mirror[f.status] or abs(f.ci.low) < 1e-6 or abs(f.ci.high) < 1e-6


@settings(max_examples=30, deadline=None)
@given(pairs_st, st.integers(min_value=0, max_value=1000))
def test_property_deterministic_and_delta_inside_bounds(
    pairs: list[tuple[dict[str, Any], ...]], seed: int
) -> None:
    a = _exp("a", [x for x, _ in pairs])
    b = _exp("b", [y for _, y in pairs])
    first = compare(a, b, resamples=200, seed=seed)
    second = compare(a, b, resamples=200, seed=seed)
    assert first.metrics == second.metrics
    for m in first.metrics:
        diffs = [c.deltas[m.metric] for c in first.cases if m.metric in c.deltas]
        assert min(diffs) <= m.ci.low <= m.ci.high <= max(diffs)
        assert m.delta == pytest.approx(m.candidate - m.baseline, abs=1e-9)


# -- end to end with the suite runner and the store ----------------------------------


def make_agent(wrong: set[str]) -> Any:
    def agent(question: str) -> str:
        llm = agentlens.wrap_llm(get_client("mock"))
        answer = llm.complete(LLMRequest(messages=[Message.user(question)])).content or ""
        return "nope" if question in wrong else answer

    return agent


def make_suite(agent: Any) -> Suite:
    return Suite.model_validate(
        {
            "name": "echo",
            "agent": agent,
            "repetitions": 2,
            "cases": [{"id": f"q{i}", "input": f"q{i}", "expected": f"q{i}"} for i in range(25)],
            "defaults": {"evaluators": ["no_error", "exact_match"]},
        }
    )


def test_end_to_end_compare_stored_experiments() -> None:
    with SQLiteStore(":memory:") as store:
        good = run_suite(make_suite(make_agent(set())), store=store, name="main")
        bad = run_suite(
            make_suite(make_agent({f"q{i}" for i in range(10)})), store=store, name="pr-1"
        )
        result: Comparison = compare_experiments(store, "main", "pr-1", **FAST)
        assert result.baseline.id == good.experiment.id
        assert result.candidate.id == bad.experiment.id
        assert result.metric(PASS_RATE).delta == pytest.approx(-0.4)
        assert result.metric(PASS_RATE).regressed
        assert result.metric(score_metric("exact_match")).regressed
        assert result.metric(STEPS).status is MetricStatus.UNCHANGED
        assert sorted(c.case_id for c in result.regressed_cases) == sorted(
            f"q{i}" for i in range(10)
        )
        assert result.case("q0").baseline.runs == 2

        # Same numbers without a store round-trip.
        direct = compare(metrics_from_run(good), metrics_from_run(bad), **FAST)
        assert [(m.metric, m.delta, m.ci) for m in direct if m.metric != LATENCY] == [
            (m.metric, m.delta, m.ci) for m in result if m.metric != LATENCY
        ]

        with pytest.raises(ExperimentNotFoundError):
            compare_experiments(store, "main", "missing")
