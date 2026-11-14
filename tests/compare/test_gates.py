from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
from pydantic import ValidationError

from agentlens.compare import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    CaseComparison,
    CaseMetric,
    Comparison,
    Direction,
    GateReport,
    GateResult,
    GateRule,
    Gates,
    GatesError,
    MetricDelta,
    Status,
    compare_runs,
    evaluate_gates,
    load_gates,
    resolve_metric,
    run_metrics,
)
from agentlens.schema import Case, Experiment
from agentlens.suite import CaseResult, ExperimentRun, Suite, SuiteError
from tests.compare.helpers import cases, run

DESIGN_EXAMPLE = """
success_rate: {min: 0.9}
cost_usd: {max_delta_pct: 20}
regressions: {max: 0}
"""


def write(tmp_path: Path, text: str, name: str = "gates.yaml") -> Path:
    path = tmp_path / name
    path.write_text(textwrap.dedent(text).lstrip("\n"), encoding="utf-8")
    return path


def problems(path: Path) -> list[str]:
    """Issues of a broken gates file as ``line:col: path: message`` (file name stripped)."""
    with pytest.raises(GatesError) as info:
        load_gates(path)
    return [str(issue).removeprefix(f"{path}:") for issue in info.value.issues]


def delta(
    name: str,
    *,
    baseline: float | None = 1.0,
    candidate: float | None = 1.0,
    status: Status = Status.UNCHANGED,
    n: int = 30,
    delta_pct: float | None = None,
) -> MetricDelta:
    diff = candidate - baseline if candidate is not None and baseline is not None else None
    if delta_pct is None and diff is not None and baseline:
        delta_pct = diff / abs(baseline) * 100
    return MetricDelta(
        name=name,
        direction=Direction.LOWER_IS_BETTER
        if name in {"cost_usd", "latency_ms", "steps"}
        else Direction.HIGHER_IS_BETTER,
        n=n,
        baseline=baseline,
        candidate=candidate,
        delta=diff,
        delta_pct=delta_pct,
        ci_low=diff,
        ci_high=diff,
        status=status,
    )


def comparison(*metrics: MetricDelta, n_cases: int = 30, **kwargs: object) -> Comparison:
    case_list = [
        CaseComparison(
            case_id=f"c{i}",
            baseline_runs=1,
            candidate_runs=1,
            metrics={"pass_rate": CaseMetric(baseline=1.0, candidate=1.0, delta=0.0)},
            status=Status.UNCHANGED,
        )
        for i in range(n_cases)
    ]
    return Comparison(
        metrics=list(metrics), cases=case_list, resamples=100, confidence=0.95, **kwargs
    )


def gates(**rules: dict[str, object]) -> Gates:
    return Gates.from_dict(rules)


# -- metric names ------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("name", "metric"),
    [
        ("success_rate", "pass_rate"),
        ("pass_rate", "pass_rate"),
        ("cost", "cost_usd"),
        ("cost_usd", "cost_usd"),
        ("latency", "latency_ms"),
        ("steps", "steps"),
        ("score.exact_match", "score.exact_match"),
        ("regressions", "regressions"),
        ("case_regressions", "case_regressions"),
    ],
)
def test_resolve_metric(name: str, metric: str) -> None:
    assert resolve_metric(name) == metric


def test_resolve_metric_unknown_suggests() -> None:
    with pytest.raises(ValueError, match=r"unknown metric 'sucess_rate' \(did you mean 'success"):
        resolve_metric("sucess_rate")
    with pytest.raises(ValueError, match=r"score\.<evaluator>"):
        resolve_metric("score.")


# -- models ------------------------------------------------------------------------------


def test_rule_bounds_in_order() -> None:
    rule = GateRule(max_delta_pct=20, min=0.5)
    assert rule.bounds() == [("min", 0.5), ("max_delta_pct", 20.0)]


@pytest.mark.parametrize(
    ("fields", "message"),
    [
        ({}, "at least one of min, max"),
        ({"min": 1, "max": 0}, r"min \(1\) is greater than max \(0\)"),
        ({"min_delta": 1, "max_delta": 0}, "min_delta"),
        ({"min_delta_pct": 5, "max_delta_pct": -5}, "min_delta_pct"),
        ({"min": True}, "not a boolean"),
        ({"max": float("inf")}, "finite"),
        ({"max": "a lot"}, "valid number"),
        ({"min": 0.1, "on_missing": "ignore"}, "'fail' or 'skip'"),
    ],
)
def test_rule_validation(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        GateRule.model_validate(fields)


def test_gates_model_validates_names() -> None:
    with pytest.raises(ValidationError, match="unknown metric"):
        Gates(rules={"bogus": GateRule(min=1)})
    with pytest.raises(ValidationError, match="same metric as 'success_rate'"):
        Gates(rules={"success_rate": GateRule(min=1), "pass_rate": GateRule(min=1)})
    with pytest.raises(ValidationError, match="only min/max apply"):
        Gates(rules={"regressions": GateRule(max_delta=1)})
    with pytest.raises(ValidationError):
        Gates(rules={})


def test_from_dict_design_example() -> None:
    loaded = Gates.from_dict(
        {
            "success_rate": {"min": 0.9},
            "cost_usd": {"max_delta_pct": 20},
            "regressions": {"max": 0},
            "options": {"min_cases": 5, "on_missing": "skip"},
        }
    )
    assert list(loaded.rules) == ["success_rate", "cost_usd", "regressions"]
    assert loaded.rules["cost_usd"].max_delta_pct == 20.0
    assert loaded.min_cases == 5
    assert loaded.on_missing == "skip"


def test_from_dict_collects_every_problem() -> None:
    with pytest.raises(GatesError) as info:
        Gates.from_dict({"sucess_rate": {"min": 1}, "cost": 3, "steps": {"mx": 1}})
    issues = [str(issue) for issue in info.value.issues]
    assert issues == [
        "sucess_rate: unknown metric 'sucess_rate' (did you mean 'success_rate'?); known: "
        "case_regressions, cost, cost_usd, latency, latency_ms, pass_rate, regressions, "
        "steps, success_rate, score.<evaluator>",
        "cost: a gate is a mapping of rule to threshold, e.g. cost: {min: 0.9}",
        "steps.mx: unknown rule 'mx' (did you mean 'max'?)",
    ]
    assert isinstance(info.value, SuiteError)
    assert str(info.value).startswith("invalid gates (3 problems):")


@pytest.mark.parametrize("data", [None, [1, 2], "success_rate"])
def test_from_dict_not_a_mapping(data: object) -> None:
    with pytest.raises(GatesError, match="must be a mapping of metric to rule"):
        Gates.from_dict(data)


def test_from_dict_empty() -> None:
    with pytest.raises(GatesError, match="no gates defined"):
        Gates.from_dict({"options": {"min_cases": 3}})


# -- loading -----------------------------------------------------------------------------


def test_load_design_example(tmp_path: Path) -> None:
    loaded = load_gates(write(tmp_path, DESIGN_EXAMPLE))
    assert loaded == Gates(
        rules={
            "success_rate": GateRule(min=0.9),
            "cost_usd": GateRule(max_delta_pct=20),
            "regressions": GateRule(max=0),
        }
    )
    assert loaded.min_cases == 20
    assert loaded.on_missing == "fail"


def test_load_errors_have_positions(tmp_path: Path) -> None:
    path = write(
        tmp_path,
        """
        success_rate: {min: 0.9, max: 0.5}
        sucess: {min: 1}
        cost_usd:
          max_delta_pct: twenty
        steps:
          mx: 3
        regressions: {max_delta: 0}
        case_regressions: 2
        pass_rate: {min: 1}
        options: {min_case: 10, on_missing: maybe}
        """,
    )
    assert problems(path) == [
        "1:15: success_rate: min (0.9) is greater than max (0.5)",
        "2:1: sucess: unknown metric 'sucess' (did you mean 'success_rate'?); known: "
        "case_regressions, cost, cost_usd, latency, latency_ms, pass_rate, regressions, "
        "steps, success_rate, score.<evaluator>",
        "4:18: cost_usd.max_delta_pct: Input should be a valid number, unable to parse "
        "string as a number",
        "6:3: steps.mx: unknown rule 'mx' (did you mean 'max'?)",
        "7:15: regressions.max_delta: 'regressions' is a count of the comparison: only "
        "min/max apply, not 'max_delta'",
        "8:19: case_regressions: a gate is a mapping of rule to threshold, e.g. "
        "case_regressions: {min: 0.9}",
        "9:1: pass_rate: 'pass_rate' is the same metric as 'success_rate'",
        "10:11: options.min_case: unknown option 'min_case' (did you mean 'min_cases'?)",
        "10:37: options.on_missing: Input should be 'fail' or 'skip'",
    ]


def test_load_options_not_a_mapping(tmp_path: Path) -> None:
    path = write(tmp_path, "success_rate: {min: 1}\noptions: 3\n")
    assert problems(path) == ["2:10: options: options must be a mapping, e.g. {min_cases: 20}"]


def test_load_root_not_a_mapping(tmp_path: Path) -> None:
    path = write(tmp_path, "- success_rate\n")
    assert problems(path) == [
        "1:1: a gates file must be a mapping of metric to rule, e.g. 'success_rate: {min: 0.9}'"
    ]


def test_load_invalid_yaml(tmp_path: Path) -> None:
    path = write(tmp_path, "success_rate: {min: 0.9\n")
    with pytest.raises(GatesError, match="invalid YAML") as info:
        load_gates(path)
    assert info.value.issues[0].line == 2


def test_load_duplicate_key(tmp_path: Path) -> None:
    path = write(tmp_path, "success_rate: {min: 0.9}\nsuccess_rate: {min: 0.8}\n")
    assert problems(path) == ["2:1: success_rate: duplicate key 'success_rate'"]


def test_load_missing_and_empty_file(tmp_path: Path) -> None:
    with pytest.raises(GatesError, match="gates file not found"):
        load_gates(tmp_path / "nope.yaml")
    with pytest.raises(GatesError, match="gates file is empty"):
        load_gates(write(tmp_path, "# nothing\n"))


# -- evaluation against a comparison ------------------------------------------------------


def test_design_example_passes() -> None:
    loaded = Gates.from_dict(
        {"success_rate": {"min": 0.9}, "cost_usd": {"max_delta_pct": 20}, "regressions": {"max": 0}}
    )
    report = evaluate_gates(
        loaded,
        comparison=comparison(
            delta("pass_rate", baseline=0.9, candidate=0.95),
            delta("cost_usd", baseline=1.0, candidate=1.1),
        ),
    )
    assert report.passed
    assert report.exit_code == EXIT_OK
    assert report.failures == []
    assert report.notes == []
    assert [(r.name, r.metric, r.rule, r.actual) for r in report.results] == [
        ("success_rate", "pass_rate", "min", 0.95),
        ("cost_usd", "cost_usd", "max_delta_pct", pytest.approx(10.0)),
        ("regressions", "regressions", "max", 0.0),
    ]
    assert report.results[0].reason == "pass_rate 0.95 is within min 0.9"
    assert report.results[1].reason == "cost_usd change +10% is within max_delta_pct +20%"


def test_design_example_fails() -> None:
    loaded = Gates.from_dict(
        {"success_rate": {"min": 0.9}, "cost_usd": {"max_delta_pct": 20}, "regressions": {"max": 0}}
    )
    report = loaded.evaluate(
        comparison(
            delta("pass_rate", baseline=0.95, candidate=0.8, status=Status.REGRESSION),
            delta("cost_usd", baseline=1.0, candidate=1.25),
        )
    )
    assert not report.passed
    assert report.exit_code == EXIT_GATE_FAILED
    assert [r.reason for r in report.failures] == [
        "pass_rate 0.8 is below min 0.9",
        "cost_usd change +25% is above max_delta_pct +20%",
        "regressions 1 is above max 0",
    ]
    assert str(report.failures[0]) == "FAIL success_rate min 0.9: pass_rate 0.8 is below min 0.9"
    assert report.summary().splitlines()[0] == "gates FAILED: 0 passed, 3 failed, 0 skipped"


def test_bounds_are_inclusive_and_float_tolerant() -> None:
    report = evaluate_gates(
        gates(pass_rate={"min": 0.9, "max": 0.9}),
        comparison=comparison(delta("pass_rate", candidate=0.1 * 9)),  # 0.9000000000000001
    )
    assert report.passed
    assert report.results[0].actual == pytest.approx(0.9)


def test_delta_rules() -> None:
    report = evaluate_gates(
        gates(
            **{
                "score.judge": {"min_delta": -0.05},
                "latency": {"max_delta": 100},
                "steps": {"min_delta_pct": -10},
            }
        ),
        comparison=comparison(
            delta("score.judge", baseline=0.8, candidate=0.7),
            delta("latency_ms", baseline=1000, candidate=1050),
            delta("steps", baseline=4, candidate=4),
        ),
    )
    assert [(r.metric, r.passed) for r in report.results] == [
        ("score.judge", False),
        ("latency_ms", True),
        ("steps", True),
    ]
    assert report.results[0].reason == "score.judge change -0.1 is below min_delta -0.05"


def test_case_regressions_gate() -> None:
    comp = comparison(delta("pass_rate"))
    comp.cases[0].status = Status.REGRESSION
    comp.cases[1].status = Status.REGRESSION
    report = evaluate_gates(gates(case_regressions={"max": 1}), comparison=comp)
    assert report.results[0].actual == 2.0
    assert not report.passed


def test_missing_metric_fails_by_default() -> None:
    report = evaluate_gates(
        gates(**{"score.judge": {"min": 0.5, "min_delta": 0}}),
        comparison=comparison(delta("pass_rate"), delta("score.exact_match")),
    )
    assert [(r.passed, r.skipped, r.actual) for r in report.results] == [
        (False, False, None),
        (False, False, None),
    ]
    assert [r.reason for r in report.results] == [
        "no data for 'score.judge' (scores available: score.exact_match)",
        "no metric 'score.judge' in the comparison (scores available: score.exact_match)",
    ]


def test_no_data_metric() -> None:
    no_cost = MetricDelta(
        name="cost_usd", direction=Direction.LOWER_IS_BETTER, n=0, status=Status.NO_DATA
    )
    report = evaluate_gates(
        gates(cost={"max": 1, "max_delta_pct": 10}), comparison=comparison(no_cost)
    )
    assert [r.reason for r in report.results] == [
        "no data for 'cost_usd'",
        "no case has a 'cost_usd' value in both experiments",
    ]
    assert not report.passed


def test_on_missing_skip_globally_and_per_rule() -> None:
    comp = comparison(delta("pass_rate"))
    skip_all = Gates.from_dict({"cost": {"max": 1}, "options": {"on_missing": "skip"}})
    report = evaluate_gates(skip_all, comparison=comp)
    assert report.passed
    assert report.skipped == report.results
    assert str(report.results[0]) == "SKIP cost max 1: no data for 'cost_usd'"

    per_rule = Gates.from_dict({"cost": {"max": 1, "on_missing": "skip"}, "latency": {"max": 1}})
    report = evaluate_gates(per_rule, comparison=comp)
    assert [(r.skipped, r.passed) for r in report.results] == [(True, True), (False, False)]
    assert report.summary().splitlines()[0] == "gates FAILED: 0 passed, 1 failed, 1 skipped"


def test_percent_change_with_zero_baseline() -> None:
    gate = gates(cost={"max_delta_pct": 20})
    same = evaluate_gates(gate, comparison=comparison(delta("cost_usd", baseline=0, candidate=0)))
    assert same.passed
    assert same.results[0].actual == 0.0
    grew = evaluate_gates(gate, comparison=comparison(delta("cost_usd", baseline=0, candidate=0.5)))
    assert not grew.passed
    assert grew.results[0].reason == "baseline 'cost_usd' is 0: percentage change undefined"


def test_notes_for_few_and_unpaired_cases() -> None:
    comp = comparison(
        delta("pass_rate"),
        n_cases=3,
        added_cases=["new1", "new2"],
        removed_cases=[f"old{i}" for i in range(7)],
    )
    report = evaluate_gates(gates(pass_rate={"min": 0.5}), comparison=comp)
    assert report.passed  # notes never fail
    assert report.notes == [
        "only 3 cases compared (present in both experiments): at least 20 are recommended "
        "for meaningful results (ADR 0005); not a gate failure",
        "2 cases only in the candidate, not part of any delta: new1, new2",
        "7 cases only in the baseline, not part of any delta: old0, old1, old2, old3, old4, ...",
    ]
    assert report.summary().splitlines()[-1].startswith("  note: 7 cases only in the baseline")


def test_min_cases_option() -> None:
    comp = comparison(delta("pass_rate"), n_cases=1)
    report = evaluate_gates(
        Gates.from_dict({"pass_rate": {"min": 0}, "options": {"min_cases": 0}}), comparison=comp
    )
    assert report.notes == []
    report = evaluate_gates(
        Gates.from_dict({"pass_rate": {"min": 0}, "options": {"min_cases": 2}}), comparison=comp
    )
    assert report.notes[0].startswith("only 1 case compared")


# -- evaluation without a baseline ---------------------------------------------------------


def test_run_only_gates_absolute_values_and_skips_deltas() -> None:
    runs = [*cases(18, cost=0.01), run("bad", passed=False, cost=0.03)]
    loaded = Gates.from_dict(
        {
            "success_rate": {"min": 0.9},
            "cost": {"max": 0.02, "max_delta_pct": 20},
            "regressions": {"max": 0},
        }
    )
    report = evaluate_gates(loaded, run=runs)
    assert report.passed
    assert [(r.metric, r.rule, r.skipped) for r in report.results] == [
        ("pass_rate", "min", False),
        ("cost_usd", "max", False),
        ("cost_usd", "max_delta_pct", True),
        ("regressions", "max", True),
    ]
    assert report.results[0].actual == pytest.approx(18 / 19)
    assert report.results[3].reason == "needs a baseline comparison"
    assert report.notes == [
        "only 19 cases evaluated: at least 20 are recommended for meaningful results "
        "(ADR 0005); not a gate failure",
        "no baseline comparison: 2 delta/regression checks skipped",
    ]


def test_run_only_with_skip_note_singular_and_experiment_run() -> None:
    runs = cases(20)
    experiment_run = ExperimentRun(
        experiment=Experiment(name="x", suite="s"),
        suite=Suite(name="s", agent=str, cases=[Case(id="c0", input="hi")]),
        cases=[CaseResult(case=Case(id=r.case_id, input="hi"), runs=[r]) for r in runs],
    )
    report = evaluate_gates(gates(steps={"max": 1, "max_delta": 0}), run=experiment_run)
    assert report.passed
    assert report.notes == ["no baseline comparison: 1 delta/regression check skipped"]


def test_run_takes_precedence_for_absolute_values() -> None:
    base = cases(20)
    cand = [*cases(20), run("extra", passed=False)]  # a new failing case, not paired
    comp = compare_runs(base, cand, resamples=50)
    loaded = gates(pass_rate={"min": 1})
    assert evaluate_gates(loaded, comparison=comp).passed  # paired cases only
    report = evaluate_gates(loaded, comparison=comp, run=cand)  # every candidate case
    assert report.results[0].actual == pytest.approx(20 / 21)
    assert not report.passed


def test_needs_comparison_or_run() -> None:
    with pytest.raises(ValueError, match="needs a comparison, a run, or both"):
        evaluate_gates(gates(pass_rate={"min": 1}))


def test_report_round_trips_through_json() -> None:
    report = evaluate_gates(
        gates(pass_rate={"min": 1}), comparison=comparison(delta("pass_rate", candidate=0.5))
    )
    again = GateReport.model_validate_json(report.model_dump_json())
    assert again == report
    assert again.exit_code == 1
    assert isinstance(again.results[0], GateResult)


# -- run_metrics ---------------------------------------------------------------------------


def test_run_metrics() -> None:
    runs = [
        run("a", 0, passed=True, score=1.0, cost=0.02, latency_ms=100, steps=2),
        run("a", 1, passed=False, score=0.0, cost=None, latency_ms=300, steps=2),
        run("b", 0, passed=True, score=1.0, cost=None, latency_ms=200, steps=4),
    ]
    metrics = run_metrics(runs)
    assert list(metrics) == ["pass_rate", "score.check", "cost_usd", "latency_ms", "steps"]
    assert metrics["pass_rate"] == (pytest.approx(0.75), 2)
    assert metrics["score.check"] == (pytest.approx(0.75), 2)
    assert metrics["cost_usd"] == (pytest.approx(0.02), 1)  # b has no known cost
    assert metrics["latency_ms"] == (pytest.approx(200.0), 2)
    assert metrics["steps"] == (pytest.approx(3.0), 2)
    assert run_metrics([]) == {}
