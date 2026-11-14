from __future__ import annotations

import json
from pathlib import Path

import pytest

from agentlens.compare import Gates, compare_runs, evaluate_gates
from agentlens.report import (
    CaseSummary,
    ComparisonReport,
    ExperimentReport,
    ReportError,
    load_report,
)
from agentlens.schema import Case, Experiment
from agentlens.suite import CaseResult, ExperimentRun, Suite
from tests.compare.helpers import run

EXP = Experiment(name="main", suite="toy")


def test_from_runs_summarises_cases_in_first_appearance_order() -> None:
    runs = [
        run("b", 0, passed={"exact": False, "no_error": True}, cost=0.5),
        run("a", 0, passed=True),
        run("b", 1, passed=True, error="ValueError: boom"),
    ]
    report = ExperimentReport.from_runs(EXP, runs)

    assert (report.runs, report.passed, report.errors) == (3, 1, 1)
    assert report.pass_rate == pytest.approx(1 / 3)
    assert [c.case_id for c in report.cases] == ["b", "a"]
    b = report.cases[0]
    assert b == CaseSummary(
        case_id="b",
        runs=2,
        passed=0,
        errors=1,
        pass_rate=0.0,
        scores={"exact": 0.0, "no_error": 1.0, "check": 1.0},
        failed_evaluators=["exact"],
    )
    assert report.metrics["pass_rate"] == pytest.approx(0.5)  # mean over cases, not runs
    assert report.metrics["cost_usd"] == pytest.approx(0.5)
    assert report.gates is None


def test_from_runs_accepts_an_experiment_run_and_gates() -> None:
    suite = Suite(name="toy", agent="mod:fn", cases=[{"id": "a", "input": "x"}])
    exp_run = ExperimentRun(
        experiment=EXP,
        suite=suite,
        cases=[CaseResult(case=Case(id="a", input="x"), runs=[run("a")])],
    )
    gates = evaluate_gates(Gates.from_dict({"pass_rate": {"min": 0.5}}), run=exp_run)
    report = ExperimentReport.from_runs(EXP, exp_run, gates=gates)
    assert report.runs == 1
    assert report.gates is not None
    assert report.gates.passed


def test_from_runs_with_no_runs() -> None:
    report = ExperimentReport.from_runs(EXP, [])
    assert (report.runs, report.pass_rate, report.metrics, report.cases) == (0, 0.0, {}, [])


def _comparison_report() -> ComparisonReport:
    comparison = compare_runs(
        [run("a"), run("b")], [run("a"), run("b", passed=False)], resamples=50
    )
    return ComparisonReport(comparison=comparison)


def test_load_report_round_trips_both_kinds(tmp_path: Path) -> None:
    cmp_report = _comparison_report()
    exp_report = ExperimentReport.from_runs(EXP, [run("a")])
    (tmp_path / "cmp.json").write_text(cmp_report.model_dump_json())
    (tmp_path / "exp.json").write_text(exp_report.model_dump_json())

    assert load_report(tmp_path / "cmp.json") == cmp_report
    assert load_report(tmp_path / "exp.json") == exp_report


def test_load_report_accepts_a_bare_comparison(tmp_path: Path) -> None:
    comparison = _comparison_report().comparison
    path = tmp_path / "bare.json"
    path.write_text(comparison.model_dump_json())
    assert load_report(path) == ComparisonReport(comparison=comparison)


@pytest.mark.parametrize(
    ("content", "message"),
    [
        ("not json", "cannot read report"),
        ("[1, 2]", "expected a JSON object"),
        (json.dumps({"comparison": {"resamples": 0}}), "comparison.resamples"),
        (json.dumps({"metrics": {}, "experiment": {}}), "is not an AgentLens report"),
    ],
)
def test_load_report_rejects_bad_files(tmp_path: Path, content: str, message: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text(content)
    with pytest.raises(ReportError, match=message):
        load_report(path)


def test_load_report_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ReportError, match="cannot read report"):
        load_report(tmp_path / "missing.json")


def test_load_report_validation_error_without_location(tmp_path: Path) -> None:
    path = tmp_path / "bare.json"
    path.write_text(json.dumps({"resamples": 10, "confidence": 0.95, "unknown": 1}))
    with pytest.raises(ReportError, match="unknown"):
        load_report(path)
