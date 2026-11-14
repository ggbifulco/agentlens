from __future__ import annotations

import importlib
import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from agentlens.compare import Gates, compare_runs, evaluate_gates
from agentlens.report import (
    ComparisonReport,
    ExperimentReport,
    PlainTextRenderer,
    ReportFormat,
    ReportFormatUnavailableError,
    available_formats,
    fmt_value,
    get_renderer,
    load_report,
    register_renderer,
    render,
    table,
)
from agentlens.schema import Experiment
from tests.compare.helpers import run

render_module = importlib.import_module("agentlens.report.render")

CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)


def _experiment(**config: object) -> Experiment:
    return Experiment(
        id="e1", name="main", suite="toy", created_at=CREATED, config=config, is_baseline=True
    )


@pytest.mark.parametrize(
    ("value", "signed", "expected"),
    [(None, False, "-"), (0.123456, False, "0.1235"), (2.0, True, "+2"), (-0.5, True, "-0.5")],
)
def test_fmt_value(value: float | None, signed: bool, expected: str) -> None:
    assert fmt_value(value, signed=signed) == expected


def test_table_aligns_columns_without_trailing_spaces() -> None:
    assert table(["a", "bb"], [["xxx", ""], ["y", "z"]], indent="  ").splitlines() == [
        "  a    bb",
        "  xxx",
        "  y    z",
    ]


def test_experiment_text_has_header_metrics_cases_and_gates() -> None:
    runs = [
        run("a", passed=True, cost=0.25),
        run("b", passed={"exact": False}, error="ValueError: boom"),
        run("b", 1, passed={"exact": False}, error="ValueError: boom"),
    ]
    gates = evaluate_gates(Gates.from_dict({"pass_rate": {"min": 0.9}}), run=runs)
    experiment = _experiment(model="mock", git_sha="0123456789abcdef", git_dirty=True)
    text = render(ExperimentReport.from_runs(experiment, runs, gates=gates))

    assert "experiment  main (e1) [baseline]" in text
    assert "created     2026-01-02 03:04:05 UTC" in text
    assert "model       mock" in text
    assert "git         0123456789ab (dirty)" in text
    assert "runs        1/3 passed (33%), 2 errors" in text
    assert "  pass_rate    0.5" in text
    assert "  b     0/2     0%    2       exact" in text
    assert "gates FAILED" in text


def test_experiment_text_minimal() -> None:
    experiment = Experiment(name="x", suite="toy", created_at=CREATED)
    text = render(ExperimentReport.from_runs(experiment, [run("a", error="E: x")]))
    assert "runs        0/1 passed (0%), 1 error\n" in text
    assert "model" not in text
    assert "git" not in text
    assert "[baseline]" not in text

    empty = render(ExperimentReport.from_runs(experiment, []))
    assert "metrics" not in empty
    assert "cases" not in empty


def test_comparison_text() -> None:
    base = [run("a"), run("b", passed={"exact": True}), run("c", passed=False), run("gone")]
    cand = [run("a"), run("b", passed={"exact": False}), run("c"), run("new")]
    comparison = compare_runs(base, cand, resamples=100)
    gates = evaluate_gates(Gates.from_dict({"regressions": {"max": 0}}), comparison=comparison)
    text = render(ComparisonReport(comparison=comparison, gates=gates))

    assert "baseline   (in memory)" in text
    assert "cases      3 paired, 1 added, 1 removed; paired bootstrap, 100 resamples" in text
    assert "95% CI, seed 0" in text
    assert "case regressions (1)" in text
    assert "  b     1 -> 0     exact" in text
    assert "case improvements (1)" in text
    assert "added cases (1): new" in text
    assert "removed cases (1): gone" in text
    assert "cost_usd" in text  # no costs recorded: no CI, shown as "-"
    cost_line = next(line for line in text.splitlines() if line.startswith("cost_usd"))
    assert cost_line.split() == ["cost_usd", "0", "-", "-", "-", "-", "no_data"]
    assert "\ngates " in text


def test_comparison_text_marks_regressions_and_experiments() -> None:
    base = [run(f"c{i}") for i in range(5)]
    cand = [run(f"c{i}", passed=False) for i in range(5)]
    comparison = compare_runs(
        base,
        cand,
        resamples=100,
        baseline_experiment=_experiment(),
        candidate_experiment=Experiment(id="e2", name="pr", suite="toy", created_at=CREATED),
    )
    text = render(ComparisonReport(comparison=comparison))
    assert "baseline   main (e1) [baseline]" in text
    assert "candidate  pr (e2)" in text
    pass_line = next(line for line in text.splitlines() if line.startswith("pass_rate"))
    assert pass_line.endswith("REGRESSION")
    assert "gates" not in text


def test_json_format_round_trips() -> None:
    report = ExperimentReport.from_runs(_experiment(), [run("a")])
    data = json.loads(render(report, "json"))
    assert data["experiment"]["name"] == "main"
    cmp = ComparisonReport(comparison=compare_runs([run("a")], [run("a")], resamples=10))
    assert ComparisonReport.model_validate_json(render(cmp, ReportFormat.JSON)) == cmp


def test_planned_formats_are_not_available_yet() -> None:
    assert available_formats() == [ReportFormat.TEXT, ReportFormat.JSON]
    with pytest.raises(ReportFormatUnavailableError, match=r"Markdown report \(#27\)"):
        get_renderer("markdown")
    with pytest.raises(ValueError, match="pdf"):
        get_renderer("pdf")


def test_register_renderer_plugs_in_a_format(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "_RENDERERS", dict(render_module._RENDERERS))

    class Upper(PlainTextRenderer):
        def experiment(self, report: ExperimentReport) -> str:
            return super().experiment(report).upper()

    register_renderer("html", Upper())
    assert ReportFormat.HTML in available_formats()
    report = ExperimentReport.from_runs(_experiment(), [run("a")])
    assert render(report, "html").startswith("EXPERIMENT")


def test_unavailable_hint_only_for_planned_formats(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(render_module, "_RENDERERS", {})
    with pytest.raises(ReportFormatUnavailableError) as info:
        get_renderer("text")
    assert "arrives" not in str(info.value)


def test_load_report_reads_rendered_json(tmp_path: Path) -> None:
    path = tmp_path / "r.json"
    report = ExperimentReport.from_runs(_experiment(), [run("a")])
    path.write_text(render(report, "json"))
    assert load_report(path) == report
