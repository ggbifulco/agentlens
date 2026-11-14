"""The rich terminal report: snapshot tests plus the colour/arrow rules.

Snapshots live in ``tests/report/snapshots/``. After an intended change of the
layout, regenerate them and review the diff::

    AGENTLENS_UPDATE_SNAPSHOTS=1 uv run pytest tests/report/test_terminal.py
"""

from __future__ import annotations

import io
import os
from datetime import UTC, datetime
from pathlib import Path

import pytest
from rich.console import Console, RenderableType
from rich.text import Text

from agentlens.compare import (
    Comparison,
    GateReport,
    Gates,
    MetricDelta,
    compare_runs,
    evaluate_gates,
)
from agentlens.compare.model import Direction, Status
from agentlens.report import (
    ComparisonReport,
    ExperimentReport,
    ReportFormat,
    RichRenderer,
    comparison_renderable,
    experiment_renderable,
    get_renderer,
    render,
    terminal,
)
from agentlens.schema import Experiment
from agentlens.suite import RunResult
from tests.compare.helpers import run

SNAPSHOTS = Path(__file__).parent / "snapshots"
UPDATE = os.environ.get("AGENTLENS_UPDATE_SNAPSHOTS") == "1"
CREATED = datetime(2026, 1, 2, 3, 4, 5, tzinfo=UTC)
BASELINE = Experiment(
    id="exp-base",
    name="main",
    suite="support",
    created_at=CREATED,
    is_baseline=True,
    config={"model": "openai:gpt-4o-mini", "git_sha": "0123456789abcdef", "git_dirty": True},
)
CANDIDATE = Experiment(
    id="exp-cand",
    name="pr-42",
    suite="support",
    created_at=CREATED,
    config={"model": "openai:gpt-4o", "git_sha": "fedcba9876543210"},
)


def _baseline_runs() -> list[RunResult]:
    runs = [
        run(f"case-{i:02d}", passed={"exact": True, "tone": True}, cost=0.010, latency_ms=900)
        for i in range(8)
    ]
    return [*runs, run("retired", cost=0.01)]


def _candidate_runs() -> list[RunResult]:
    runs = [
        run(
            f"case-{i:02d}",
            passed={"exact": i >= 3, "tone": True},
            cost=0.013,
            latency_ms=700 + 10 * i,
        )
        for i in range(8)
    ]
    return [*runs, run("case-new", error="TimeoutError: agent call timed out")]


def _comparison_report() -> ComparisonReport:
    comparison = compare_runs(
        _baseline_runs(),
        _candidate_runs(),
        resamples=500,
        seed=0,
        baseline_experiment=BASELINE,
        candidate_experiment=CANDIDATE,
    )
    gates = evaluate_gates(
        Gates.from_dict(
            {
                "pass_rate": {"min": 0.9},
                "cost_usd": {"max_delta_pct": 20},
                "latency_ms": {"max_delta": 0},
                "regressions": {"max": 0},
            }
        ),
        comparison=comparison,
    )
    return ComparisonReport(comparison=comparison, gates=gates)


def _experiment_report() -> ExperimentReport:
    runs = _candidate_runs()
    gates = evaluate_gates(Gates.from_dict({"success_rate": {"min": 0.8}}), run=runs)
    return ExperimentReport.from_runs(CANDIDATE, runs, gates=gates)


def _export(renderable: RenderableType) -> str:
    console = Console(
        file=io.StringIO(), record=True, width=100, color_system=None, force_terminal=False
    )
    console.print(renderable)
    # Trailing spaces are padding; strip them so editors and pre-commit hooks leave files alone.
    return "\n".join(line.rstrip() for line in console.export_text().splitlines()) + "\n"


def _check_snapshot(name: str, text: str) -> None:
    path = SNAPSHOTS / name
    if UPDATE or not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
        if not UPDATE:
            pytest.fail(f"snapshot {path} did not exist and was written: review and commit it")
    assert text == path.read_text(encoding="utf-8"), (
        f"rendering differs from {path}; if intended, regenerate with "
        "AGENTLENS_UPDATE_SNAPSHOTS=1 and review the diff"
    )


def test_comparison_snapshot() -> None:
    _check_snapshot("comparison.txt", _export(comparison_renderable(_comparison_report())))


def test_experiment_snapshot() -> None:
    _check_snapshot("experiment.txt", _export(experiment_renderable(_experiment_report())))


def test_rich_is_a_registered_format() -> None:
    assert isinstance(get_renderer(ReportFormat.RICH), RichRenderer)
    assert ReportFormat("rich") is ReportFormat.RICH


def test_renderer_string_matches_the_console_export_without_colour() -> None:
    report = _comparison_report()
    text = RichRenderer(width=100, color=False).comparison(report)
    assert "\x1b[" not in text
    assert [line.rstrip() for line in text.splitlines()] == _export(
        comparison_renderable(report)
    ).splitlines()


def test_renderer_colours_regressions_red_and_improvements_green(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("COLORTERM", "truecolor")
    monkeypatch.setenv("TERM", "xterm-256color")
    text = RichRenderer(width=120).comparison(_comparison_report())
    regression = next(line for line in text.splitlines() if "REGRESSION" in line)
    improvement = next(line for line in text.splitlines() if "improvement" in line)
    assert "\x1b[1;31m" in regression  # bold red
    assert "\x1b[32m" in improvement  # green


def test_renderer_uses_the_terminal_width_by_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("COLUMNS", "60")
    text = RichRenderer(color=False).experiment(_experiment_report())
    assert max(len(line) for line in text.splitlines()) <= 60
    assert render(_experiment_report(), "rich")  # the registered instance works too


def _delta(status: Status, delta: float | None, direction: Direction) -> MetricDelta:
    return MetricDelta(
        name="m", direction=direction, n=3, delta=delta, delta_pct=None, status=status
    )


@pytest.mark.parametrize(
    ("delta", "direction", "status", "arrow", "style"),
    [
        (-0.2, Direction.HIGHER_IS_BETTER, Status.REGRESSION, "↓", "bold red"),
        (0.2, Direction.HIGHER_IS_BETTER, Status.IMPROVEMENT, "↑", "green"),
        (0.2, Direction.LOWER_IS_BETTER, Status.REGRESSION, "↑", "bold red"),
        (-0.2, Direction.LOWER_IS_BETTER, Status.IMPROVEMENT, "↓", "green"),
        (0.01, Direction.LOWER_IS_BETTER, Status.UNCHANGED, "↑", "dim"),
    ],
)
def test_delta_arrow_follows_the_value_and_colour_the_direction(
    delta: float, direction: Direction, status: Status, arrow: str, style: str
) -> None:
    cell = terminal._delta_cell(_delta(status, delta, direction))
    assert cell.plain.startswith(arrow)
    assert cell.style == style


def test_missing_values_are_dim_dashes() -> None:
    metric = _delta(Status.NO_DATA, None, Direction.LOWER_IS_BETTER)
    for cell in (terminal._delta_cell(metric), terminal._delta_pct_cell(metric)):
        assert cell == Text("-", style="dim")
    assert terminal._ci_cell(metric).plain == "-"
    assert terminal._delta_cell(_delta(Status.UNCHANGED, 0.0, Direction.HIGHER_IS_BETTER)).plain


def test_comparison_without_changes_gates_or_experiments() -> None:
    comparison = compare_runs([run("a"), run("b")], [run("a"), run("b")], resamples=10)
    text = _export(comparison_renderable(ComparisonReport(comparison=comparison)))
    assert "baseline   (in memory)" in text
    assert "✓ no metric regressed" in text
    assert "Case regressions" not in text
    assert "Gates" not in text
    assert "added cases" not in text


def test_comparison_lists_case_improvements_and_only_added_cases() -> None:
    comparison = compare_runs(
        [run("a", passed=False)], [run("a"), run("new")], resamples=10, seed=1
    )
    text = _export(comparison_renderable(ComparisonReport(comparison=comparison)))
    assert "Case improvements (1)" in text
    assert "added cases (1): new" in text
    assert "removed cases" not in text
    assert "2 improved" in text


def test_experiment_all_passed_and_empty() -> None:
    experiment = Experiment(name="x", suite="toy", created_at=CREATED)
    text = _export(experiment_renderable(ExperimentReport.from_runs(experiment, [run("a")])))
    assert "✓ all 1 cases passed" in text
    assert "model" not in text
    assert "Failures" not in text

    empty = _export(experiment_renderable(ExperimentReport.from_runs(experiment, [])))
    assert "0/0 passed" in empty
    assert "Metrics" not in empty
    assert "all 0 cases" not in empty


def test_experiment_single_error_is_singular() -> None:
    experiment = Experiment(name="x", suite="toy", created_at=CREATED)
    report = ExperimentReport.from_runs(experiment, [run("a", error="E: x"), run("b")])
    assert "1/2 passed (50%), 1 error\n" in _export(experiment_renderable(report))


def test_gates_panel_skipped_gates_and_no_gates() -> None:
    runs = [run("a")]
    gates = evaluate_gates(
        Gates.from_dict({"pass_rate": {"min_delta": -0.1}, "cost_usd": {"max": 1}}), run=runs
    )
    report = ExperimentReport.from_runs(CANDIDATE, runs, gates=gates)
    text = _export(experiment_renderable(report))
    assert "✗ Gates FAILED" in text  # cost_usd missing: fails by default
    assert "SKIP" in text
    assert "min_delta -0.1" in text

    empty = Comparison(resamples=10, confidence=0.95)
    panel = ComparisonReport(comparison=empty, gates=GateReport(passed=True))
    text = _export(comparison_renderable(panel))
    assert "✓ Gates passed" in text
    assert "no gates" in text


def test_experiment_label_marks_the_baseline() -> None:
    assert terminal._experiment_label(BASELINE).plain == "main (exp-base) baseline"
    assert terminal._experiment_label(None).plain == "(in memory)"
