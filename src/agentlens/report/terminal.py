"""The rich terminal report (FR-C1): coloured tables for humans at a terminal.

:class:`RichRenderer` is registered as the ``rich`` format, so
``render(report, "rich")`` returns the report as text with ANSI colours. To
print to a console of your own (e.g. one that records for an export), build the
renderables with :func:`experiment_renderable` / :func:`comparison_renderable`::

    from rich.console import Console
    from agentlens.report import comparison_renderable

    Console().print(comparison_renderable(report))

Colours carry meaning, never alone: a regression is red *and* says
``REGRESSION``, an improvement green, an unchanged metric dim. A delta has an
arrow for the way the value moved (``↑``/``↓``) and the colour says whether
that is good, following the metric's :class:`~agentlens.compare.Direction`
(lower cost is good, lower pass rate is bad).
"""

from __future__ import annotations

import io
import re
import shutil
from typing import Any

from rich import box
from rich.console import Console, Group, RenderableType
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from agentlens.compare.gates import GateReport, GateResult
from agentlens.compare.model import CaseComparison, Comparison, Direction, MetricDelta, Status
from agentlens.report.model import ComparisonReport, ExperimentReport
from agentlens.report.plain import MISSING, fmt_value
from agentlens.schema import Experiment

__all__ = [
    "DEFAULT_WIDTH",
    "RichRenderer",
    "comparison_renderable",
    "experiment_renderable",
]

DEFAULT_WIDTH = 100
"""Width used when the terminal size is unknown."""

_BOX = box.SIMPLE_HEAD
"""Tables: a rule under the header, no other lines (saves width for numbers)."""

_TABLE: dict[str, Any] = {
    "box": _BOX,
    "show_edge": False,
    "pad_edge": False,
    "padding": (0, 1, 0, 0),  # with the box separator: two spaces between columns
}

_TRAILING_PADDING = re.compile(r" +((?:\x1b\[[0-9;]*m)*)$")
"""Spaces at the end of a line, possibly followed by ANSI style codes."""

_STATUS_STYLE = {
    Status.REGRESSION: "bold red",
    Status.IMPROVEMENT: "green",
    Status.UNCHANGED: "dim",
    Status.NO_DATA: "dim",
}
_STATUS_LABEL = {
    Status.REGRESSION: "REGRESSION",
    Status.IMPROVEMENT: "improvement",
    Status.UNCHANGED: "unchanged",
    Status.NO_DATA: "no data",
}


# -- shared pieces -------------------------------------------------------------------


def _experiment_label(experiment: Experiment | None) -> Text:
    if experiment is None:
        return Text("(in memory)", style="dim")
    label = Text.assemble((experiment.name, "bold"), f" ({experiment.id})")
    if experiment.is_baseline:
        label.append(" baseline", style="cyan")
    return label


def _header(rows: list[tuple[str, RenderableType]]) -> Table:
    grid = Table.grid(padding=(0, 2))
    grid.add_column(style="bold")
    grid.add_column()
    for key, value in rows:
        grid.add_row(key, value)
    return grid


def _pct(value: float) -> str:
    return f"{value:.0%}"


def _rate_style(rate: float) -> str:
    return "green" if rate >= 1.0 else "red" if rate <= 0.0 else "yellow"


def _bound(rule: str, value: float) -> str:
    if rule.endswith("_pct"):
        return f"{value:+.4g}%"
    return fmt_value(value, signed=rule.endswith("_delta"))


def _gate_row(result: GateResult) -> tuple[Text, ...]:
    if result.skipped:
        verdict = Text("SKIP", style="dim")
    elif result.passed:
        verdict = Text("✓ PASS", style="green")
    else:
        verdict = Text("✗ FAIL", style="bold red")
    return (
        verdict,
        Text(result.name),
        Text(f"{result.rule} {_bound(result.rule, result.threshold)}"),
        Text(MISSING if result.actual is None else _bound(result.rule, result.actual)),
        Text(result.reason, style="dim" if result.passed else ""),
    )


def _gates_panel(gates: GateReport) -> Panel:
    table = Table(**_TABLE)
    for header in ("", "gate", "rule", "actual", "reason"):
        table.add_column(header)
    table.columns[3].justify = "right"
    for result in gates.results:
        table.add_row(*_gate_row(result))
    n_skipped, n_failed = len(gates.skipped), len(gates.failures)
    n_passed = len(gates.results) - n_skipped - n_failed
    counts = f"{n_passed} passed, {n_failed} failed, {n_skipped} skipped"
    parts: list[RenderableType] = [table] if gates.results else []
    parts += [Text(f"note: {note}", style="dim") for note in gates.notes]
    if gates.passed:
        title = Text.assemble(("✓ Gates passed", "bold green"), f"  {counts}")
    else:
        title = Text.assemble(("✗ Gates FAILED", "bold red"), f"  {counts}")
    return Panel(
        Group(*parts) if parts else Text("no gates", style="dim"),
        title=title,
        title_align="left",
        border_style="green" if gates.passed else "red",
    )


# -- experiment ----------------------------------------------------------------------


def _metric_label(name: str) -> str:
    return {
        "pass_rate": "pass rate",
        "cost_usd": "cost (USD)",
        "latency_ms": "latency (ms)",
    }.get(name, name)


def experiment_renderable(report: ExperimentReport) -> RenderableType:
    """One experiment's results: header, metrics, failing cases and gates."""
    exp = report.experiment
    rows: list[tuple[str, RenderableType]] = [
        ("experiment", _experiment_label(exp)),
        ("suite", exp.suite),
        ("created", exp.created_at.strftime("%Y-%m-%d %H:%M:%S UTC")),
    ]
    model = exp.config.get("model")
    if isinstance(model, str):
        rows.append(("model", model))
    sha = exp.config.get("git_sha")
    if isinstance(sha, str):
        rows.append(("git", f"{sha[:12]}{' (dirty)' if exp.config.get('git_dirty') else ''}"))
    runs = Text.assemble(
        f"{report.passed}/{report.runs} passed ",
        (f"({_pct(report.pass_rate)})", _rate_style(report.pass_rate) if report.runs else "dim"),
    )
    if report.errors:
        runs.append(f", {report.errors} error{'s' if report.errors != 1 else ''}", style="red")
    rows.append(("runs", runs))
    parts: list[RenderableType] = [_header(rows)]

    if report.metrics:
        metrics = Table(title="Metrics", title_justify="left", **_TABLE)
        metrics.add_column("metric")
        metrics.add_column("mean over cases", justify="right")
        for name, value in report.metrics.items():
            cell = Text(fmt_value(value))
            if name == "pass_rate":
                cell = Text(_pct(value), style=_rate_style(value))
            metrics.add_row(_metric_label(name), cell)
        parts.append(metrics)

    failing = [case for case in report.cases if case.passed < case.runs]
    if failing:
        table = Table(
            title=f"Failures ({len(failing)} of {len(report.cases)} cases)",
            title_justify="left",
            title_style="bold red",
            **_TABLE,
        )
        table.add_column("case")
        table.add_column("passed", justify="right")
        table.add_column("errors", justify="right")
        table.add_column("failed evaluators")
        for case in failing:
            table.add_row(
                case.case_id,
                Text(f"{case.passed}/{case.runs}", style=_rate_style(case.pass_rate)),
                Text(str(case.errors), style="red") if case.errors else Text(""),
                ", ".join(case.failed_evaluators),
            )
        parts.append(table)
    elif report.cases:
        parts.append(Text(f"✓ all {len(report.cases)} cases passed", style="green"))

    if report.gates is not None:
        parts.append(_gates_panel(report.gates))
    return _spaced(parts)


# -- comparison ----------------------------------------------------------------------


def _delta_cell(metric: MetricDelta) -> Text:
    if metric.delta is None:
        return Text(MISSING, style="dim")
    arrow = "↑ " if metric.delta > 0 else "↓ " if metric.delta < 0 else "  "
    return Text(arrow + fmt_value(metric.delta, signed=True), style=_STATUS_STYLE[metric.status])


def _delta_pct_cell(metric: MetricDelta) -> Text:
    if metric.delta_pct is None:
        return Text(MISSING, style="dim")
    return Text(f"{metric.delta_pct:+.1f}%", style=_STATUS_STYLE[metric.status])


def _ci_cell(metric: MetricDelta) -> Text:
    if metric.ci_low is None or metric.ci_high is None:
        return Text(MISSING, style="dim")
    low, high = fmt_value(metric.ci_low, signed=True), fmt_value(metric.ci_high, signed=True)
    return Text(f"[{low}, {high}]", style="" if metric.status is Status.UNCHANGED else "bold")


def _status_cell(status: Status) -> Text:
    return Text(_STATUS_LABEL[status], style=_STATUS_STYLE[status])


def _metrics_table(cmp: Comparison) -> Table:
    level = f"{cmp.confidence:.0%}"
    table = Table(
        title="Metrics (candidate vs baseline, paired cases)",
        title_justify="left",
        **_TABLE,
        caption=f"↑ higher is better, ↓ lower is better; changed = the {level} CI excludes 0",
        caption_justify="left",
    )
    table.add_column("metric", overflow="fold")
    table.add_column("n", justify="right", no_wrap=True)
    table.add_column("baseline", justify="right", no_wrap=True)
    table.add_column("candidate", justify="right", no_wrap=True)
    table.add_column("delta", justify="right", no_wrap=True)
    table.add_column("delta %", justify="right", no_wrap=True)
    table.add_column(f"{level} CI", justify="right", overflow="fold")
    table.add_column("status", no_wrap=True, min_width=11)
    for m in cmp.metrics:
        better = "↑" if m.direction is Direction.HIGHER_IS_BETTER else "↓"
        table.add_row(
            Text.assemble(
                (_metric_label(m.name), "bold" if m.status is Status.REGRESSION else ""),
                (f" {better}", "dim"),
            ),
            str(m.n),
            fmt_value(m.baseline),
            fmt_value(m.candidate),
            _delta_cell(m),
            _delta_pct_cell(m),
            _ci_cell(m),
            _status_cell(m.status),
        )
    return table


def _cases_table(title: str, cases: list[CaseComparison], style: str) -> Table:
    table = Table(
        title=title,
        title_justify="left",
        title_style=style,
        **_TABLE,
    )
    table.add_column("case")
    table.add_column("pass rate", justify="right")
    table.add_column("runs (base/cand)", justify="right")
    table.add_column("regressed evaluators")
    for case in cases:
        rate = case.pass_rate
        table.add_row(
            case.case_id,
            Text(f"{fmt_value(rate.baseline)} → {fmt_value(rate.candidate)}", style=style),
            f"{case.baseline_runs}/{case.candidate_runs}",
            ", ".join(case.regressed_evaluators),
        )
    return table


def comparison_renderable(report: ComparisonReport) -> RenderableType:
    """A comparison: header, metric deltas with CIs, per-case changes and gates."""
    cmp = report.comparison
    cases = Text(f"{len(cmp.cases)} paired")
    if cmp.added_cases:
        cases.append(f", {len(cmp.added_cases)} added")
    if cmp.removed_cases:
        cases.append(f", {len(cmp.removed_cases)} removed")
    method = (
        f"paired bootstrap, {cmp.resamples} resamples, {cmp.confidence:.0%} CI, seed {cmp.seed}"
    )
    parts: list[RenderableType] = [
        _header(
            [
                ("baseline", _experiment_label(cmp.baseline)),
                ("candidate", _experiment_label(cmp.candidate)),
                ("cases", cases),
                ("method", Text(method, style="dim")),
            ]
        ),
        _metrics_table(cmp),
    ]
    verdict = (
        Text(f"✗ {len(cmp.regressions)} metric(s) regressed", style="bold red")
        if cmp.regressions
        else Text("✓ no metric regressed", style="green")
    )
    if cmp.improvements:
        verdict.append(f"; {len(cmp.improvements)} improved", style="green")
    parts.append(verdict)
    if cmp.case_regressions:
        title = f"Case regressions ({len(cmp.case_regressions)})"
        parts.append(_cases_table(title, cmp.case_regressions, "red"))
    if cmp.case_improvements:
        title = f"Case improvements ({len(cmp.case_improvements)})"
        parts.append(_cases_table(title, cmp.case_improvements, "green"))
    unpaired = [
        Text(f"{label} ({len(ids)}): {', '.join(ids)}", style="dim")
        for label, ids in (("added cases", cmp.added_cases), ("removed cases", cmp.removed_cases))
        if ids
    ]
    if unpaired:
        parts.append(Group(*unpaired))
    if report.gates is not None:
        parts.append(_gates_panel(report.gates))
    return _spaced(parts)


def _spaced(parts: list[RenderableType]) -> Group:
    spaced: list[RenderableType] = []
    for part in parts:
        if spaced:
            spaced.append(Text(""))
        spaced.append(part)
    return Group(*spaced)


# -- renderer ------------------------------------------------------------------------


class RichRenderer:
    """Renders reports with rich into a string (ANSI colours unless disabled).

    Args:
        width: Line width; default the terminal's width (``$COLUMNS``), else
            :data:`DEFAULT_WIDTH`.
        color: ``False`` for the same layout without colours. Rich also
            honours ``$NO_COLOR``.
    """

    def __init__(self, *, width: int | None = None, color: bool = True) -> None:
        self.width = width
        self.color = color

    def _render(self, renderable: RenderableType) -> str:
        width = self.width or shutil.get_terminal_size((DEFAULT_WIDTH, 24)).columns
        buffer = io.StringIO()
        console = Console(
            file=buffer,
            width=width,
            force_terminal=self.color,
            color_system="auto" if self.color else None,
            highlight=False,
            emoji=False,
        )
        console.print(renderable)
        # Drop the padding rich adds up to the full width (noise when copied or diffed).
        lines = buffer.getvalue().rstrip("\n").split("\n")
        return "\n".join(_TRAILING_PADDING.sub(r"\1", line) for line in lines)

    def experiment(self, report: ExperimentReport) -> str:
        """One experiment's results (see :func:`experiment_renderable`)."""
        return self._render(experiment_renderable(report))

    def comparison(self, report: ComparisonReport) -> str:
        """A comparison (see :func:`comparison_renderable`)."""
        return self._render(comparison_renderable(report))
