"""Minimal reporters: plain text and JSON.

Plain text is what the CLI prints; it has no colours or terminal tricks, so it
reads the same in a terminal, a CI log or a file. The rich terminal report
(#26), Markdown (#27) and HTML (#28) are separate renderers registered in
:mod:`agentlens.report.render`.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from agentlens.compare.gates import GateReport
from agentlens.compare.model import Comparison, Status
from agentlens.report.model import ComparisonReport, ExperimentReport
from agentlens.schema import Experiment

__all__ = ["JSONRenderer", "PlainTextRenderer", "fmt_value", "table"]

MISSING = "-"
"""Shown for a value that is not known (e.g. no costs recorded)."""


def fmt_value(value: float | None, *, signed: bool = False) -> str:
    """A number with 4 significant digits (``-`` for ``None``), ``+`` sign if ``signed``."""
    if value is None:
        return MISSING
    return f"{value:+.4g}" if signed else f"{value:.4g}"


def _pct(value: float) -> str:
    return f"{value:.0%}"


def table(headers: Sequence[str], rows: Sequence[Sequence[str]], *, indent: str = "") -> str:
    """Left-aligned columns separated by two spaces (no trailing spaces)."""
    widths = [len(h) for h in headers]
    for row in rows:
        widths = [max(w, len(cell)) for w, cell in zip(widths, row, strict=True)]
    lines = []
    for row in [list(headers), *rows]:
        cells = [cell.ljust(width) for cell, width in zip(row, widths, strict=True)]
        lines.append(indent + "  ".join(cells).rstrip())
    return "\n".join(lines)


def _interval(low: float | None, high: float | None) -> str:
    if low is None or high is None:
        return MISSING
    return f"[{fmt_value(low, signed=True)}, {fmt_value(high, signed=True)}]"


def _when(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S UTC")


def _experiment_label(experiment: Experiment | None) -> str:
    if experiment is None:
        return "(in memory)"
    mark = " [baseline]" if experiment.is_baseline else ""
    return f"{experiment.name} ({experiment.id}){mark}"


def _git(experiment: Experiment) -> str | None:
    sha = experiment.config.get("git_sha")
    if not isinstance(sha, str):
        return None
    return f"{sha[:12]}{' (dirty)' if experiment.config.get('git_dirty') else ''}"


def _gates_text(gates: GateReport | None) -> list[str]:
    return ["", gates.summary()] if gates is not None else []


class PlainTextRenderer:
    """Plain-text reports (the CLI's default output)."""

    def experiment(self, report: ExperimentReport) -> str:
        """Header, totals, aggregate metrics, per-case table and gates."""
        exp = report.experiment
        header = [
            ("experiment", _experiment_label(exp)),
            ("suite", exp.suite),
            ("created", _when(exp.created_at)),
        ]
        model = exp.config.get("model")
        if isinstance(model, str):
            header.append(("model", model))
        git = _git(exp)
        if git:
            header.append(("git", git))
        errors = f", {report.errors} error{'s' if report.errors != 1 else ''}"
        header.append(
            (
                "runs",
                f"{report.passed}/{report.runs} passed ({_pct(report.pass_rate)})"
                + (errors if report.errors else ""),
            )
        )
        width = max(len(key) for key, _ in header)
        lines = [f"{key.ljust(width)}  {value}" for key, value in header]
        if report.metrics:
            lines += ["", "metrics (mean over cases)"]
            rows = [[name, fmt_value(value)] for name, value in report.metrics.items()]
            lines.append(table(["metric", "value"], rows, indent="  "))
        if report.cases:
            lines += ["", "cases"]
            rows = [
                [
                    case.case_id,
                    f"{case.passed}/{case.runs}",
                    _pct(case.pass_rate),
                    str(case.errors) if case.errors else "",
                    ", ".join(case.failed_evaluators),
                ]
                for case in report.cases
            ]
            lines.append(
                table(["case", "passed", "rate", "errors", "failed evaluators"], rows, indent="  ")
            )
        lines += _gates_text(report.gates)
        return "\n".join(lines)

    def comparison(self, report: ComparisonReport) -> str:
        """Header, metric deltas with CIs, case regressions/improvements and gates."""
        cmp = report.comparison
        level = f"{cmp.confidence:.0%}"
        lines = [
            f"baseline   {_experiment_label(cmp.baseline)}",
            f"candidate  {_experiment_label(cmp.candidate)}",
            f"cases      {len(cmp.cases)} paired"
            + (f", {len(cmp.added_cases)} added" if cmp.added_cases else "")
            + (f", {len(cmp.removed_cases)} removed" if cmp.removed_cases else "")
            + f"; paired bootstrap, {cmp.resamples} resamples, {level} CI, seed {cmp.seed}",
        ]
        rows = [
            [
                m.name,
                str(m.n),
                fmt_value(m.baseline),
                fmt_value(m.candidate),
                fmt_value(m.delta, signed=True),
                _interval(m.ci_low, m.ci_high),
                m.status.value.upper() if m.status is Status.REGRESSION else m.status.value,
            ]
            for m in cmp.metrics
        ]
        lines += [
            "",
            table(
                ["metric", "n", "baseline", "candidate", "delta", f"{level} CI", "status"],
                rows,
            ),
        ]
        lines += _case_lines(cmp)
        lines += _gates_text(report.gates)
        return "\n".join(lines)


def _case_lines(cmp: Comparison) -> list[str]:
    lines: list[str] = []
    for title, cases in (
        ("case regressions", cmp.case_regressions),
        ("case improvements", cmp.case_improvements),
    ):
        if not cases:
            continue
        lines += ["", f"{title} ({len(cases)})"]
        rows = []
        for case in cases:
            rate = case.pass_rate
            rows.append(
                [
                    case.case_id,
                    f"{fmt_value(rate.baseline)} -> {fmt_value(rate.candidate)}",
                    ", ".join(case.regressed_evaluators),
                ]
            )
        lines.append(table(["case", "pass rate", "regressed evaluators"], rows, indent="  "))
    for title, ids in (("added cases", cmp.added_cases), ("removed cases", cmp.removed_cases)):
        if ids:
            lines += ["", f"{title} ({len(ids)}): {', '.join(ids)}"]
    return lines


class JSONRenderer:
    """The report model as indented JSON (readable back with :func:`load_report`)."""

    def experiment(self, report: ExperimentReport) -> str:
        """``ExperimentReport`` JSON."""
        return report.model_dump_json(indent=2)

    def comparison(self, report: ComparisonReport) -> str:
        """``ComparisonReport`` JSON."""
        return report.model_dump_json(indent=2)
