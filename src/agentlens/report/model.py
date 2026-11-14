"""What a report shows: an experiment's results or a comparison, plus gate verdicts.

These models are the input of every renderer (plain text and JSON today, the
rich terminal report, Markdown and HTML later), so a report can be rendered
from a live run, from the store, or from a JSON file written earlier
(``agentlens compare --json``).
"""

from __future__ import annotations

import json
import os
from collections.abc import Iterable
from pathlib import Path

from pydantic import Field, ValidationError

from agentlens.compare.comparator import run_metrics
from agentlens.compare.gates import GateReport
from agentlens.compare.model import Comparison
from agentlens.schema import Experiment
from agentlens.schema._base import SchemaModel
from agentlens.suite.runner import ExperimentRun, RunResult

__all__ = [
    "CaseSummary",
    "ComparisonReport",
    "ExperimentReport",
    "ReportError",
    "load_report",
]


class ReportError(ValueError):
    """A report file could not be read or is not a report AgentLens wrote."""


class CaseSummary(SchemaModel):
    """All repetitions of one case of an experiment."""

    case_id: str
    runs: int = Field(ge=0)
    passed: int = Field(ge=0)
    errors: int = Field(ge=0)
    """Runs that raised or timed out."""
    pass_rate: float
    scores: dict[str, float] = Field(default_factory=dict)
    """Mean score per evaluator over the repetitions."""
    failed_evaluators: list[str] = Field(default_factory=list)
    """Evaluators that failed at least one repetition (sorted)."""


class ExperimentReport(SchemaModel):
    """One experiment's results: totals, aggregate metrics, per-case summaries."""

    experiment: Experiment
    runs: int = Field(ge=0)
    passed: int = Field(ge=0)
    errors: int = Field(ge=0)
    pass_rate: float
    """Fraction of *runs* that passed (``metrics["pass_rate"]`` averages per case first)."""
    metrics: dict[str, float] = Field(default_factory=dict)
    """:func:`~agentlens.compare.run_metrics` means: what absolute gates check."""
    cases: list[CaseSummary] = Field(default_factory=list)
    """In the order the cases first appear in the runs."""
    gates: GateReport | None = None

    @classmethod
    def from_runs(
        cls,
        experiment: Experiment,
        runs: ExperimentRun | Iterable[RunResult],
        *,
        gates: GateReport | None = None,
    ) -> ExperimentReport:
        """Summarise ``runs`` (a :func:`~agentlens.suite.run_suite` result or stored runs)."""
        all_runs = runs.runs if isinstance(runs, ExperimentRun) else list(runs)
        by_case: dict[str, list[RunResult]] = {}
        for run in all_runs:
            by_case.setdefault(run.case_id, []).append(run)
        cases = [_case_summary(case_id, case_runs) for case_id, case_runs in by_case.items()]
        passed = sum(run.passed for run in all_runs)
        return cls(
            experiment=experiment,
            runs=len(all_runs),
            passed=passed,
            errors=sum(run.error is not None for run in all_runs),
            pass_rate=passed / len(all_runs) if all_runs else 0.0,
            metrics={name: mean for name, (mean, _) in run_metrics(all_runs).items()},
            cases=cases,
            gates=gates,
        )


def _case_summary(case_id: str, runs: list[RunResult]) -> CaseSummary:
    scores: dict[str, list[float]] = {}
    failed: set[str] = set()
    for run in runs:
        for result in run.eval_results:
            scores.setdefault(result.evaluator, []).append(result.score)
            if not result.passed:
                failed.add(result.evaluator)
    passed = sum(run.passed for run in runs)
    return CaseSummary(
        case_id=case_id,
        runs=len(runs),
        passed=passed,
        errors=sum(run.error is not None for run in runs),
        pass_rate=passed / len(runs),
        scores={name: sum(values) / len(values) for name, values in scores.items()},
        failed_evaluators=sorted(failed),
    )


class ComparisonReport(SchemaModel):
    """A comparison and, when gates were checked, their verdict.

    This is the file ``agentlens compare --json`` writes and ``agentlens
    report`` reads back.
    """

    comparison: Comparison
    gates: GateReport | None = None


def load_report(path: str | os.PathLike[str]) -> ExperimentReport | ComparisonReport:
    """Read a report JSON file.

    Accepts a :class:`ComparisonReport` (``agentlens compare --json``), an
    :class:`ExperimentReport` (``agentlens report --format json``) or a bare
    :class:`~agentlens.compare.Comparison` (``comparison.model_dump_json()``).

    Raises:
        ReportError: if the file cannot be read or is none of those.
    """
    file = Path(path)
    try:
        data = json.loads(file.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ReportError(f"cannot read report {file}: {exc}") from exc
    if not isinstance(data, dict):
        raise ReportError(f"{file} is not an AgentLens report (expected a JSON object)")
    try:
        if "comparison" in data:
            return ComparisonReport.model_validate(data)
        if isinstance(data.get("metrics"), dict):
            return ExperimentReport.model_validate(data)
        return ComparisonReport(comparison=Comparison.model_validate(data))
    except ValidationError as exc:
        first = exc.errors()[0]
        loc = ".".join(str(part) for part in first["loc"])
        problem = ": ".join(part for part in (loc, first["msg"]) if part)
        raise ReportError(f"{file} is not an AgentLens report: {problem}") from exc
