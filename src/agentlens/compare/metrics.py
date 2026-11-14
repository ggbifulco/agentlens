"""Per-case metrics of an experiment: what the comparator compares.

Every run (one trace + its eval results) gives a :class:`RunMetrics`; the runs of
one case are averaged into a :class:`CaseMetrics`, so a case run 5 times weighs
the same as a case run once and the bootstrap resamples *cases* (ADR 0005).

Metric names (``docs/design.md`` §7):

==================  =========================================  =================
name                per run                                    better
==================  =========================================  =================
``pass_rate``       1 if no error and every evaluator passed   higher
``score:<name>``    the evaluator's normalised score            higher
``cost_usd``        :attr:`Trace.total_cost` (when known)       lower
``latency_ms``      :attr:`Trace.latency_ms` (when ended)       lower
``steps``           :attr:`Trace.n_steps`                       lower
==================  =========================================  =================

A metric that a case does not have (no cost recorded, an evaluator not run) is
simply absent from that case; it is never filled with zero, since unknown cost
is not free.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from agentlens.schema import EvalResult, Experiment, Trace
from agentlens.store.base import Store
from agentlens.suite.experiment import find_experiment
from agentlens.suite.runner import ExperimentRun

__all__ = [
    "COST",
    "LATENCY",
    "PASS_RATE",
    "SCORE_PREFIX",
    "STEPS",
    "CaseMetrics",
    "Direction",
    "ExperimentMetrics",
    "RunMetrics",
    "collect_metrics",
    "metric_direction",
    "metrics_from_run",
    "run_metrics",
    "score_metric",
]

PASS_RATE = "pass_rate"
COST = "cost_usd"
LATENCY = "latency_ms"
STEPS = "steps"
SCORE_PREFIX = "score:"


def score_metric(evaluator: str) -> str:
    """The metric name of an evaluator's score: ``"score:<evaluator>"``."""
    return f"{SCORE_PREFIX}{evaluator}"


class Direction(StrEnum):
    """Which way a metric should move."""

    HIGHER_IS_BETTER = "higher"
    LOWER_IS_BETTER = "lower"


def metric_direction(metric: str) -> Direction:
    """``pass_rate`` and scores should go up; cost, latency and steps down.

    Raises:
        ValueError: for an unknown metric name.
    """
    if metric == PASS_RATE or metric.startswith(SCORE_PREFIX):
        return Direction.HIGHER_IS_BETTER
    if metric in (COST, LATENCY, STEPS):
        return Direction.LOWER_IS_BETTER
    raise ValueError(f"unknown metric {metric!r}")


@dataclass(frozen=True)
class RunMetrics:
    """The measurements of one run (one repetition of one case)."""

    case_id: str
    repetition: int
    passed: bool
    error: str | None
    cost_usd: float | None
    latency_ms: float | None
    steps: int
    scores: dict[str, float] = field(default_factory=dict)
    """Score per evaluator name (the mean, should an evaluator appear twice)."""


def run_metrics(
    trace: Trace,
    eval_results: Sequence[EvalResult],
    *,
    case_id: str | None = None,
    repetition: int | None = None,
) -> RunMetrics:
    """Measure one run from its trace and eval results.

    ``case_id`` and ``repetition`` default to the trace metadata written by the
    suite runner (falling back to the trace name and 0).
    """
    if case_id is None:
        raw_case = trace.metadata.get("case_id")
        case_id = raw_case if isinstance(raw_case, str) else trace.name
    if repetition is None:
        raw_rep = trace.metadata.get("repetition")
        repetition = raw_rep if isinstance(raw_rep, int) and not isinstance(raw_rep, bool) else 0
    by_name: dict[str, list[float]] = {}
    for result in eval_results:
        by_name.setdefault(result.evaluator, []).append(result.score)
    return RunMetrics(
        case_id=case_id,
        repetition=repetition,
        passed=trace.error is None and all(r.passed for r in eval_results),
        error=trace.error,
        cost_usd=trace.total_cost,
        latency_ms=trace.latency_ms,
        steps=trace.n_steps,
        scores={name: _mean(values) for name, values in by_name.items()},
    )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


@dataclass(frozen=True)
class CaseMetrics:
    """The runs of one case, averaged per metric."""

    case_id: str
    runs: int
    """Number of runs (repetitions) measured."""
    values: dict[str, float]
    """Mean per metric name; metrics no run had are absent."""
    errors: int = 0
    """Runs that raised or timed out."""

    @classmethod
    def from_runs(cls, case_id: str, runs: Sequence[RunMetrics]) -> CaseMetrics:
        """Average ``runs`` (all of case ``case_id``; at least one)."""
        if not runs:
            raise ValueError(f"case {case_id!r} has no runs")
        samples: dict[str, list[float]] = {PASS_RATE: [], STEPS: []}
        for run in runs:
            samples[PASS_RATE].append(1.0 if run.passed else 0.0)
            samples[STEPS].append(float(run.steps))
            if run.cost_usd is not None:
                samples.setdefault(COST, []).append(run.cost_usd)
            if run.latency_ms is not None:
                samples.setdefault(LATENCY, []).append(run.latency_ms)
            for name, score in run.scores.items():
                samples.setdefault(score_metric(name), []).append(score)
        return cls(
            case_id=case_id,
            runs=len(runs),
            values={name: _mean(values) for name, values in samples.items()},
            errors=sum(run.error is not None for run in runs),
        )

    @property
    def pass_rate(self) -> float:
        """Fraction of the runs that passed."""
        return self.values[PASS_RATE]


@dataclass(frozen=True)
class ExperimentMetrics:
    """Per-case metrics of one experiment, keyed by case id."""

    experiment: Experiment
    cases: dict[str, CaseMetrics]
    """In suite order when built from an :class:`ExperimentRun`, else in order of
    each case's first run."""

    @classmethod
    def from_runs(cls, experiment: Experiment, runs: Iterable[RunMetrics]) -> ExperimentMetrics:
        """Group ``runs`` by case (keeping first-seen order) and average them."""
        grouped: dict[str, list[RunMetrics]] = {}
        for run in runs:
            grouped.setdefault(run.case_id, []).append(run)
        return cls(
            experiment=experiment,
            cases={case_id: CaseMetrics.from_runs(case_id, rs) for case_id, rs in grouped.items()},
        )

    @property
    def metric_names(self) -> set[str]:
        """Every metric at least one case has."""
        return {name for case in self.cases.values() for name in case.values}

    def mean(self, metric: str) -> float | None:
        """Mean of ``metric`` over the cases that have it (``None`` if none do)."""
        values = [c.values[metric] for c in self.cases.values() if metric in c.values]
        return _mean(values) if values else None


def collect_metrics(store: Store, experiment: str | Experiment) -> ExperimentMetrics:
    """Load an experiment's traces and eval results from ``store`` and measure them.

    ``experiment`` is an id, a name (the most recent with that name) or an
    :class:`Experiment`. Runs are grouped by their ``case_id`` metadata (set by
    the suite runner); traces without one are grouped by trace name.

    Raises:
        agentlens.suite.ExperimentNotFoundError: if the experiment is not stored.
    """
    found = find_experiment(store, experiment)
    traces = store.list_traces(experiment_id=found.id)
    results: Mapping[str, list[EvalResult]] = store.list_eval_results(found.id)
    traces.sort(key=lambda t: (t.start_time, t.id))
    return ExperimentMetrics.from_runs(
        found, (run_metrics(trace, results.get(trace.id, [])) for trace in traces)
    )


def metrics_from_run(run: ExperimentRun) -> ExperimentMetrics:
    """Measure the result of :func:`agentlens.suite.run_suite` without a store round-trip."""
    return ExperimentMetrics(
        experiment=run.experiment,
        cases={
            case.case_id: CaseMetrics.from_runs(
                case.case_id,
                [
                    run_metrics(r.trace, r.eval_results, case_id=r.case_id, repetition=r.repetition)
                    for r in case.runs
                ],
            )
            for case in run.cases
            if case.runs
        },
    )
