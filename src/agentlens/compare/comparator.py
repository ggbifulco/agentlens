"""Compare a candidate experiment against a baseline (FR-S4, ``docs/design.md`` §7, ADR 0005).

::

    from agentlens.compare import compare_experiments
    from agentlens.store import SQLiteStore

    with SQLiteStore("runs.db") as store:
        comparison = compare_experiments(store, "main", "pr-123")
    for metric in comparison.regressions:
        print(metric.name, metric.delta, (metric.ci_low, metric.ci_high))

**Metrics.** Per case, each averaged over the case's repetitions:
``pass_rate`` (fraction of runs with no error and every evaluator passed, as
:attr:`agentlens.suite.RunResult.passed`), ``score.<evaluator>`` (mean score),
``cost_usd`` (runs with a known cost only; unknown is not free), ``latency_ms``
and ``steps``.

**Pairing.** Cases are matched by id; only cases present in both experiments
count (the others are listed as added/removed). A metric's aggregate delta is
the mean over the cases where *both* sides have a value of ``candidate -
baseline``, with a paired bootstrap CI (:func:`paired_bootstrap`) over those
cases. A metric regresses only when its CI lies entirely on the bad side of 0
(below for pass rate and scores, above for cost, latency and steps).
"""

from __future__ import annotations

import logging
import math
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass

from agentlens.compare.bootstrap import DEFAULT_CONFIDENCE, DEFAULT_RESAMPLES, paired_bootstrap
from agentlens.compare.model import (
    CaseComparison,
    CaseMetric,
    Comparison,
    Direction,
    MetricDelta,
    Status,
)
from agentlens.schema import Experiment
from agentlens.store.base import Store
from agentlens.suite.experiment import find_experiment
from agentlens.suite.runner import ExperimentRun, RunResult

__all__ = [
    "LOWER_IS_BETTER",
    "PASS_RATE",
    "SCORE_PREFIX",
    "classify",
    "compare_experiments",
    "compare_runs",
    "experiment_runs",
    "metric_direction",
    "run_metrics",
]

logger = logging.getLogger("agentlens.compare")

PASS_RATE = "pass_rate"
SCORE_PREFIX = "score."
"""Metric name of an evaluator's mean score: ``score.<evaluator>``."""
COST = "cost_usd"
LATENCY = "latency_ms"
STEPS = "steps"
LOWER_IS_BETTER = frozenset({COST, LATENCY, STEPS})
"""Metrics where a decrease is an improvement; every other metric is higher-is-better."""


def metric_direction(name: str) -> Direction:
    """Whether metric ``name`` should go up (pass rate, scores) or down (cost, latency, steps)."""
    return Direction.LOWER_IS_BETTER if name in LOWER_IS_BETTER else Direction.HIGHER_IS_BETTER


def classify(direction: Direction, ci_low: float, ci_high: float) -> Status:
    """Regression/improvement when the CI ``[ci_low, ci_high]`` excludes 0, else unchanged."""
    if ci_high < 0:
        worse = direction is Direction.HIGHER_IS_BETTER
    elif ci_low > 0:
        worse = direction is Direction.LOWER_IS_BETTER
    else:
        return Status.UNCHANGED
    return Status.REGRESSION if worse else Status.IMPROVEMENT


# -- per-case values ---------------------------------------------------------------


@dataclass(frozen=True)
class _CaseValues:
    runs: int
    metrics: dict[str, float]
    """Metric name -> mean over the runs that have a value (absent: no run has one)."""
    evaluator_pass: dict[str, float]
    """Evaluator -> fraction of its results that passed."""


def _mean(values: Iterable[float | None]) -> float | None:
    known = [v for v in values if v is not None]
    return math.fsum(known) / len(known) if known else None


def _case_values(runs: Sequence[RunResult]) -> _CaseValues:
    metrics: dict[str, float] = {PASS_RATE: sum(run.passed for run in runs) / len(runs)}
    scores: dict[str, list[float]] = {}
    passes: dict[str, list[float]] = {}
    for run in runs:
        for result in run.eval_results:
            scores.setdefault(result.evaluator, []).append(result.score)
            passes.setdefault(result.evaluator, []).append(float(result.passed))
    for evaluator, values in sorted(scores.items()):
        metrics[SCORE_PREFIX + evaluator] = math.fsum(values) / len(values)
    extractors: dict[str, Callable[[RunResult], float | None]] = {
        COST: lambda run: run.trace.total_cost,
        LATENCY: lambda run: run.trace.latency_ms,
        STEPS: lambda run: float(run.trace.n_steps),
    }
    for name, extract in extractors.items():
        value = _mean(extract(run) for run in runs)
        if value is not None:
            metrics[name] = value
    evaluator_pass = {name: math.fsum(v) / len(v) for name, v in passes.items()}
    return _CaseValues(len(runs), metrics, evaluator_pass)


def _group_by_case(runs: Iterable[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for run in runs:
        grouped.setdefault(run.case_id, []).append(run)
    for case_runs in grouped.values():
        case_runs.sort(key=lambda run: run.repetition)
    return grouped


def _metric_order(name: str) -> tuple[int, str]:
    if name == PASS_RATE:
        return (0, name)
    if name.startswith(SCORE_PREFIX):
        return (1, name)
    return (2 + [COST, LATENCY, STEPS].index(name), name)


def run_metrics(runs: ExperimentRun | Iterable[RunResult]) -> dict[str, tuple[float, int]]:
    """Absolute metrics of one experiment, computed like the comparison's.

    Each case is first averaged over its repetitions, then each metric is the
    mean over the cases that have a value for it. Used to gate a single run
    without a baseline.

    Returns:
        ``{metric name: (mean over cases, number of cases with a value)}``, in
        the comparison's metric order; ``cost_usd``/``latency_ms`` are absent
        when no run has a value.
    """
    if isinstance(runs, ExperimentRun):
        runs = runs.runs
    per_case: dict[str, list[float]] = {}
    for case_runs in _group_by_case(runs).values():
        for name, value in _case_values(case_runs).metrics.items():
            per_case.setdefault(name, []).append(value)
    return {
        name: (math.fsum(per_case[name]) / len(per_case[name]), len(per_case[name]))
        for name in sorted(per_case, key=_metric_order)
    }


# -- comparison ----------------------------------------------------------------------


def _aggregate(
    name: str,
    pairs: Sequence[tuple[float, float]],
    *,
    resamples: int,
    confidence: float,
    seed: int | None,
) -> MetricDelta:
    direction = metric_direction(name)
    evaluator = name[len(SCORE_PREFIX) :] if name.startswith(SCORE_PREFIX) else None
    if not pairs:
        return MetricDelta(
            name=name, direction=direction, evaluator=evaluator, n=0, status=Status.NO_DATA
        )
    n = len(pairs)
    baseline = math.fsum(b for b, _ in pairs) / n
    candidate = math.fsum(c for _, c in pairs) / n
    diffs = [c - b for b, c in pairs]
    delta = math.fsum(diffs) / n
    low, high = paired_bootstrap(diffs, resamples=resamples, confidence=confidence, seed=seed)
    return MetricDelta(
        name=name,
        direction=direction,
        evaluator=evaluator,
        n=n,
        baseline=baseline,
        candidate=candidate,
        delta=delta,
        delta_pct=delta / abs(baseline) * 100 if baseline != 0 else None,
        ci_low=low,
        ci_high=high,
        status=classify(direction, low, high),
    )


def _compare_case(case_id: str, base: _CaseValues, cand: _CaseValues) -> CaseComparison:
    names = sorted(set(base.metrics) | set(cand.metrics), key=_metric_order)
    metrics: dict[str, CaseMetric] = {}
    for name in names:
        b, c = base.metrics.get(name), cand.metrics.get(name)
        delta = c - b if b is not None and c is not None else None
        metrics[name] = CaseMetric(baseline=b, candidate=c, delta=delta)
    pass_delta = cand.metrics[PASS_RATE] - base.metrics[PASS_RATE]
    if pass_delta < 0:
        status = Status.REGRESSION
    elif pass_delta > 0:
        status = Status.IMPROVEMENT
    else:
        status = Status.UNCHANGED
    regressed = sorted(
        name
        for name, rate in base.evaluator_pass.items()
        if name in cand.evaluator_pass and cand.evaluator_pass[name] < rate
    )
    return CaseComparison(
        case_id=case_id,
        baseline_runs=base.runs,
        candidate_runs=cand.runs,
        metrics=metrics,
        status=status,
        regressed_evaluators=regressed,
    )


def compare_runs(
    baseline: ExperimentRun | Iterable[RunResult],
    candidate: ExperimentRun | Iterable[RunResult],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = 0,
    confidence: float = DEFAULT_CONFIDENCE,
    baseline_experiment: Experiment | None = None,
    candidate_experiment: Experiment | None = None,
) -> Comparison:
    """Compare two sets of in-memory runs (e.g. two :func:`~agentlens.suite.run_suite` results).

    Args:
        baseline: The reference runs: an :class:`ExperimentRun` or any
            iterable of :class:`RunResult` (grouped by ``case_id``).
        candidate: The runs under test, same forms.
        resamples: Bootstrap resamples per metric (ADR 0005: 10 000).
        seed: Bootstrap seed, so the same inputs give the same CIs; ``None``
            for a non-reproducible run.
        confidence: Confidence level of the CIs.
        baseline_experiment: Recorded on the result (default: the
            ``ExperimentRun``'s experiment, if one was given).
        candidate_experiment: Same, for the candidate.

    Returns:
        A :class:`Comparison` with aggregate :class:`MetricDelta` per metric,
        a :class:`CaseComparison` per shared case and the added/removed cases.

    Raises:
        ValueError: for ``resamples < 1`` or ``confidence`` outside ``(0, 1)``.
    """
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    if isinstance(baseline, ExperimentRun):
        baseline_experiment = baseline_experiment or baseline.experiment
        baseline = baseline.runs
    if isinstance(candidate, ExperimentRun):
        candidate_experiment = candidate_experiment or candidate.experiment
        candidate = candidate.runs
    base_cases = {cid: _case_values(runs) for cid, runs in _group_by_case(baseline).items()}
    cand_cases = {cid: _case_values(runs) for cid, runs in _group_by_case(candidate).items()}
    shared = [cid for cid in base_cases if cid in cand_cases]

    pairs: dict[str, list[tuple[float, float]]] = {}
    names: set[str] = set()
    for cid in shared:
        base, cand = base_cases[cid].metrics, cand_cases[cid].metrics
        names.update(base, cand)
        for name in base.keys() & cand.keys():
            pairs.setdefault(name, []).append((base[name], cand[name]))
    names.update({PASS_RATE, COST, LATENCY, STEPS})

    return Comparison(
        baseline=baseline_experiment,
        candidate=candidate_experiment,
        metrics=[
            _aggregate(
                name, pairs.get(name, []), resamples=resamples, confidence=confidence, seed=seed
            )
            for name in sorted(names, key=_metric_order)
        ],
        cases=[_compare_case(cid, base_cases[cid], cand_cases[cid]) for cid in shared],
        added_cases=[cid for cid in cand_cases if cid not in base_cases],
        removed_cases=[cid for cid in base_cases if cid not in cand_cases],
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


# -- from a store -------------------------------------------------------------------


def experiment_runs(store: Store, experiment: str | Experiment) -> list[RunResult]:
    """Rebuild the runs of a stored experiment (traces + their eval results).

    Each trace's ``case_id`` and ``repetition`` come from its metadata, as
    recorded by :func:`~agentlens.suite.run_suite`. Traces without a string
    ``case_id`` (saved under the experiment by other means) are skipped with
    a warning. Runs are returned oldest first.

    Raises:
        ExperimentNotFoundError: if the experiment is not stored.
    """
    found = find_experiment(store, experiment)
    results = store.list_eval_results(found.id)
    runs: list[RunResult] = []
    skipped = 0
    for trace in reversed(store.list_traces(experiment_id=found.id)):  # oldest first
        case_id = trace.metadata.get("case_id")
        if not isinstance(case_id, str):
            skipped += 1
            continue
        repetition = trace.metadata.get("repetition")
        runs.append(
            RunResult(
                case_id=case_id,
                repetition=repetition if isinstance(repetition, int) else 0,
                trace=trace,
                eval_results=results.get(trace.id, []),
            )
        )
    if skipped:
        logger.warning(
            "experiment %s: skipped %d trace(s) without a case_id in their metadata",
            found.id,
            skipped,
        )
    return runs


def compare_experiments(
    store: Store,
    baseline: str | Experiment,
    candidate: str | Experiment,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int | None = 0,
    confidence: float = DEFAULT_CONFIDENCE,
) -> Comparison:
    """Compare two stored experiments (each an id, a name or an :class:`Experiment`).

    Loads both with :func:`experiment_runs` and calls :func:`compare_runs`;
    see there for the arguments. A name resolves to the most recent
    experiment with that name (:func:`~agentlens.suite.find_experiment`).

    Raises:
        ExperimentNotFoundError: if either experiment is not stored.
        ValueError: for ``resamples < 1`` or ``confidence`` outside ``(0, 1)``.
    """
    base = find_experiment(store, baseline)
    cand = find_experiment(store, candidate)
    return compare_runs(
        experiment_runs(store, base),
        experiment_runs(store, cand),
        resamples=resamples,
        seed=seed,
        confidence=confidence,
        baseline_experiment=base,
        candidate_experiment=cand,
    )
