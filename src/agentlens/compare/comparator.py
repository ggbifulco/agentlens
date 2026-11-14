"""Compare a candidate experiment against a baseline (FR-S4, ADR 0005).

::

    from agentlens.compare import compare_experiments
    from agentlens.store import SQLiteStore

    with SQLiteStore() as store:
        result = compare_experiments(store, baseline="main", candidate="pr-123")
    for m in result.metrics:
        print(m.metric, f"{m.delta:+.3f}", m.ci, m.status)
    if result.regressions:
        raise SystemExit(1)

**Pairing.** Only cases present in both experiments are compared; the others
are listed in :attr:`Comparison.added_cases` / :attr:`Comparison.removed_cases`.
For each metric the per-case differences ``candidate - baseline`` are
bootstrapped over the cases that have the metric on both sides.

**Verdict.** A metric *regressed* only when its whole confidence interval is on
the bad side of zero (below for ``pass_rate``/scores, above for cost, latency,
steps); *improved* when it is all on the good side; otherwise *unchanged*:
the difference is within the noise. With fewer than ``min_cases`` paired cases
the verdict is *insufficient_data* (the interval is still reported).

**Per case** (:class:`CaseDelta`) only the quality metrics (``pass_rate`` and
scores) decide the status, since cost and latency of a single case are too noisy
to call; their deltas are still reported.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass, field
from enum import StrEnum

from agentlens.compare.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ConfidenceInterval,
    bootstrap_mean_ci,
)
from agentlens.compare.metrics import (
    COST,
    LATENCY,
    PASS_RATE,
    SCORE_PREFIX,
    STEPS,
    CaseMetrics,
    Direction,
    ExperimentMetrics,
    collect_metrics,
    metric_direction,
)
from agentlens.schema import Experiment
from agentlens.store.base import Store

__all__ = [
    "DEFAULT_MIN_CASES",
    "TOLERANCE",
    "CaseDelta",
    "CaseStatus",
    "Comparison",
    "MetricDelta",
    "MetricStatus",
    "compare",
    "compare_experiments",
    "metric_order",
]

DEFAULT_MIN_CASES = 2
"""Below this many paired cases a metric gets no verdict. ADR 0005 recommends
~20 cases for a meaningful interval; that is advice, not a hard limit."""

TOLERANCE = 1e-12
"""Differences smaller than this count as zero (floating-point noise)."""


class MetricStatus(StrEnum):
    """Verdict on an aggregate metric."""

    REGRESSED = "regressed"
    IMPROVED = "improved"
    UNCHANGED = "unchanged"
    INSUFFICIENT_DATA = "insufficient_data"


class CaseStatus(StrEnum):
    """Verdict on one case (quality metrics only)."""

    REGRESSED = "regressed"
    IMPROVED = "improved"
    MIXED = "mixed"
    """Some quality metric got better and another worse."""
    UNCHANGED = "unchanged"


def metric_order(metric: str) -> tuple[int, str]:
    """Sort key: ``pass_rate``, scores (by name), cost, latency, steps."""
    fixed = {PASS_RATE: 0, COST: 2, LATENCY: 3, STEPS: 4}
    return (fixed.get(metric, 1 if metric.startswith(SCORE_PREFIX) else 5), metric)


@dataclass(frozen=True)
class MetricDelta:
    """Aggregate change of one metric over the paired cases."""

    metric: str
    direction: Direction
    n: int
    """Paired cases that have the metric on both sides."""
    baseline: float
    """Mean over those cases in the baseline."""
    candidate: float
    """Mean over those cases in the candidate."""
    delta: float
    """``candidate - baseline`` (= mean of the per-case differences)."""
    ci: ConfidenceInterval
    """Bootstrap interval of ``delta``."""
    status: MetricStatus

    @property
    def delta_pct(self) -> float | None:
        """``delta`` relative to the baseline, in percent (``None`` if the baseline is 0)."""
        if abs(self.baseline) <= TOLERANCE:
            return None
        return self.delta / abs(self.baseline) * 100.0

    @property
    def regressed(self) -> bool:
        """The whole interval is on the bad side of zero."""
        return self.status is MetricStatus.REGRESSED

    @property
    def improved(self) -> bool:
        """The whole interval is on the good side of zero."""
        return self.status is MetricStatus.IMPROVED


@dataclass(frozen=True)
class CaseDelta:
    """One case present in both experiments."""

    case_id: str
    baseline: CaseMetrics
    candidate: CaseMetrics
    deltas: dict[str, float]
    """``candidate - baseline`` per metric both sides have, in :func:`metric_order`."""
    status: CaseStatus

    @property
    def pass_rate_delta(self) -> float:
        """Change of the fraction of passing runs."""
        return self.deltas[PASS_RATE]


@dataclass(frozen=True)
class Comparison:
    """Everything :func:`compare` found; reporters (#26-#28) and gates (#23) read this."""

    baseline: Experiment
    candidate: Experiment
    metrics: list[MetricDelta]
    """Aggregate deltas, in :func:`metric_order`."""
    cases: list[CaseDelta]
    """Paired cases, in the baseline's order."""
    added_cases: list[str] = field(default_factory=list)
    """Only in the candidate (not compared)."""
    removed_cases: list[str] = field(default_factory=list)
    """Only in the baseline (not compared)."""
    resamples: int = DEFAULT_RESAMPLES
    confidence: float = DEFAULT_CONFIDENCE
    seed: int = DEFAULT_SEED

    def __iter__(self) -> Iterator[MetricDelta]:
        return iter(self.metrics)

    def metric(self, name: str) -> MetricDelta:
        """The aggregate delta of metric ``name`` (``KeyError`` if not compared)."""
        for delta in self.metrics:
            if delta.metric == name:
                return delta
        raise KeyError(name)

    def case(self, case_id: str) -> CaseDelta:
        """The delta of case ``case_id`` (``KeyError`` if not paired)."""
        for delta in self.cases:
            if delta.case_id == case_id:
                return delta
        raise KeyError(case_id)

    @property
    def regressions(self) -> list[MetricDelta]:
        """Metrics whose interval is entirely on the bad side."""
        return [m for m in self.metrics if m.regressed]

    @property
    def improvements(self) -> list[MetricDelta]:
        """Metrics whose interval is entirely on the good side."""
        return [m for m in self.metrics if m.improved]

    @property
    def regressed_cases(self) -> list[CaseDelta]:
        """Cases whose quality metrics only got worse."""
        return [c for c in self.cases if c.status is CaseStatus.REGRESSED]

    @property
    def improved_cases(self) -> list[CaseDelta]:
        """Cases whose quality metrics only got better."""
        return [c for c in self.cases if c.status is CaseStatus.IMPROVED]


def _metric_status(
    direction: Direction, ci: ConfidenceInterval, n: int, min_cases: int
) -> MetricStatus:
    if n < min_cases:
        return MetricStatus.INSUFFICIENT_DATA
    worse = ci.high < -TOLERANCE if direction is Direction.HIGHER_IS_BETTER else ci.low > TOLERANCE
    better = ci.low > TOLERANCE if direction is Direction.HIGHER_IS_BETTER else ci.high < -TOLERANCE
    if worse:
        return MetricStatus.REGRESSED
    if better:
        return MetricStatus.IMPROVED
    return MetricStatus.UNCHANGED


def _case_status(deltas: dict[str, float]) -> CaseStatus:
    worse = better = False
    for metric, delta in deltas.items():
        if metric != PASS_RATE and not metric.startswith(SCORE_PREFIX):
            continue
        worse |= delta < -TOLERANCE
        better |= delta > TOLERANCE
    if worse and better:
        return CaseStatus.MIXED
    if worse:
        return CaseStatus.REGRESSED
    if better:
        return CaseStatus.IMPROVED
    return CaseStatus.UNCHANGED


def compare(
    baseline: ExperimentMetrics,
    candidate: ExperimentMetrics,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    min_cases: int = DEFAULT_MIN_CASES,
) -> Comparison:
    """Per-case and aggregate deltas with paired-bootstrap confidence intervals.

    Every metric is bootstrapped with the same ``seed``, so metrics over the same
    cases are resampled identically and the result is reproducible.

    Args:
        baseline: Metrics of the reference experiment.
        candidate: Metrics of the experiment under test.
        resamples: Bootstrap resamples per metric (ADR 0005: 10 000).
        confidence: Interval coverage, in ``(0, 1)``.
        seed: Random seed of the bootstrap.
        min_cases: Paired cases a metric needs to get a verdict (>= 1).

    Raises:
        ValueError: on invalid ``resamples``/``confidence``/``min_cases``.
    """
    if min_cases < 1:
        raise ValueError(f"min_cases must be >= 1, got {min_cases}")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    paired = [cid for cid in baseline.cases if cid in candidate.cases]
    added = [cid for cid in candidate.cases if cid not in baseline.cases]
    removed = [cid for cid in baseline.cases if cid not in candidate.cases]

    cases: list[CaseDelta] = []
    for cid in paired:
        base, cand = baseline.cases[cid], candidate.cases[cid]
        shared = sorted(base.values.keys() & cand.values.keys(), key=metric_order)
        deltas = {m: cand.values[m] - base.values[m] for m in shared}
        cases.append(CaseDelta(cid, base, cand, deltas, _case_status(deltas)))

    names = sorted({m for case in cases for m in case.deltas}, key=metric_order)
    metrics: list[MetricDelta] = []
    for name in names:
        having = [c for c in cases if name in c.deltas]
        base_values = [c.baseline.values[name] for c in having]
        cand_values = [c.candidate.values[name] for c in having]
        diffs = [c.deltas[name] for c in having]
        n = len(having)
        ci = bootstrap_mean_ci(diffs, resamples=resamples, confidence=confidence, seed=seed)
        direction = metric_direction(name)
        metrics.append(
            MetricDelta(
                metric=name,
                direction=direction,
                n=n,
                baseline=sum(base_values) / n,
                candidate=sum(cand_values) / n,
                delta=sum(diffs) / n,
                ci=ci,
                status=_metric_status(direction, ci, n, min_cases),
            )
        )

    return Comparison(
        baseline=baseline.experiment,
        candidate=candidate.experiment,
        metrics=metrics,
        cases=cases,
        added_cases=added,
        removed_cases=removed,
        resamples=resamples,
        confidence=confidence,
        seed=seed,
    )


def compare_experiments(
    store: Store,
    baseline: str | Experiment,
    candidate: str | Experiment,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
    min_cases: int = DEFAULT_MIN_CASES,
) -> Comparison:
    """Load two stored experiments (id, name or object) and :func:`compare` them.

    Raises:
        agentlens.suite.ExperimentNotFoundError: if either is not stored.
    """
    return compare(
        collect_metrics(store, baseline),
        collect_metrics(store, candidate),
        resamples=resamples,
        confidence=confidence,
        seed=seed,
        min_cases=min_cases,
    )
