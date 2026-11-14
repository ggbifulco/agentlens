"""The result of comparing two experiments: :class:`Comparison` and its parts.

Every model is a pydantic model, so a comparison round-trips through JSON
(``comparison.model_dump_json()`` / ``Comparison.model_validate_json(...)``)
for gates, reporters and CI artifacts.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import Field

from agentlens.schema import Experiment
from agentlens.schema._base import SchemaModel

__all__ = [
    "CaseComparison",
    "CaseMetric",
    "Comparison",
    "Direction",
    "MetricDelta",
    "Status",
]


class Direction(StrEnum):
    """Which way a metric should move."""

    HIGHER_IS_BETTER = "higher_is_better"
    """Pass rate and evaluator scores."""
    LOWER_IS_BETTER = "lower_is_better"
    """Cost, latency and steps."""


class Status(StrEnum):
    """Verdict on one metric (aggregate) or one case."""

    REGRESSION = "regression"
    IMPROVEMENT = "improvement"
    UNCHANGED = "unchanged"
    NO_DATA = "no_data"
    """No case has a value in both experiments (e.g. no costs recorded)."""


class MetricDelta(SchemaModel):
    """Aggregate change of one metric over the cases present in both experiments.

    ``baseline``/``candidate`` are means over the *paired* cases (each case
    first averaged over its repetitions), so ``delta == candidate - baseline``
    is the mean paired difference whose bootstrap CI is ``[ci_low, ci_high]``.
    """

    name: str
    """``pass_rate``, ``score.<evaluator>``, ``cost_usd``, ``latency_ms`` or ``steps``."""
    direction: Direction
    evaluator: str | None = None
    """The evaluator of a ``score.<evaluator>`` metric."""
    n: int = Field(ge=0)
    """Number of paired cases the delta is computed on."""
    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    """``candidate - baseline``."""
    delta_pct: float | None = None
    """``delta / |baseline| * 100`` (``None`` when the baseline is 0)."""
    ci_low: float | None = None
    ci_high: float | None = None
    status: Status

    @property
    def is_regression(self) -> bool:
        """The CI lies entirely on the bad side of 0."""
        return self.status is Status.REGRESSION

    @property
    def is_improvement(self) -> bool:
        """The CI lies entirely on the good side of 0."""
        return self.status is Status.IMPROVEMENT


class CaseMetric(SchemaModel):
    """One metric of one case, each side averaged over its repetitions."""

    baseline: float | None = None
    candidate: float | None = None
    delta: float | None = None
    """``candidate - baseline`` when both are known."""


class CaseComparison(SchemaModel):
    """One case present in both experiments.

    ``status`` follows the case's pass rate: ``regression`` when it dropped
    (e.g. passed in the baseline, fails in the candidate), ``improvement``
    when it rose. With repetitions a single flaky run is enough to change
    it, so read it as "where to look"; whether the change is beyond noise
    is what the aggregate :class:`MetricDelta` CIs say.
    """

    case_id: str
    baseline_runs: int = Field(ge=0)
    candidate_runs: int = Field(ge=0)
    metrics: dict[str, CaseMetric] = Field(default_factory=dict)
    """Keyed by metric name (as in :attr:`MetricDelta.name`)."""
    status: Status
    regressed_evaluators: list[str] = Field(default_factory=list)
    """Evaluators whose pass rate on this case dropped (sorted)."""

    @property
    def pass_rate(self) -> CaseMetric:
        """The ``pass_rate`` metric of this case."""
        return self.metrics["pass_rate"]


class Comparison(SchemaModel):
    """What :func:`~agentlens.compare.compare_experiments` returns: candidate vs baseline."""

    baseline: Experiment | None = None
    candidate: Experiment | None = None
    metrics: list[MetricDelta] = Field(default_factory=list)
    """``pass_rate``, ``score.<evaluator>`` (by name), ``cost_usd``, ``latency_ms``, ``steps``."""
    cases: list[CaseComparison] = Field(default_factory=list)
    """Cases present in both experiments, in baseline order."""
    added_cases: list[str] = Field(default_factory=list)
    """Case ids only in the candidate (not part of any delta)."""
    removed_cases: list[str] = Field(default_factory=list)
    """Case ids only in the baseline (not part of any delta)."""
    resamples: int = Field(ge=1)
    confidence: float = Field(gt=0.0, lt=1.0)
    seed: int | None = None

    def metric(self, name: str) -> MetricDelta:
        """The aggregate metric called ``name`` (``KeyError`` if absent)."""
        for metric in self.metrics:
            if metric.name == name:
                return metric
        raise KeyError(name)

    def case(self, case_id: str) -> CaseComparison:
        """The comparison of case ``case_id`` (``KeyError`` if not in both experiments)."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)

    @property
    def regressions(self) -> list[MetricDelta]:
        """Metrics whose CI lies entirely on the bad side."""
        return [m for m in self.metrics if m.status is Status.REGRESSION]

    @property
    def improvements(self) -> list[MetricDelta]:
        """Metrics whose CI lies entirely on the good side."""
        return [m for m in self.metrics if m.status is Status.IMPROVEMENT]

    @property
    def case_regressions(self) -> list[CaseComparison]:
        """Cases whose pass rate dropped."""
        return [c for c in self.cases if c.status is Status.REGRESSION]

    @property
    def case_improvements(self) -> list[CaseComparison]:
        """Cases whose pass rate rose."""
        return [c for c in self.cases if c.status is Status.IMPROVEMENT]

    @property
    def has_regressions(self) -> bool:
        """Whether any aggregate metric regressed."""
        return bool(self.regressions)
