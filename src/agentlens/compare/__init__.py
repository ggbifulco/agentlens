"""Compare experiments: per-case and aggregate deltas with paired bootstrap CIs (FR-S4).

::

    from agentlens.compare import compare_experiments

    result = compare_experiments(store, baseline="main", candidate="pr-123")
    for m in result.metrics:            # pass_rate, score:<evaluator>, cost_usd, latency_ms, steps
        print(f"{m.metric}: {m.delta:+.3f} [{m.ci.low:+.3f}, {m.ci.high:+.3f}] {m.status}")
    print([c.case_id for c in result.regressed_cases])

A metric *regresses* only when its whole 95% interval is on the bad side of zero
(ADR 0005), so noise between two runs of the same agent does not flag anything.
"""

from agentlens.compare.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    DEFAULT_SEED,
    ConfidenceInterval,
    bootstrap_mean_ci,
    percentile,
)
from agentlens.compare.comparator import (
    DEFAULT_MIN_CASES,
    TOLERANCE,
    CaseDelta,
    CaseStatus,
    Comparison,
    MetricDelta,
    MetricStatus,
    compare,
    compare_experiments,
    metric_order,
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
    RunMetrics,
    collect_metrics,
    metric_direction,
    metrics_from_run,
    run_metrics,
    score_metric,
)

__all__ = [
    "COST",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MIN_CASES",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "LATENCY",
    "PASS_RATE",
    "SCORE_PREFIX",
    "STEPS",
    "TOLERANCE",
    "CaseDelta",
    "CaseMetrics",
    "CaseStatus",
    "Comparison",
    "ConfidenceInterval",
    "Direction",
    "ExperimentMetrics",
    "MetricDelta",
    "MetricStatus",
    "RunMetrics",
    "bootstrap_mean_ci",
    "collect_metrics",
    "compare",
    "compare_experiments",
    "metric_direction",
    "metric_order",
    "metrics_from_run",
    "percentile",
    "run_metrics",
    "score_metric",
]
