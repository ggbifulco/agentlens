"""Compare a candidate experiment against a baseline (``docs/design.md`` §7, ADR 0005).

Per case and in aggregate: pass rate, mean score per evaluator, cost,
latency and steps, each case averaged over its repetitions, with a paired
bootstrap confidence interval (10 000 resamples, seeded) on every aggregate
delta. A metric is a regression only when its CI lies entirely on the bad
side of 0, so noise does not make a gate flap::

    from agentlens.compare import compare_experiments
    from agentlens.store import SQLiteStore

    with SQLiteStore("runs.db") as store:
        comparison = compare_experiments(store, "main", "pr-123")
    print([m.name for m in comparison.regressions])
    print([c.case_id for c in comparison.case_regressions])
    print(comparison.model_dump_json(indent=2))

:func:`compare_runs` does the same on in-memory results (two
:class:`~agentlens.suite.ExperimentRun`), and :func:`paired_bootstrap` is the
statistic on its own.
"""

from agentlens.compare.bootstrap import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    paired_bootstrap,
    percentile,
)
from agentlens.compare.comparator import (
    LOWER_IS_BETTER,
    PASS_RATE,
    SCORE_PREFIX,
    classify,
    compare_experiments,
    compare_runs,
    experiment_runs,
    metric_direction,
)
from agentlens.compare.model import (
    CaseComparison,
    CaseMetric,
    Comparison,
    Direction,
    MetricDelta,
    Status,
)

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "LOWER_IS_BETTER",
    "PASS_RATE",
    "SCORE_PREFIX",
    "CaseComparison",
    "CaseMetric",
    "Comparison",
    "Direction",
    "MetricDelta",
    "Status",
    "classify",
    "compare_experiments",
    "compare_runs",
    "experiment_runs",
    "metric_direction",
    "paired_bootstrap",
    "percentile",
]
