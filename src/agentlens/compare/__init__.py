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

Gates (FR-S5) turn a comparison, or a single run, into a pass/fail verdict
and a process exit code::

    from agentlens.compare import evaluate_gates, load_gates

    report = evaluate_gates(load_gates("gates.yaml"), comparison=comparison)
    print(report.summary())
    raise SystemExit(report.exit_code)   # 0 pass, 1 a gate failed
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
    run_metrics,
)
from agentlens.compare.gates import (
    CASE_REGRESSIONS,
    COUNT_GATES,
    DEFAULT_MIN_CASES,
    EXIT_GATE_FAILED,
    EXIT_OK,
    METRIC_ALIASES,
    REGRESSIONS,
    GateReport,
    GateResult,
    GateRule,
    Gates,
    GatesError,
    OnMissing,
    evaluate_gates,
    load_gates,
    resolve_metric,
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
    "CASE_REGRESSIONS",
    "COUNT_GATES",
    "DEFAULT_CONFIDENCE",
    "DEFAULT_MIN_CASES",
    "DEFAULT_RESAMPLES",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "LOWER_IS_BETTER",
    "METRIC_ALIASES",
    "PASS_RATE",
    "REGRESSIONS",
    "SCORE_PREFIX",
    "CaseComparison",
    "CaseMetric",
    "Comparison",
    "Direction",
    "GateReport",
    "GateResult",
    "GateRule",
    "Gates",
    "GatesError",
    "MetricDelta",
    "OnMissing",
    "Status",
    "classify",
    "compare_experiments",
    "compare_runs",
    "evaluate_gates",
    "experiment_runs",
    "load_gates",
    "metric_direction",
    "paired_bootstrap",
    "percentile",
    "resolve_metric",
    "run_metrics",
]
