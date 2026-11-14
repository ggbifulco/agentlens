# Comparison & gates

`compare_experiments` pairs two experiments by case id and reports, per case and in aggregate,
the change (candidate − baseline) in pass rate, mean score per evaluator, cost, latency and steps.
Each aggregate delta gets a **paired bootstrap 95% confidence interval** over cases (10 000
resamples, seeded, so the same data always gives the same answer); a metric is a regression only
when its interval lies entirely on the bad side of 0, so one flaky case does not fail the build:

```python
from agentlens.compare import compare_experiments
from agentlens.store import SQLiteStore
from agentlens.suite import Suite, run_suite

def agent(question: str) -> str:
    return question.upper()                 # the baseline behaviour

def new_agent(question: str) -> str:
    return question if "refund" in question else question.upper()   # a "prompt change"

cases = [{"id": f"q{i}", "input": f"refund {i}" if i % 3 == 0 else f"hello {i}",
          "expected": (f"refund {i}" if i % 3 == 0 else f"hello {i}").upper()}
         for i in range(30)]
defaults = {"evaluators": ["no_error", "exact_match"]}

with SQLiteStore("runs.db") as store:
    run_suite(Suite(name="demo", agent=agent, cases=cases, defaults=defaults),
              store=store, name="main")
    run_suite(Suite(name="demo", agent=new_agent, cases=cases, defaults=defaults),
              store=store, name="pr-123")
    comparison = compare_experiments(store, "main", "pr-123")   # 10k resamples, seed 0

pass_rate = comparison.metric("pass_rate")
print(f"{pass_rate.delta:+.2f} [{pass_rate.ci_low:+.2f}, {pass_rate.ci_high:+.2f}]",
      pass_rate.status)
print([m.name for m in comparison.regressions])
print([c.case_id for c in comparison.case_regressions][:3],
      comparison.case_regressions[0].regressed_evaluators)
# -0.33 [-0.50, -0.17] regression
# ['pass_rate', 'score.exact_match']
# ['q0', 'q3', 'q6'] ['exact_match']
```

Cases only in one experiment are listed in `added_cases`/`removed_cases`; `compare_runs` compares
two in-memory `run_suite` results, `paired_bootstrap(diffs)` is the statistic on its own, and
`comparison.model_dump_json()` is what gates and reports consume. The interval needs about 20+
cases to be meaningful (ADR 0005).

## Gates
A gates file sets the thresholds a candidate must meet (FR-S5). Per metric (`pass_rate` or its
alias `success_rate`, `score.<evaluator>`, `cost_usd`/`cost`, `latency_ms`/`latency`, `steps`):
`min`/`max` bound the candidate's value, `min_delta`/`max_delta` the change vs the baseline and
`min_delta_pct`/`max_delta_pct` that change in percent. `regressions` counts metrics whose
confidence interval is entirely on the bad side, `case_regressions` cases whose pass rate dropped.
`evaluate_gates` returns a `GateReport` whose `exit_code` is what CI should exit with:

```python
from pathlib import Path

from agentlens.compare import compare_runs, evaluate_gates, load_gates
from agentlens.suite import Suite, run_suite

Path("gates.yaml").write_text("""
success_rate: {min: 0.9}          # alias of pass_rate: candidate value
cost_usd: {max_delta_pct: 20, on_missing: skip}   # +20% at most; skip if no costs
regressions: {max: 0}             # no metric whose CI is entirely on the bad side
""")

def agent(question: str) -> str:
    return question.upper()

def new_agent(question: str) -> str:
    return question if "refund" in question else question.upper()

cases = [{"id": f"q{i}", "input": f"refund {i}" if i % 3 == 0 else f"hello {i}",
          "expected": (f"refund {i}" if i % 3 == 0 else f"hello {i}").upper()}
         for i in range(30)]
defaults = {"evaluators": ["no_error", "exact_match"]}
main = run_suite(Suite(name="demo", agent=agent, cases=cases, defaults=defaults))
pr = run_suite(Suite(name="demo", agent=new_agent, cases=cases, defaults=defaults))

report = evaluate_gates(load_gates("gates.yaml"), comparison=compare_runs(main, pr))
print(report.summary())
print(report.exit_code)
# gates FAILED: 0 passed, 2 failed, 1 skipped

first = evaluate_gates(load_gates("gates.yaml"), run=main)   # no baseline yet: absolute rules only
print(first.passed, first.notes)
# gates FAILED: 0 passed, 2 failed, 1 skipped
#   FAIL success_rate min 0.9: pass_rate 0.6667 is below min 0.9
#   SKIP cost_usd max_delta_pct +20%: no case has a 'cost_usd' value in both experiments
#   FAIL regressions max 0: regressions 2 is above max 0
# 1
# True ['no baseline comparison: 2 delta/regression checks skipped']
```

A gate whose metric has no data fails with the reason (`on_missing: skip` per gate, or
`options: {on_missing: skip}` for all, skips it instead). Without a comparison (`run=` only, e.g. a
first run with no baseline yet) delta and regression gates are skipped and noted. Fewer than
`options.min_cases` (default 20) cases adds a note, never a failure. A broken file raises
`GatesError` with every problem at its `gates.yaml:line:column`.


See also: [ADR 0005](../adr/0005-bootstrap-regression.md),
[design §7](../design.md#7-comparison-gates), API: [`agentlens.compare`](../api/compare.md).
