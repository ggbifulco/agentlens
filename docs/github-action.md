# GitHub Action

`action.yml` at the root of this repository is a composite action that checks every pull request
for regressions: it runs your suite on the PR's **base commit** (the baseline) and on the PR
(the candidate) in the same job, compares them with confidence intervals, posts the Markdown
report as a PR comment, and fails the job when a gate fails.

```yaml
# .github/workflows/agentlens.yml
name: AgentLens
on: pull_request
permissions:
  contents: read
  pull-requests: write          # to post the comment
jobs:
  agentlens:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: ggbifulco/agentlens@main      # pin a tag or sha
        with:
          suite: suites/support.yaml
          gates: gates.yaml
          install: "-e ."                    # your agent's own dependencies
```

The comment starts with a hidden `<!-- agentlens-report -->` marker, so each push **edits the
same comment** instead of adding one (`marker:` gives each suite its own when a PR runs
several). It reads like this (case changes are collapsed):

> ### ❌ AgentLens: gates failed · 1 case regressed
> **pr-7** vs **baseline-b11d197d1567** · suite `ci-demo` · 12 paired cases
>
> | Metric | Baseline | Candidate | Δ | Δ% | 95% CI | Status |
> | --- | ---: | ---: | ---: | ---: | ---: | --- |
> | pass rate ↑ | 1 | 0.9167 | -0.08333 | -8.3% | [-0.25, +0] | unchanged |
> | steps ↓ | 1 | 1 | +0 | +0.0% | [+0, +0] | unchanged |
>
> #### Gates: ❌ failed (2 passed, 2 failed, 0 skipped)
> | | Gate | Rule | Actual | Reason |
> | --- | --- | --- | ---: | --- |
> | ❌ **fail** | case_regressions | max 0 | 1 | case_regressions 1 is above max 0 |

| Input | Default | |
|---|---|---|
| `suite` | (required) | Suite YAML, relative to `working-directory`. |
| `gates` | none | Gates YAML; without it the job never fails on results. |
| `baseline-ref` | the PR's base commit | Code run as the baseline (from a temporary `git worktree`). Empty: compare with `baseline` from the database. |
| `baseline` | `baseline` | With `baseline-ref: ""`: experiment id/name, or the suite's marked baseline. |
| `candidate-name` | `pr-<number>` | Name of the candidate experiment. |
| `db` | `.agentlens/agentlens.db` | Database (keep it with `actions/cache` to reuse baselines). |
| `install` | none | Extra `uv pip install` arguments for your agent (`-e .`, `-r requirements.txt`). |
| `python-version`, `agentlens-version`, `working-directory` | `3.12`, the action's own source, `.` | |
| `comment`, `github-token`, `marker` | `true`, `github.token`, `agentlens-report` | |

Outputs: `passed`, `exit-code`, `report-path` (Markdown), `json-path`, `baseline-experiment`,
`candidate-experiment`. The report is also written to the run's summary page, so pull requests
from forks (whose token cannot comment) still show it.

**Where the baseline comes from.** By default the action re-runs the suite on the base commit:
always comparable (same runner, same AgentLens, same suite), no state to keep, but the job runs
the suite twice, so with a real model it pays for both runs (use replay cassettes or a cheaper
model to keep that down). If a suite is new (missing at the base commit) only the absolute gates
are checked. To compare with a stored baseline instead, run `agentlens run SUITE --set-baseline`
on pushes to `main`, keep the database with `actions/cache` (or upload it as an artifact), restore
it in the PR job and set `baseline-ref: ""`: one run per PR, but the baseline can be stale or
come from a different environment. This repository runs the action on a small deterministic
demo suite ([`examples/ci_demo`](https://github.com/ggbifulco/agentlens/tree/main/examples/ci_demo)) in
[`.github/workflows/agentlens.yml`](https://github.com/ggbifulco/agentlens/blob/main/.github/workflows/agentlens.yml).


Why re-run the baseline instead of storing it: [ADR 0006](adr/0006-ci-baseline.md).
