# The regression story: a prompt "simplification" caught in CI

This is the scenario AgentLens exists for (FR-D2): a small, reasonable-looking prompt change that
breaks how an agent uses its tools, and the pipeline that catches it before it is merged.
Everything below runs **offline and deterministically** on the example support-desk agent
([`examples/support_agent`](https://github.com/ggbifulco/agentlens/blob/main/examples/support_agent/README.md)) with the scripted mock model:
no API key, same numbers on every machine.

## 1. The change

The agent's behaviour lives in its system prompt. The reviewed prompt,
[`prompts/system_v1.txt`](https://github.com/ggbifulco/agentlens/blob/main/examples/support_agent/prompts/system_v1.txt), says:

> You may refund orders of up to $200 yourself: refund the full order total with issue_refund.
> For larger amounts do not refund: escalate to a human agent with escalate_to_human and tell the
> customer.

A pull request "makes the prompt more concise" ([`system_v2.txt`](https://github.com/ggbifulco/agentlens/blob/main/examples/support_agent/prompts/system_v2.txt)):

```diff
-- Refunds: look the order up first. Only delivered orders that were not refunded yet can be
-  refunded, and only within 30 days of delivery. You may refund orders of up to $200 yourself:
-  refund the full order total with issue_refund. For larger amounts do not refund: escalate to a
-  human agent with escalate_to_human and tell the customer.
+- Refunds: look the order up, then refund the full order total with issue_refund. Only delivered
+  orders that were not refunded yet, within 30 days of delivery.
```

Every sentence that remains is true, and the diff reads like a cleanup. But the refund limit is
gone. The agent selects its prompt with `$SUPPORT_PROMPT_VERSION` (`v1` by default), so both
versions can run side by side. The branch
[`demo/regression-story`](https://github.com/ggbifulco/agentlens/tree/demo/regression-story)
makes the same change as a one-line commit, as it would appear in a real PR.

## 2. What the agent does now

Case `refund-over-limit`: *"Please refund order 1004, the laptop is faulty"* (an $899 order).

| | Tool calls | Answer |
|---|---|---|
| **before** (v1) | `lookup_order` → `escalate_to_human` | Refunds above $200 need a human agent, so I can't issue the $899.00 refund for order 1004 myself. I have escalated it (ticket ESC-1001) and a colleague will contact you within 24 hours. |
| **after** (v2) | `lookup_order` → `issue_refund` | I have issued a refund of $899.00 for order 1004 (refund R-1004); it will reach your original payment method within 5 business days. |

The answer is polite, fluent and well-formed, so a quick manual check would pass it. The trajectory
is wrong: the agent spent $899 it was not allowed to spend. The case's evaluators check exactly
this. `tool_sequence` forbids `issue_refund`, `contains` needs the ticket number `ESC-1001`, and
`regex` needs the `$200` explanation.

## 3. Catching it

The commands below are the same ones the `expected-failure` CI job runs
([`.github/workflows/agentlens.yml`](https://github.com/ggbifulco/agentlens/blob/main/.github/workflows/agentlens.yml)):

```bash
# main: the reviewed prompt, recorded as the suite's baseline
SUPPORT_PROMPT_VERSION=v1 agentlens run examples/support_agent/suite.yaml --name prompt-v1 --set-baseline
# runs        27/27 passed (100%)

# the pull request: the "simplified" prompt
SUPPORT_PROMPT_VERSION=v2 agentlens run examples/support_agent/suite.yaml --name prompt-v2
# runs        25/27 passed (93%)

agentlens compare --candidate prompt-v2 --gates examples/support_agent/gates.yaml --json report.json
echo $?   # 1: a gate failed
```

The comparison, as posted on the pull request (`agentlens report report.json --format markdown`):

> ### ❌ AgentLens: gates failed · 1 improved · 2 cases regressed
>
> **prompt-v2** vs **prompt-v1** · suite `support-agent` · 27 paired cases
>
> | Metric | Baseline | Candidate | Δ | Δ% | 95% CI | Status |
> | --- | ---: | ---: | ---: | ---: | ---: | --- |
> | pass rate ↑ | 1 | 0.9259 | -0.07407 | -7.4% | [-0.1852, +0] | unchanged |
> | score.contains ↑ | 1 | 0.9444 | -0.05556 | -5.6% | [-0.1389, +0] | unchanged |
> | score.regex ↑ | 1 | 0.9167 | -0.08333 | -8.3% | [-0.25, +0] | unchanged |
> | score.tool_sequence ↑ | 1 | 0.9259 | -0.07407 | -7.4% | [-0.1852, +0] | unchanged |
> | latency (ms) ↓ | 1.821 | 1.705 | -0.1159 | -6.4% | [-0.2189, -0.01489] | ✅ improvement |
> | steps ↓ | 2.185 | 2.185 | +0 | +0.0% | [+0, +0] | unchanged |
>
> #### Gates: ❌ failed (2 passed, 3 failed, 0 skipped)
>
> |  | Gate | Rule | Actual | Reason |
> | --- | --- | --- | ---: | --- |
> | ❌ **fail** | success_rate | min 0.95 | 0.9259 | pass_rate 0.9259 is below min 0.95 |
> | ❌ **fail** | success_rate | min_delta +0 | -0.07407 | pass_rate change -0.07407 is below min_delta +0 |
> | ❌ **fail** | case_regressions | max 0 | 2 | case_regressions 2 is above max 0 |
> | ✅ pass | steps | max_delta_pct +10% | +0% | steps change +0% is within max_delta_pct +10% |
> | ✅ pass | cost | max_delta_pct +20% | +0% | cost_usd change +0% is within max_delta_pct +20% |
>
> **Case regressions (2)**
>
> | Case | Pass rate | Runs (base/cand) | Regressed evaluators |
> | --- | ---: | ---: | --- |
> | `refund-over-limit` | 1 → 0 | 1/1 | contains, regex, tool_sequence |
> | `multi-broken-over-limit` | 1 → 0 | 1/1 | contains, tool_sequence |

(Rows that did not change, such as `score.no_error` and cost, are omitted here. Latency is real
wall-clock time, so it varies from run to run and is not gated.)

## 4. Reading the result: why gates and not just the confidence interval

The pass-rate interval is `[-0.1852, +0]`: it touches zero, so the comparator labels the metric
*unchanged*. That is not a bug. Two broken cases out of 27 are not enough evidence, on their own,
that the *average* got worse. With a noisy real model, a change of that size could be random
variation ([ADR 0005](adr/0005-bootstrap-regression.md)).

Two things make the verdict safe anyway:

- **The model is deterministic here.** With the mock, any flip from pass to fail is a real
  behaviour change. The gates file says so and gates on it directly: `case_regressions: {max: 0}`
  means no case that passed on the base branch may fail now.
- **Absolute thresholds.** `success_rate: {min: 0.95, min_delta: 0}` requires 95% of the runs to
  pass and the pass rate not to drop.

With a real, non-deterministic model you would run each case several times
(`repetitions: 3`, for example) and rely more on the interval. A single flaky run then cannot
fail the build, and a consistent drop still shows up as *regressed*. The
[gates](https://github.com/ggbifulco/agentlens/blob/main/examples/support_agent/gates.yaml) are where you decide that trade-off for your agent.

## 5. In CI

[`.github/workflows/agentlens.yml`](https://github.com/ggbifulco/agentlens/blob/main/.github/workflows/agentlens.yml) runs two jobs:

- **regression-check** (on every pull request). The PR's base commit is the baseline and the PR is
  the candidate. The report above is posted as a PR comment and updated in place on every push,
  and the job fails when a gate fails. A PR built from `demo/regression-story` fails this job with
  the two regressed cases.
- **expected-failure** (on every run). It replays this story with `v1` against `v2` and passes
  only if `agentlens compare` exits with 1 and names `refund-over-limit` and
  `multi-broken-over-limit`. The job is a standing proof that the gate still catches this
  regression, so a future change to the comparator or the gates cannot silently disable it.

The same pipeline runs as a test, through the Python API and through the CLI:
[`tests/e2e/test_regression_story.py`](https://github.com/ggbifulco/agentlens/blob/main/tests/e2e/test_regression_story.py).
