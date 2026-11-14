# ADR 0006: CI baseline: re-run the suite on the PR's base commit

- Status: Accepted
- Date: 2026-11-14

## Context
The GitHub Action (FR-C3) compares a pull request against a baseline. A PR job starts from a
fresh runner: there is no database with the baseline experiment in it. Options:

1. **Re-run on the base commit in the same job**: check out the PR's base sha in a temporary
   `git worktree`, run the suite there, then on the PR, then compare.
2. **Stored baseline**: run with `--set-baseline` on pushes to `main`, keep the SQLite database
   with `actions/cache` or as an artifact, restore it in PR jobs.
3. **Committed baseline**: export the baseline's JSONL into the repository.

## Decision
Default to (1): `baseline-ref` defaults to `github.event.pull_request.base.sha`. Support (2) with
`baseline-ref: ""` plus `baseline` (an experiment id/name or the suite's marked baseline) and the
`db` input. Do not do (3): it puts generated data in every PR's diff.

## Consequences
- Always comparable: same runner, same AgentLens version, same suite and gates; nothing to keep
  between jobs, nothing stale, works on the first PR.
- The suite runs twice per job, so with a real model a PR pays for both runs (and twice the
  time). Replay cassettes, a cheap model or option (2) reduce that; the bootstrap CI (ADR 0005)
  already treats both runs as noisy samples.
- The baseline runs the base commit's *agent code* with the PR's installed dependencies; a PR
  that changes dependencies is compared under the new ones.
- A suite that does not exist at the base commit has no baseline: only absolute gates are
  checked (noted in the report), so adding a suite never fails its own PR on delta gates.
