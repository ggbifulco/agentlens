# ADR 0005: Paired bootstrap for regression detection

- Status: Accepted
- Date: 2026-11-11

## Context
LLM outputs are noisy: a 2-point drop on 30 cases may be noise. Gates must not flap.

## Decision
Run each case N times; compare baseline vs candidate with paired bootstrap over cases (10k resamples), 95% CI. A metric regresses only if the CI is entirely on the bad side. Gates can also use absolute thresholds.

## Consequences
Statistically defensible, explainable in a PR comment. Needs >= ~20 cases to be meaningful; documented.
