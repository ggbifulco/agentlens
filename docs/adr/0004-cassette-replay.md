# ADR 0004: Cassette-style replay

- Status: Accepted
- Date: 2026-11-11

## Context
Agent runs are non-deterministic; to test code changes in isolation we need deterministic re-execution.

## Decision
Record each LLM/tool call keyed by (call index, sha256 of normalised request). Playback modes: strict, lenient, fallthrough. Report first divergence.

## Consequences
Same idea as VCR.py, proven pattern. Hash normalisation must ignore volatile fields (ids, timestamps).
