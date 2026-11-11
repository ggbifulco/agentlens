# ADR 0003: SQLite default store, JSONL interchange

- Status: Accepted
- Date: 2026-11-11

## Context
Traces and experiments must persist locally, offline, with zero setup, and be shareable/committable.

## Decision
SQLite (stdlib `sqlite3`) behind a `Store` protocol; JSON columns for span payloads. JSONL export/import for fixtures, cassettes and sharing.

## Consequences
Zero infra, good enough for tens of thousands of traces. Not for concurrent multi-writer production use (out of scope v1).
