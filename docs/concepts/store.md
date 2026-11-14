# Store

Traces, experiments and evaluation results are kept in a **SQLite database**: one file, no
server, safe to cache in CI (ADR 0003). The CLI uses `.agentlens/agentlens.db` unless you pass
`--db PATH` or set `$AGENTLENS_DB`. **JSONL** is the interchange format: one trace per line, easy
to `grep`, `jq`, diff and commit as a fixture.

```python
import agentlens
from agentlens.store import SQLiteStore, export_jsonl, import_jsonl

with SQLiteStore("runs.db") as store:
    for question in ["Where is order 1234?", "Refund order 1234"]:
        with agentlens.run("support-agent", input=question, sink=store.save_trace) as trace:
            trace.output = f"echo: {question}"

    recent = store.list_traces(name="support-agent", limit=10)      # most recent first
    print(len(recent), recent[0].input)        # 2 Refund order 1234
    print(store.get_trace(recent[0].id).output)                   # echo: Refund order 1234

    print(export_jsonl(store, "support.jsonl", name="support-agent"))   # 2 (oldest first)

with SQLiteStore("copy.db") as other:
    print(import_jsonl(other, "support.jsonl"))                     # 2 (idempotent upserts)
```

What the store keeps:

| Object | Saved by | Read with |
|---|---|---|
| `Trace` (spans, usage, metadata) | `save_trace` (a run's `sink`), `run_suite` | `get_trace`, `list_traces(name=, experiment_id=, metadata=, has_error=, limit=, offset=)` |
| `Experiment` (one suite run: config, git sha, suite hash) | `run_suite`, `agentlens run` | `get_experiment`, `list_experiments(name=, suite=, is_baseline=)` |
| `EvalResult` per trace | `run_suite` | `get_eval_results`, `list_eval_results(experiment_id)` |

- **Baselines**: one experiment per suite can be marked as the baseline
  (`agentlens run --set-baseline`, `agentlens baseline set`, `agentlens.suite.set_baseline`);
  `agentlens compare` compares against it by default.
- **Metadata filters** compare JSON values exactly: `list_traces(metadata={"case_id": "refund"})`.
- **Migrations** are applied automatically; a database written by a newer AgentLens raises
  `SchemaVersionError` instead of being damaged.
- **Safe writes**: each write is one transaction, a JSONL export goes to a temporary file first,
  and an import validates the whole file before saving anything (`JSONLFormatError` names the
  line).
- From the CLI: `agentlens experiments ls`, `agentlens traces ls|show`,
  `agentlens export PATH` and `agentlens import PATH` (see the [CLI reference](../cli.md)).

The `Store` protocol is small, so another backend (Postgres, say) can be added without touching
the rest. See also: [ADR 0003](../adr/0003-sqlite-store.md),
[design §2b-2c](../design.md#2b-store-agentlensstore-adr-0003), API:
[`agentlens.store`](../api/store.md).
