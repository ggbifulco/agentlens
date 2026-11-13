"""Persistence of traces, experiments and evaluation results (ADR 0003, FR-T4).

Typical use::

    import agentlens
    from agentlens.store import SQLiteStore

    with SQLiteStore("runs.db") as store:
        with agentlens.run("support-agent", input="hi", sink=store.save_trace):
            ...
        recent = store.list_traces(name="support-agent", limit=10)
"""

from agentlens.store.base import MetadataScalar, SchemaVersionError, Store, StoreError
from agentlens.store.migrations import SCHEMA_VERSION
from agentlens.store.sqlite import DEFAULT_PATH, SQLiteStore

__all__ = [
    "DEFAULT_PATH",
    "SCHEMA_VERSION",
    "MetadataScalar",
    "SQLiteStore",
    "SchemaVersionError",
    "Store",
    "StoreError",
]
