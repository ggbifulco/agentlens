"""JSONL export/import of traces: the interchange format of ADR 0003 (FR-T4).

One trace per line, each line the JSON document of a :class:`~agentlens.schema.Trace`
(exactly what ``Trace.model_dump_json()`` produces). The format is plain enough
to ``grep``, ``jq``, diff in a PR or commit as a test fixture::

    from agentlens.store import SQLiteStore, export_jsonl, import_jsonl

    with SQLiteStore("runs.db") as store:
        export_jsonl(store, "fixtures/refunds.jsonl", metadata={"case_id": "refund-simple"})

    with SQLiteStore("other.db") as store:
        import_jsonl(store, "fixtures/refunds.jsonl")

The functions only use the :class:`~agentlens.store.Store` protocol, so they work
with any backend. :func:`write_jsonl` / :func:`read_jsonl` work on traces
directly, without a store.
"""

from __future__ import annotations

import os
import tempfile
from collections.abc import Iterable, Iterator, Mapping
from pathlib import Path

from pydantic import ValidationError

from agentlens.schema import Trace
from agentlens.store.base import MetadataScalar, Store, StoreError

__all__ = ["JSONLFormatError", "export_jsonl", "import_jsonl", "read_jsonl", "write_jsonl"]


class JSONLFormatError(StoreError):
    """A line of a JSONL file is not a valid trace.

    The message names the file and the (1-based) line number.
    """

    def __init__(self, path: str | os.PathLike[str], line: int, reason: str) -> None:
        self.path = os.fspath(path)
        self.line = line
        super().__init__(f"{self.path}:{line}: invalid trace: {reason}")


def write_jsonl(traces: Iterable[Trace], path: str | os.PathLike[str]) -> int:
    """Write ``traces`` to ``path``, one JSON document per line; return how many.

    The file is written to a temporary sibling and then atomically renamed, so an
    existing file is never left half-overwritten if serialisation fails midway.
    Missing parent directories are created.
    """
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(dir=target.parent, prefix=f".{target.name}.", suffix=".tmp")
    count = 0
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as fh:
            for trace in traces:
                fh.write(trace.model_dump_json())
                fh.write("\n")
                count += 1
        os.replace(tmp_name, target)
    except BaseException:
        Path(tmp_name).unlink(missing_ok=True)
        raise
    return count


def read_jsonl(path: str | os.PathLike[str]) -> Iterator[Trace]:
    """Yield the traces stored in a JSONL file, in file order.

    Blank lines are skipped. A line that is not a valid trace raises
    :class:`JSONLFormatError` (when the iterator reaches it).
    """
    with Path(path).open(encoding="utf-8") as fh:
        for number, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                yield Trace.model_validate_json(line)
            except ValidationError as exc:
                raise JSONLFormatError(path, number, _summarise(exc)) from exc


def _summarise(exc: ValidationError) -> str:
    """First validation error as ``loc: msg`` (full details stay on ``__cause__``)."""
    first = exc.errors()[0]
    loc = ".".join(str(part) for part in first["loc"])
    more = exc.error_count() - 1
    suffix = f" (+{more} more)" if more else ""
    return f"{loc}: {first['msg']}{suffix}" if loc else f"{first['msg']}{suffix}"


def export_jsonl(
    store: Store,
    path: str | os.PathLike[str],
    *,
    name: str | None = None,
    experiment_id: str | None = None,
    metadata: Mapping[str, MetadataScalar] | None = None,
    limit: int | None = None,
) -> int:
    """Export the traces of ``store`` matching the filters to ``path``; return how many.

    The filters are those of :meth:`Store.list_traces` (``limit`` keeps the most
    recent ones). Traces are written oldest first, so files exported at different
    times diff cleanly and read like a timeline.
    """
    traces = store.list_traces(
        name=name, experiment_id=experiment_id, metadata=metadata, limit=limit
    )
    return write_jsonl(reversed(traces), path)


def import_jsonl(
    store: Store,
    path: str | os.PathLike[str],
    *,
    experiment_id: str | None = None,
) -> int:
    """Import every trace of a JSONL file into ``store``; return how many.

    The whole file is validated before anything is saved, so a malformed line
    (:class:`JSONLFormatError`) leaves the store untouched. Saving is an upsert:
    importing the same file twice is a no-op the second time. ``experiment_id``
    links all imported traces to that (already stored) experiment.
    """
    traces = list(read_jsonl(path))
    for trace in traces:
        store.save_trace(trace, experiment_id=experiment_id)
    return len(traces)
