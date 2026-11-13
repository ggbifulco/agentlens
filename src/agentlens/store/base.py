"""The :class:`Store` protocol: where traces, experiments and eval results live.

Every component that persists or reads data (the recorder's sink, the suite
runner, the comparator, the CLI) depends on this protocol, not on a concrete
backend, so another backend (e.g. Postgres) can be added later without touching
them (``docs/architecture.md``, *Extension points*).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Protocol, runtime_checkable

from agentlens.schema import EvalResult, Experiment, Trace

__all__ = ["MetadataScalar", "SchemaVersionError", "Store", "StoreError"]

MetadataScalar = str | int | float | bool | None
"""Values accepted by the ``metadata`` filter of :meth:`Store.list_traces`."""


class StoreError(Exception):
    """A store operation failed (e.g. a reference to an unknown experiment)."""


class SchemaVersionError(StoreError):
    """The database was written by a newer AgentLens with an unknown schema version."""


@runtime_checkable
class Store(Protocol):
    """Persistence for traces, experiments and evaluation results.

    Saving is an *upsert*: saving an object whose id already exists replaces it,
    so re-running a step (e.g. re-evaluating a trace) is idempotent.
    """

    # -- traces --------------------------------------------------------------

    def save_trace(self, trace: Trace, *, experiment_id: str | None = None) -> None:
        """Persist ``trace``, optionally linking it to an existing experiment.

        Usable directly as a tracing sink: ``agentlens.run(..., sink=store.save_trace)``.
        Re-saving a trace without ``experiment_id`` keeps its existing link.
        """
        ...

    def get_trace(self, trace_id: str) -> Trace | None:
        """The trace with id ``trace_id``, or ``None`` if it is not stored."""
        ...

    def list_traces(
        self,
        *,
        name: str | None = None,
        experiment_id: str | None = None,
        metadata: Mapping[str, MetadataScalar] | None = None,
        limit: int | None = None,
    ) -> list[Trace]:
        """Stored traces matching *all* given filters, most recent first.

        ``metadata`` matches traces whose metadata has each key with exactly that
        scalar value (``{"case_id": "refund-simple", "repetition": 0}``).
        """
        ...

    # -- experiments ---------------------------------------------------------

    def save_experiment(self, experiment: Experiment) -> None:
        """Persist ``experiment`` (upsert by id)."""
        ...

    def get_experiment(self, experiment_id: str) -> Experiment | None:
        """The experiment with id ``experiment_id``, or ``None``."""
        ...

    def list_experiments(
        self,
        *,
        name: str | None = None,
        suite: str | None = None,
        is_baseline: bool | None = None,
    ) -> list[Experiment]:
        """Stored experiments matching all given filters, most recent first."""
        ...

    # -- eval results --------------------------------------------------------

    def save_eval_results(self, trace_id: str, results: Sequence[EvalResult]) -> None:
        """Replace the evaluation results of a stored trace with ``results``."""
        ...

    def get_eval_results(self, trace_id: str) -> list[EvalResult]:
        """The evaluation results of one trace, in the order they were saved."""
        ...

    def list_eval_results(self, experiment_id: str) -> dict[str, list[EvalResult]]:
        """Evaluation results of every trace in an experiment, keyed by trace id."""
        ...
