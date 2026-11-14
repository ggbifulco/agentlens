"""The suite runner: every case x N repetitions, bounded concurrency, one experiment (FR-S2).

::

    from agentlens.store import SQLiteStore
    from agentlens.suite import load_suite, run_suite

    with SQLiteStore() as store:
        result = run_suite(load_suite("suites/support.yaml"), store=store)
    print(result.experiment.id, f"{result.pass_rate:.0%}")

Each run is recorded with :func:`agentlens.run` (trace name = suite name,
metadata ``case_id``, ``repetition``, ``suite``, ``experiment``), scored with
the case's evaluators and saved under the experiment. One run failing (the
agent raising or timing out) is recorded on its trace (``trace.error``) and
never stops the suite.

**How the agent is called.** ``agent(case.input)``; if the agent has a
parameter named ``model`` and a model is known (``model=`` here, else
``suite.model``, else ``$AGENTLENS_MODEL``) it is also passed as
``model="provider:name"``, ready for :func:`agentlens.get_client`. ``async
def`` agents are awaited on the event loop; sync agents run in a worker thread
(:func:`asyncio.to_thread`), which copies the context, so the tracing context
(:func:`agentlens.wrap_llm`, :func:`agentlens.wrap_tool`) works inside both.

**Concurrency.** At most ``concurrency`` runs are in flight at once
(:class:`asyncio.Semaphore`); a slot covers the agent call *and* its
evaluation, so LLM judges are bounded too.

**Timeouts.** ``timeout_s`` bounds each agent call (not the evaluation). An
async agent is cancelled; a sync agent's thread cannot be stopped, so it keeps
running in the background while the run is recorded as timed out — spans it
creates afterwards are not part of the trace.
"""

from __future__ import annotations

import asyncio
import contextlib
import inspect
import logging
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from pydantic import JsonValue

from agentlens.evals import Evaluator, EvaluatorRegistry
from agentlens.evals.run import aevaluate_case, resolve_evaluators
from agentlens.schema import Case, EvalResult, Experiment, Trace
from agentlens.schema._base import utc_now
from agentlens.store.base import Store
from agentlens.suite.agent import is_async_agent
from agentlens.suite.experiment import experiment_config, resolve_model
from agentlens.suite.loader import load_suite
from agentlens.suite.model import Suite
from agentlens.tracing import api as tracing
from agentlens.tracing._serialize import to_json_value
from agentlens.tracing.redaction import DEFAULT_REDACTOR, Redactor

__all__ = [
    "CaseResult",
    "ExperimentRun",
    "ProgressCallback",
    "RunResult",
    "RunTimeoutError",
    "arun_suite",
    "default_experiment_name",
    "run_suite",
]

logger = logging.getLogger("agentlens.suite")

REDACTION_FAILED = "AgentLens: redaction failed, trace content was dropped (see the log)"
"""``trace.error`` of a run whose trace could not be redacted (nothing secret is stored)."""


class RunTimeoutError(TimeoutError):
    """An agent call exceeded the per-run timeout (recorded on the trace)."""


@dataclass(frozen=True)
class RunResult:
    """One execution of one case: the stored trace and its evaluation results."""

    case_id: str
    repetition: int
    """0-based repetition index."""
    trace: Trace
    """The trace as stored (redacted)."""
    eval_results: list[EvalResult] = field(default_factory=list)

    @property
    def error(self) -> str | None:
        """The agent's exception (``"Type: message"``), or ``None``."""
        return self.trace.error

    @property
    def timed_out(self) -> bool:
        """Whether the run was stopped by the per-run timeout."""
        return (self.trace.error or "").startswith(f"{RunTimeoutError.__name__}:")

    @property
    def passed(self) -> bool:
        """No error and every evaluator passed (no evaluators: no error is enough)."""
        return self.trace.error is None and all(r.passed for r in self.eval_results)


@dataclass(frozen=True)
class CaseResult:
    """All repetitions of one case."""

    case: Case
    """The resolved case (suite defaults merged in)."""
    runs: list[RunResult]
    """Ordered by repetition."""

    @property
    def case_id(self) -> str:
        """The case id."""
        return self.case.id

    @property
    def passed(self) -> int:
        """How many repetitions passed."""
        return sum(run.passed for run in self.runs)

    @property
    def pass_rate(self) -> float:
        """Fraction of repetitions that passed (0 when there are none)."""
        return self.passed / len(self.runs) if self.runs else 0.0

    @property
    def errors(self) -> int:
        """How many repetitions raised or timed out."""
        return sum(run.error is not None for run in self.runs)

    def mean_scores(self) -> dict[str, float]:
        """Mean score per evaluator name over the repetitions."""
        scores: dict[str, list[float]] = {}
        for run in self.runs:
            for result in run.eval_results:
                scores.setdefault(result.evaluator, []).append(result.score)
        return {name: sum(values) / len(values) for name, values in scores.items()}


@dataclass(frozen=True)
class ExperimentRun:
    """What :func:`run_suite` returns: the experiment and every run, by case."""

    experiment: Experiment
    suite: Suite
    cases: list[CaseResult]
    """In suite order."""

    @property
    def runs(self) -> list[RunResult]:
        """Every run, in case order then repetition order."""
        return [run for case in self.cases for run in case.runs]

    @property
    def total(self) -> int:
        """Number of runs."""
        return sum(len(case.runs) for case in self.cases)

    @property
    def passed(self) -> int:
        """Number of runs that passed."""
        return sum(case.passed for case in self.cases)

    @property
    def errors(self) -> int:
        """Number of runs that raised or timed out."""
        return sum(case.errors for case in self.cases)

    @property
    def pass_rate(self) -> float:
        """Fraction of runs that passed (0 when there are none)."""
        return self.passed / self.total if self.total else 0.0

    def case(self, case_id: str) -> CaseResult:
        """The result of case ``case_id`` (``KeyError`` if none)."""
        for case in self.cases:
            if case.case_id == case_id:
                return case
        raise KeyError(case_id)


ProgressCallback = Callable[[RunResult, int, int], None]
"""``progress(result, completed, total)``, called after each run is saved."""


def default_experiment_name(suite: Suite, created_at: datetime) -> str:
    """``"<suite>-YYYYmmdd-HHMMSS"`` (UTC), the name used when none is given."""
    return f"{suite.name}-{created_at:%Y%m%d-%H%M%S}"


def _accepts_model(agent: Callable[..., Any]) -> bool:
    try:
        parameters = inspect.signature(agent).parameters
    except (TypeError, ValueError):
        return False
    param = parameters.get("model")
    return param is not None and param.kind in (
        inspect.Parameter.POSITIONAL_OR_KEYWORD,
        inspect.Parameter.KEYWORD_ONLY,
    )


class _Runner:
    def __init__(
        self,
        *,
        suite: Suite,
        agent: Callable[..., Any],
        experiment: Experiment,
        store: Store | None,
        model: str | None,
        timeout_s: float | None,
        redactor: Redactor | None,
        progress: ProgressCallback | None,
        total: int,
    ) -> None:
        self.suite = suite
        self.agent = agent
        self.is_async = is_async_agent(agent)
        self.kwargs: dict[str, Any] = (
            {"model": model} if model is not None and _accepts_model(agent) else {}
        )
        self.experiment = experiment
        self.store = store
        self.timeout_s = timeout_s
        self.redactor = redactor
        self.progress = progress
        self.total = total
        self.completed = 0

    async def _call(self, case: Case) -> object:
        if self.is_async:
            call = self.agent(case.input, **self.kwargs)
        else:
            call = asyncio.to_thread(self.agent, case.input, **self.kwargs)
        if self.timeout_s is None:
            return await call
        try:
            return await asyncio.wait_for(call, self.timeout_s)
        except TimeoutError:
            raise RunTimeoutError(f"run exceeded timeout_s={self.timeout_s:g}") from None

    async def _record(self, case: Case, repetition: int) -> Trace:
        stored: list[Trace] = []
        metadata = {
            "case_id": case.id,
            "repetition": repetition,
            "suite": self.suite.name,
            "experiment": self.experiment.id,
        }
        # An exception is recorded on the trace by agentlens.run, then suppressed.
        with (
            contextlib.suppress(Exception),
            tracing.run(
                self.suite.name,
                input=case.input,
                metadata=metadata,
                sink=stored.append,
                redactor=self.redactor,
            ) as live,
        ):
            output = await self._call(case)
            live.output = to_json_value(output)
        if stored:
            return stored[0]
        # Redaction failed: keep the run (as an error) but none of its content.
        return Trace(
            id=live.id,
            name=live.name,
            metadata=live.metadata,
            start_time=live.start_time,
            end_time=live.end_time,
            error=REDACTION_FAILED,
        )

    async def execute(
        self,
        case: Case,
        evaluators: Sequence[Evaluator],
        repetition: int,
        slots: asyncio.Semaphore,
    ) -> RunResult:
        async with slots:
            trace = await self._record(case, repetition)
            results = await aevaluate_case(trace, case, evaluators) if evaluators else []
        if self.store is not None:
            self.store.save_trace(trace, experiment_id=self.experiment.id)
            self.store.save_eval_results(trace.id, results)
        result = RunResult(case.id, repetition, trace, results)
        self.completed += 1
        if self.progress is not None:
            try:
                self.progress(result, self.completed, self.total)
            except Exception:
                logger.exception("progress callback failed")
        return result


async def arun_suite(
    suite: Suite | str | os.PathLike[str],
    *,
    store: Store | None = None,
    name: str | None = None,
    model: str | None = None,
    repetitions: int | None = None,
    concurrency: int | None = None,
    timeout_s: float | None = None,
    config: Mapping[str, JsonValue] | None = None,
    registry: EvaluatorRegistry | None = None,
    redactor: Redactor | None = DEFAULT_REDACTOR,
    progress: ProgressCallback | None = None,
) -> ExperimentRun:
    """Run every case of ``suite`` ``repetitions`` times and record an experiment.

    Async version of :func:`run_suite`, for callers already inside an event loop.

    Args:
        suite: A :class:`Suite`, or the path of a suite YAML file.
        store: Where the experiment, traces and eval results are saved
            (e.g. :class:`~agentlens.store.SQLiteStore`); ``None`` keeps them
            in the returned object only.
        name: Experiment name; default ``"<suite>-YYYYmmdd-HHMMSS"``.
        model: Model under test, overriding ``suite.model``/``$AGENTLENS_MODEL``;
            passed to agents that take a ``model`` parameter.
        repetitions: Overrides ``suite.repetitions``.
        concurrency: Overrides ``suite.concurrency``.
        timeout_s: Overrides ``suite.timeout_s`` (seconds per agent call).
        config: Extra values for ``Experiment.config`` (e.g. a prompt version);
            they win over the collected ones (:func:`experiment_config`).
        registry: Evaluator registry (default: the process-wide one).
        redactor: Applied to every trace before it is evaluated and saved;
            ``None`` disables redaction.
        progress: ``progress(result, completed, total)`` after each run; its
            exceptions are logged and ignored.

    Returns:
        An :class:`ExperimentRun` (experiment, per-case results, pass rate).

    Raises:
        SuiteError: if the suite file is invalid or the agent cannot be imported.
        EvaluatorError: if an evaluator spec cannot be built (before anything runs).
        ValueError: for ``repetitions``/``concurrency`` < 1 or ``timeout_s`` <= 0.
    """
    if not isinstance(suite, Suite):
        suite = load_suite(suite, registry=registry)
    reps = repetitions if repetitions is not None else suite.repetitions
    width = concurrency if concurrency is not None else suite.concurrency
    timeout = timeout_s if timeout_s is not None else suite.timeout_s
    if reps < 1 or width < 1:
        raise ValueError("repetitions and concurrency must be >= 1")
    if timeout is not None and timeout <= 0:
        raise ValueError("timeout_s must be > 0")

    agent = suite.load_agent()
    cases = suite.resolved_cases()
    evaluators = {case.id: resolve_evaluators(case.evaluators, registry=registry) for case in cases}
    effective_model = resolve_model(suite, model)

    created_at = utc_now()
    experiment = Experiment(
        name=name or default_experiment_name(suite, created_at),
        suite=suite.name,
        created_at=created_at,
        config=experiment_config(
            suite,
            model=effective_model,
            repetitions=reps,
            concurrency=width,
            timeout_s=timeout,
            extra=config,
        ),
    )
    if store is not None:
        store.save_experiment(experiment)

    runner = _Runner(
        suite=suite,
        agent=agent,
        experiment=experiment,
        store=store,
        model=effective_model,
        timeout_s=timeout,
        redactor=redactor,
        progress=progress,
        total=len(cases) * reps,
    )
    slots = asyncio.Semaphore(width)
    tasks = [
        [
            asyncio.create_task(runner.execute(case, evaluators[case.id], rep, slots))
            for rep in range(reps)
        ]
        for case in cases
    ]
    try:
        await asyncio.gather(*(task for row in tasks for task in row))
    finally:
        for row in tasks:
            for task in row:
                task.cancel()
    return ExperimentRun(
        experiment=experiment,
        suite=suite,
        cases=[
            CaseResult(case, [task.result() for task in row])
            for case, row in zip(cases, tasks, strict=True)
        ],
    )


def run_suite(
    suite: Suite | str | os.PathLike[str],
    *,
    store: Store | None = None,
    name: str | None = None,
    model: str | None = None,
    repetitions: int | None = None,
    concurrency: int | None = None,
    timeout_s: float | None = None,
    config: Mapping[str, JsonValue] | None = None,
    registry: EvaluatorRegistry | None = None,
    redactor: Redactor | None = DEFAULT_REDACTOR,
    progress: ProgressCallback | None = None,
) -> ExperimentRun:
    """Run ``suite`` and record an experiment (blocking; see :func:`arun_suite`).

    Starts its own event loop with :func:`asyncio.run`, so it cannot be called
    from inside a running loop — use ``await arun_suite(...)`` there.
    """
    return asyncio.run(
        arun_suite(
            suite,
            store=store,
            name=name,
            model=model,
            repetitions=repetitions,
            concurrency=concurrency,
            timeout_s=timeout_s,
            config=config,
            registry=registry,
            redactor=redactor,
            progress=progress,
        )
    )
