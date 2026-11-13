"""Running evaluators safely: exceptions become failed results, sync and async alike."""

from __future__ import annotations

import asyncio
import inspect
from collections.abc import Iterable, Sequence

from agentlens.evals.base import Evaluator, error_result
from agentlens.evals.registry import EvaluatorRegistry, SpecLike, default_registry
from agentlens.schema import Case, EvalResult, Trace

__all__ = [
    "aevaluate_case",
    "arun_evaluator",
    "evaluate_case",
    "resolve_evaluators",
    "run_evaluator",
]


def _name_of(evaluator: object) -> str:
    name = getattr(evaluator, "name", None)
    return name if isinstance(name, str) and name else type(evaluator).__name__


def _check_result(result: object) -> EvalResult:
    if not isinstance(result, EvalResult):
        raise TypeError(f"evaluate() returned {type(result).__name__}, expected EvalResult")
    return result


def run_evaluator(evaluator: Evaluator, trace: Trace, case: Case) -> EvalResult:
    """Call ``evaluator.evaluate(trace, case)``, never raising.

    An exception (or a return value that is not an :class:`EvalResult`) yields
    :func:`~agentlens.evals.error_result`: score 0, ``passed=False`` and the
    error in ``reason``/``details["error"]``, so one broken evaluator cannot
    crash a suite run. ``KeyboardInterrupt``/``SystemExit`` still propagate.
    """
    try:
        return _check_result(evaluator.evaluate(trace, case))
    except Exception as exc:
        return error_result(_name_of(evaluator), exc)


async def arun_evaluator(evaluator: Evaluator, trace: Trace, case: Case) -> EvalResult:
    """Async :func:`run_evaluator`.

    Awaits ``evaluator.aevaluate(trace, case)`` when the evaluator has one (LLM
    judges); otherwise runs the sync ``evaluate`` in a worker thread so slow
    evaluators do not block the event loop. Never raises.
    """
    aevaluate = getattr(evaluator, "aevaluate", None)
    if aevaluate is None or not callable(aevaluate):
        return await asyncio.to_thread(run_evaluator, evaluator, trace, case)
    try:
        pending = aevaluate(trace, case)
        result = await pending if inspect.isawaitable(pending) else pending
        return _check_result(result)
    except Exception as exc:
        return error_result(_name_of(evaluator), exc)


def resolve_evaluators(
    evaluators: Iterable[Evaluator | SpecLike],
    *,
    registry: EvaluatorRegistry | None = None,
) -> list[Evaluator]:
    """Turn a mix of evaluator objects and specs into evaluator objects.

    Specs are built from ``registry`` (default: the process-wide one). Config
    errors (:class:`~agentlens.evals.UnknownEvaluatorError`,
    :class:`~agentlens.evals.EvaluatorConfigError`) are raised, not swallowed.
    """
    reg = registry if registry is not None else default_registry
    resolved: list[Evaluator] = []
    for item in evaluators:
        if isinstance(item, Evaluator):
            resolved.append(item)
        else:
            resolved.append(reg.get(item))
    return resolved


def evaluate_case(
    trace: Trace,
    case: Case,
    evaluators: Sequence[Evaluator | SpecLike] | None = None,
    *,
    registry: EvaluatorRegistry | None = None,
) -> list[EvalResult]:
    """Score one trace with several evaluators, in order.

    ``evaluators`` defaults to ``case.evaluators``. All of them are resolved first
    (so a misspelt name fails before anything runs), then each runs through
    :func:`run_evaluator`, so evaluator exceptions become failed results.
    """
    chosen = case.evaluators if evaluators is None else evaluators
    return [run_evaluator(ev, trace, case) for ev in resolve_evaluators(chosen, registry=registry)]


async def aevaluate_case(
    trace: Trace,
    case: Case,
    evaluators: Sequence[Evaluator | SpecLike] | None = None,
    *,
    registry: EvaluatorRegistry | None = None,
) -> list[EvalResult]:
    """Async :func:`evaluate_case`: evaluators run concurrently, results keep their order."""
    chosen = case.evaluators if evaluators is None else evaluators
    resolved = resolve_evaluators(chosen, registry=registry)
    return list(await asyncio.gather(*(arun_evaluator(ev, trace, case) for ev in resolved)))
