"""Tests for running evaluators safely and evaluating whole cases."""

from __future__ import annotations

import asyncio
import threading

import pytest

from agentlens.evals import (
    EvaluatorConfigError,
    EvaluatorRegistry,
    UnknownEvaluatorError,
    aevaluate_case,
    arun_evaluator,
    evaluate_case,
    evaluator,
    resolve_evaluators,
    run_evaluator,
)
from agentlens.schema import Case, EvalResult, EvaluatorSpec, Trace


class Raising:
    name = "raising"

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        raise ZeroDivisionError("division by zero")


class Nameless:
    name = ""

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        return "nope"  # type: ignore[return-value]


class Fixed:
    def __init__(self, name: str = "fixed", passed: bool = True) -> None:
        self.name = name
        self.passed = passed
        self.threads: list[str] = []

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        self.threads.append(threading.current_thread().name)
        return EvalResult(evaluator=self.name, score=float(self.passed), passed=self.passed)


class AsyncOnly(Fixed):
    def __init__(self) -> None:
        super().__init__("async_only")
        self.awaited = False

    async def aevaluate(self, trace: Trace, case: Case) -> EvalResult:
        self.awaited = True
        return EvalResult(evaluator=self.name, score=0.5, passed=True)


class AsyncRaising(Fixed):
    async def aevaluate(self, trace: Trace, case: Case) -> EvalResult:
        raise RuntimeError("timeout")


class SyncAevaluate(Fixed):
    def aevaluate(self, trace: Trace, case: Case) -> EvalResult:  # not a coroutine
        return EvalResult(evaluator=self.name, score=1.0, passed=True, reason="sync")


class NotCallableAevaluate(Fixed):
    aevaluate = None


def test_run_evaluator_turns_exceptions_into_failed_results(trace: Trace, case: Case) -> None:
    result = run_evaluator(Raising(), trace, case)
    assert result.evaluator == "raising"
    assert not result.passed
    assert result.reason == "evaluator error: ZeroDivisionError: division by zero"


def test_run_evaluator_rejects_non_results(trace: Trace, case: Case) -> None:
    result = run_evaluator(Nameless(), trace, case)
    assert result.evaluator == "Nameless"
    assert (
        result.reason == "evaluator error: TypeError: evaluate() returned str, expected EvalResult"
    )


def test_run_evaluator_returns_results(trace: Trace, case: Case) -> None:
    assert run_evaluator(Fixed(), trace, case).passed


def test_keyboard_interrupt_propagates(trace: Trace, case: Case) -> None:
    class Interrupt(Fixed):
        def evaluate(self, trace: Trace, case: Case) -> EvalResult:
            raise KeyboardInterrupt

    with pytest.raises(KeyboardInterrupt):
        run_evaluator(Interrupt(), trace, case)


def test_arun_prefers_aevaluate(trace: Trace, case: Case) -> None:
    ev = AsyncOnly()
    result = asyncio.run(arun_evaluator(ev, trace, case))
    assert ev.awaited
    assert result.score == 0.5
    assert ev.threads == []


def test_arun_falls_back_to_a_thread(trace: Trace, case: Case) -> None:
    ev = Fixed()
    main = threading.current_thread().name
    assert asyncio.run(arun_evaluator(ev, trace, case)).passed
    assert ev.threads
    assert ev.threads[0] != main
    other = NotCallableAevaluate()
    assert asyncio.run(arun_evaluator(other, trace, case)).passed
    assert asyncio.run(arun_evaluator(Raising(), trace, case)).reason is not None


def test_arun_handles_errors_and_sync_aevaluate(trace: Trace, case: Case) -> None:
    failed = asyncio.run(arun_evaluator(AsyncRaising("judge"), trace, case))
    assert failed.reason == "evaluator error: RuntimeError: timeout"
    assert failed.evaluator == "judge"
    assert asyncio.run(arun_evaluator(SyncAevaluate(), trace, case)).reason == "sync"


def test_resolve_mixes_objects_and_specs(registry: EvaluatorRegistry) -> None:
    registry.register("fixed", Fixed)
    obj = Fixed("mine")
    resolved = resolve_evaluators(
        [obj, "fixed", {"fixed": "named"}, EvaluatorSpec(name="fixed")], registry=registry
    )
    assert resolved[0] is obj
    assert [ev.name for ev in resolved] == ["mine", "fixed", "named", "fixed"]


def test_evaluate_case_defaults_to_case_evaluators(
    registry: EvaluatorRegistry, trace: Trace
) -> None:
    registry.register("fixed", Fixed)
    registry.register("raising", Raising)
    case = Case(
        id="c",
        input="x",
        evaluators=[EvaluatorSpec.model_validate("fixed"), EvaluatorSpec.model_validate("raising")],
    )
    results = evaluate_case(trace, case, registry=registry)
    assert [(r.evaluator, r.passed) for r in results] == [("fixed", True), ("raising", False)]
    explicit = evaluate_case(trace, case, [Fixed("x", passed=False)], registry=registry)
    assert [(r.evaluator, r.passed) for r in explicit] == [("x", False)]


def test_evaluate_case_fails_fast_on_config_errors(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    ran = Fixed()
    with pytest.raises(UnknownEvaluatorError):
        evaluate_case(trace, case, [ran, "missing"], registry=registry)
    registry.register("fixed", Fixed)
    with pytest.raises(EvaluatorConfigError):
        evaluate_case(trace, case, [{"fixed": {"nope": 1}}], registry=registry)
    assert ran.threads == []


def test_evaluate_case_uses_the_default_registry(
    clean_default_registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    @evaluator
    def has_output(trace: Trace, case: Case) -> bool:
        return trace.output is not None

    assert evaluate_case(trace, case, ["has_output"])[0].passed


def test_aevaluate_case_keeps_order(registry: EvaluatorRegistry, trace: Trace) -> None:
    registry.register("fixed", Fixed)

    @evaluator(registry=registry)
    async def slow(trace: Trace, case: Case) -> bool:
        await asyncio.sleep(0.01)
        return True

    case = Case(id="c", input="x", evaluators=[EvaluatorSpec(name="slow")])
    assert [r.evaluator for r in asyncio.run(aevaluate_case(trace, case, registry=registry))] == [
        "slow"
    ]
    results = asyncio.run(
        aevaluate_case(trace, case, ["slow", AsyncOnly(), Raising(), "fixed"], registry=registry)
    )
    assert [r.evaluator for r in results] == ["slow", "async_only", "raising", "fixed"]
    assert [r.passed for r in results] == [True, True, False, True]
