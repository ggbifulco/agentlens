"""Tests for ``@agentlens.evaluator`` and :class:`FunctionEvaluator`."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest

import agentlens
from agentlens.evals import (
    DuplicateEvaluatorError,
    Evaluator,
    EvaluatorConfigError,
    EvaluatorRegistry,
    FunctionEvaluator,
    evaluator,
    get_evaluator,
)
from agentlens.schema import Case, EvalResult, Trace


def test_bare_decorator_registers_in_the_default_registry(
    clean_default_registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    @agentlens.evaluator
    def mentions_order(trace: Trace, case: Case) -> bool:
        """Checks the answer mentions the order."""
        return "order" in str(trace.output)

    assert isinstance(mentions_order, FunctionEvaluator)
    assert isinstance(mentions_order, Evaluator)
    assert mentions_order.name == "mentions_order"
    assert mentions_order.__doc__ == "Checks the answer mentions the order."
    assert mentions_order.__name__ == "mentions_order"  # type: ignore[attr-defined]
    assert "mentions_order" in clean_default_registry
    assert mentions_order.evaluate(trace, case) == EvalResult(
        evaluator="mentions_order", score=1.0, passed=True
    )
    built = get_evaluator("mentions_order")
    assert built.evaluate(trace, case).passed
    # still callable as the plain function
    assert mentions_order(trace, case) is True


def test_options_name_threshold_and_params(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    @evaluator(name="short_answer", threshold=0.8, registry=registry)
    def short(trace: Trace, case: Case, max_words: int = 50) -> tuple[float, str]:
        words = len(str(trace.output).split())
        return min(1.0, max_words / max(words, 1)), f"{words} words"

    assert short.name == "short_answer"
    assert short.evaluate(trace, case).passed  # 5 words, max 50
    configured = registry.get({"short_answer": 4})  # scalar -> first param
    result = configured.evaluate(trace, case)
    assert result == EvalResult(evaluator="short_answer", score=0.8, passed=True, reason="5 words")
    stricter = registry.get({"short_answer": {"max_words": 4, "threshold": 0.9}})
    assert not stricter.evaluate(trace, case).passed
    assert repr(stricter) == (
        "FunctionEvaluator('short_answer', threshold=0.9, params={'max_words': 4})"
    )
    assert repr(short) == "FunctionEvaluator('short_answer', threshold=0.8)"


def test_scalar_shorthand_of_a_parameterless_function_sets_the_threshold(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    @evaluator(registry=registry)
    def half(trace: Trace, case: Case) -> float:
        return 0.6

    assert half.evaluate(trace, case).passed
    assert not registry.get({"half": 0.7}).evaluate(trace, case).passed
    assert registry.entry("half").primary == "threshold"


def test_function_declaring_threshold_receives_it(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    seen: list[float] = []

    @evaluator(registry=registry, threshold=0.3)
    def uses_threshold(trace: Trace, case: Case, threshold: float = 0.0) -> float:
        seen.append(threshold)
        return 0.5

    uses_threshold.evaluate(trace, case)
    result = registry.get({"uses_threshold": 0.6}).evaluate(trace, case)
    assert seen == [0.3, 0.6]
    assert not result.passed


def test_kwargs_function_gets_params_but_not_threshold(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    seen: list[dict[str, Any]] = []

    @evaluator(registry=registry)
    def flexible(trace: Trace, case: Case, **params: Any) -> bool:
        seen.append(params)
        return True

    registry.get({"flexible": {"a": 1, "threshold": 0.9}}).evaluate(trace, case)
    assert seen == [{"a": 1}]


def test_bad_params_are_config_errors(registry: EvaluatorRegistry) -> None:
    @evaluator(registry=registry)
    def needs(trace: Trace, case: Case, *, pattern: str) -> bool:
        return True

    with pytest.raises(
        EvaluatorConfigError, match=r"missing a required (keyword-only )?argument: 'pattern'"
    ):
        registry.get("needs")
    with pytest.raises(EvaluatorConfigError, match="unexpected keyword"):
        registry.get({"needs": {"pattern": "x", "other": 1}})
    with pytest.raises(EvaluatorConfigError, match="threshold must be a number"):
        registry.get({"needs": {"pattern": "x", "threshold": "high"}})
    with pytest.raises(EvaluatorConfigError, match="threshold must be a number"):
        registry.get({"needs": {"pattern": "x", "threshold": True}})
    with pytest.raises(EvaluatorConfigError, match=r"threshold must be in \[0, 1\]"):
        registry.get({"needs": {"pattern": "x", "threshold": 2}})
    assert registry.get({"needs": "x"}).params == {"pattern": "x"}  # type: ignore[attr-defined]


def test_bind_merges_params() -> None:
    def fn(trace: Trace, case: Case, a: int = 0, b: int = 0) -> bool:
        return True

    base = FunctionEvaluator(fn).bind(a=1)
    assert base.bind(b=2).params == {"a": 1, "b": 2}


@pytest.mark.parametrize("threshold", [-0.1, 1.1, float("nan")])
def test_invalid_decorator_threshold(threshold: float) -> None:
    with pytest.raises(ValueError, match="threshold must be in"):
        evaluator(threshold=threshold, register=False)(lambda trace, case: True)


def test_signature_must_accept_trace_and_case() -> None:
    def one(trace: Trace) -> bool:
        return True

    def kw_case(trace: Trace, *, case: Case) -> bool:
        return True

    def required_positional_only(trace: Trace, case: Case, extra: int, /) -> bool:
        return True

    for fn in (one, kw_case):
        with pytest.raises(TypeError, match=r"must accept \(trace, case\)"):
            evaluator(fn)
    with pytest.raises(TypeError, match="positional-only and required"):
        evaluator(required_positional_only)
    with pytest.raises(TypeError, match="needs a function"):
        FunctionEvaluator(42)  # type: ignore[arg-type]


def test_var_positional_and_optional_positional_only_are_accepted(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    @evaluator(registry=registry)
    def star(*args: Any, flag: bool = True) -> bool:
        return flag and len(args) == 2

    @evaluator(registry=registry)
    def opt(trace: Trace, case: Case, extra: int = 1, /, *rest: Any, k: int = 0) -> bool:
        return extra == 1

    assert star.evaluate(trace, case).passed
    assert not registry.get({"star": False}).evaluate(trace, case).passed
    assert registry.entry("opt").primary == "k"
    assert registry.get({"opt": 3}).evaluate(trace, case).passed


def test_register_false_and_duplicates(registry: EvaluatorRegistry) -> None:
    @evaluator(register=False, registry=registry)
    def private(trace: Trace, case: Case) -> bool:
        return True

    assert "private" not in registry

    @evaluator(name="dup", registry=registry)
    def first(trace: Trace, case: Case) -> bool:
        return True

    with pytest.raises(DuplicateEvaluatorError):

        @evaluator(name="dup", registry=registry)
        def second(trace: Trace, case: Case) -> bool:
            return False

    @evaluator(name="dup", registry=registry, replace=True)
    def third(trace: Trace, case: Case) -> bool:
        return False

    assert registry.entry("dup").factory == third.bind


def test_redefining_the_same_function_replaces_it(
    registry: EvaluatorRegistry, trace: Trace, case: Case
) -> None:
    for answer in (True, False):  # like re-running a notebook cell

        @evaluator(registry=registry)
        def cell(trace: Trace, case: Case, _answer: bool = answer) -> bool:
            return _answer

    assert not registry.get("cell").evaluate(trace, case).passed


def test_exceptions_and_bad_returns_become_failed_results(trace: Trace, case: Case) -> None:
    @evaluator(register=False)
    def boom(trace: Trace, case: Case) -> bool:
        raise RuntimeError("judge offline")

    @evaluator(register=False)
    def nothing(trace: Trace, case: Case) -> None:
        return None

    result = boom.evaluate(trace, case)
    assert not result.passed
    assert result.score == 0.0
    assert result.reason == "evaluator error: RuntimeError: judge offline"
    assert result.details["error"] == {"type": "RuntimeError", "message": "judge offline"}
    assert nothing.evaluate(trace, case).reason is not None
    assert "TypeError" in str(nothing.evaluate(trace, case).reason)
    assert asyncio.run(boom.aevaluate(trace, case)) == result


def test_async_functions(trace: Trace, case: Case) -> None:
    @evaluator(register=False)
    async def judge(trace: Trace, case: Case) -> tuple[float, str]:
        await asyncio.sleep(0)
        return 0.9, "good"

    @evaluator(register=False)
    async def broken(trace: Trace, case: Case) -> float:
        raise ValueError("bad json")

    expected = EvalResult(evaluator="judge", score=0.9, passed=True, reason="good")
    assert judge.evaluate(trace, case) == expected  # no running loop
    assert asyncio.run(judge.aevaluate(trace, case)) == expected

    async def inside_loop() -> EvalResult:
        return judge.evaluate(trace, case)  # sync call from a running loop

    assert asyncio.run(inside_loop()) == expected
    assert asyncio.run(broken.aevaluate(trace, case)).reason == (
        "evaluator error: ValueError: bad json"
    )
    assert not broken.evaluate(trace, case).passed


def test_sync_function_aevaluate_runs_in_a_thread(trace: Trace, case: Case) -> None:
    @evaluator(register=False)
    def sync(trace: Trace, case: Case) -> bool:
        return True

    assert asyncio.run(sync.aevaluate(trace, case)).passed


def test_top_level_exports() -> None:
    assert agentlens.evaluator is evaluator
    assert agentlens.Evaluator is Evaluator
    assert "evaluator" in agentlens.__all__
    assert "Evaluator" in agentlens.__all__
