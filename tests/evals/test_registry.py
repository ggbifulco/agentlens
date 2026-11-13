"""Tests for the evaluator registry and spec-based construction."""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any

import pytest

from agentlens.evals import (
    DuplicateEvaluatorError,
    EvaluatorConfigError,
    EvaluatorError,
    EvaluatorRegistry,
    UnknownEvaluatorError,
    available_evaluators,
    default_registry,
    get_evaluator,
    register,
    unregister,
)
from agentlens.schema import Case, EvalResult, EvaluatorSpec, Trace


@dataclass
class MaxSteps:
    limit: int = 10
    name: str = "max_steps"

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        passed = trace.n_steps <= self.limit
        return EvalResult(evaluator=self.name, score=float(passed), passed=passed)


@dataclass
class ToolSequence:
    mode: str = "strict"
    tools: list[str] | None = None
    name: str = "tool_sequence"

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:  # pragma: no cover - unused
        raise NotImplementedError


class NoError:
    name = "no_error"

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        return EvalResult(evaluator=self.name, score=1.0, passed=trace.error is None)


def test_build_from_every_spec_spelling(registry: EvaluatorRegistry) -> None:
    registry.register("no_error", NoError)
    registry.register("max_steps", MaxSteps)
    registry.register("tool_sequence", ToolSequence)

    assert isinstance(registry.get("no_error"), NoError)
    assert registry.get({"max_steps": 8}) == MaxSteps(limit=8)
    assert registry.get({"tool_sequence": {"mode": "ordered"}}) == ToolSequence(mode="ordered")
    assert registry.get({"name": "max_steps", "params": {"limit": 3}}) == MaxSteps(limit=3)
    assert registry.get(EvaluatorSpec(name="max_steps", params={"value": 4})) == MaxSteps(limit=4)
    assert registry.get({"max_steps": None}) == MaxSteps()
    assert registry.get("max_steps", limit=2) == MaxSteps(limit=2)
    assert registry.get({"max_steps": 8}, limit=2) == MaxSteps(limit=2)


def test_scalar_shorthand_uses_explicit_primary(registry: EvaluatorRegistry) -> None:
    registry.register("tool_sequence", ToolSequence, primary="tools")
    assert registry.get({"tool_sequence": ["a", "b"]}) == ToolSequence(tools=["a", "b"])
    assert registry.entry("tool_sequence").primary == "tools"


def test_scalar_shorthand_with_value_param_or_kwargs_is_passed_through(
    registry: EvaluatorRegistry,
) -> None:
    seen: list[dict[str, Any]] = []

    def by_value(other: int = 0, value: int = 0) -> NoError:
        seen.append({"other": other, "value": value})
        return NoError()

    def by_kwargs(**params: Any) -> NoError:
        seen.append(params)
        return NoError()

    registry.register("by_value", by_value, primary="value")
    registry.register("by_kwargs", by_kwargs)
    registry.get({"by_value": 1})
    registry.get({"by_kwargs": 2})
    assert seen == [{"other": 0, "value": 1}, {"value": 2}]
    assert registry.entry("by_kwargs").primary is None


def test_primary_skips_positional_only_parameters(registry: EvaluatorRegistry) -> None:
    def factory(a: int = 0, /, *, limit: int = 1) -> MaxSteps:
        return MaxSteps(limit=limit + a)

    entry = registry.register("f", factory)
    assert entry.primary == "limit"
    assert registry.get({"f": 5}) == MaxSteps(limit=5)


def test_unknown_primary_is_rejected(registry: EvaluatorRegistry) -> None:
    with pytest.raises(ValueError, match="no parameter 'nope'"):
        registry.register("max_steps", MaxSteps, primary="nope")

    def kwargs_factory(**kw: Any) -> NoError:
        return NoError()

    assert registry.register("kw", kwargs_factory, primary="anything").primary == "anything"


def test_parameterless_factory_rejects_params(registry: EvaluatorRegistry) -> None:
    registry.register("no_error", NoError)
    with pytest.raises(EvaluatorConfigError, match="invalid parameters for evaluator 'no_error'"):
        registry.get({"no_error": True})
    with pytest.raises(EvaluatorConfigError, match="unexpected keyword"):
        registry.get({"no_error": {"x": 1}})


def test_wrong_params_are_config_errors(registry: EvaluatorRegistry) -> None:
    registry.register("max_steps", MaxSteps)
    with pytest.raises(EvaluatorConfigError, match="signature"):
        registry.get({"max_steps": {"limt": 3}})


def test_factory_errors_become_config_errors(registry: EvaluatorRegistry) -> None:
    def picky(limit: int = 1) -> MaxSteps:
        if limit < 0:
            raise ValueError("limit must be >= 0")
        return MaxSteps(limit=limit)

    registry.register("picky", picky)
    with pytest.raises(EvaluatorConfigError, match="cannot build evaluator 'picky'") as info:
        registry.get({"picky": -1})
    assert isinstance(info.value.__cause__, ValueError)


def test_factory_must_return_an_evaluator(registry: EvaluatorRegistry) -> None:
    registry.register("bad", lambda: 42)  # type: ignore[arg-type, return-value]
    with pytest.raises(EvaluatorConfigError, match="returned int"):
        registry.get("bad")


def test_uninspectable_factory_skips_the_signature_check(
    registry: EvaluatorRegistry, monkeypatch: pytest.MonkeyPatch
) -> None:
    def factory(**kw: Any) -> NoError:
        return NoError()

    real = inspect.signature

    def fake(obj: Any, *args: Any, **kwargs: Any) -> inspect.Signature:
        if obj is factory:
            raise ValueError("no signature found")
        return real(obj, *args, **kwargs)

    monkeypatch.setattr(inspect, "signature", fake)
    entry = registry.register("opaque", factory)
    assert entry.signature is None
    assert entry.primary is None
    assert isinstance(registry.get({"opaque": 1}), NoError)


def test_malformed_specs_are_config_errors(registry: EvaluatorRegistry) -> None:
    for spec in ({"a": 1, "b": 2}, {"name": ""}, 3):
        with pytest.raises(EvaluatorConfigError, match="invalid evaluator spec"):
            registry.get(spec)  # type: ignore[arg-type]


def test_unknown_name_suggests_close_matches(registry: EvaluatorRegistry) -> None:
    registry.register("max_steps", MaxSteps)
    registry.register("no_error", NoError)
    with pytest.raises(UnknownEvaluatorError) as info:
        registry.get({"max_step": 3})
    err = info.value
    assert isinstance(err, LookupError)
    assert isinstance(err, EvaluatorError)
    assert err.suggestions == ["max_steps"]
    assert str(err) == (
        "unknown evaluator 'max_step'. Did you mean 'max_steps'? Available: max_steps, no_error."
    )
    with pytest.raises(UnknownEvaluatorError, match=r"Available: none\.$"):
        EvaluatorRegistry().get("x")
    with pytest.raises(UnknownEvaluatorError, match=r"'zzz'\. Available"):
        registry.get("zzz")


def test_duplicate_names_are_rejected_unless_replace(registry: EvaluatorRegistry) -> None:
    registry.register("steps", MaxSteps)
    with pytest.raises(DuplicateEvaluatorError, match="already registered"):
        registry.register("steps", NoError)
    registry.register("steps", NoError, replace=True)
    assert isinstance(registry.get("steps"), NoError)


def test_same_origin_re_registers_silently(registry: EvaluatorRegistry) -> None:
    registry.register("steps", MaxSteps)
    registry.register("steps", MaxSteps, primary="limit")
    registry.register("x", NoError, origin="here")
    registry.register("x", MaxSteps, origin="here")
    assert isinstance(registry.get("x"), MaxSteps)


def test_factory_without_origin_cannot_be_matched(registry: EvaluatorRegistry) -> None:
    class Opaque:
        def __call__(self) -> NoError:
            return NoError()

    entry = registry.register("opaque", Opaque())
    assert entry.origin is None
    with pytest.raises(DuplicateEvaluatorError):
        registry.register("opaque", Opaque())


@pytest.mark.parametrize("name", ["", "   "])
def test_empty_names_are_rejected(registry: EvaluatorRegistry, name: str) -> None:
    with pytest.raises(ValueError, match="non-empty"):
        registry.register(name, NoError)


def test_container_behaviour_and_unregister(registry: EvaluatorRegistry) -> None:
    registry.register("b", NoError)
    registry.register("a", MaxSteps)
    assert registry.names() == ["a", "b"]
    assert list(registry) == ["a", "b"]
    assert len(registry) == 2
    assert "a" in registry
    assert "c" not in registry
    assert repr(registry) == "EvaluatorRegistry(['a', 'b'])"
    registry.unregister("a")
    assert registry.names() == ["b"]
    with pytest.raises(UnknownEvaluatorError):
        registry.unregister("a")


def test_module_level_functions_use_the_default_registry(
    clean_default_registry: EvaluatorRegistry,
) -> None:
    register("test_max_steps", MaxSteps)
    assert "test_max_steps" in available_evaluators()
    assert "test_max_steps" in default_registry
    assert get_evaluator({"test_max_steps": 2}) == MaxSteps(limit=2)
    unregister("test_max_steps")
    assert "test_max_steps" not in available_evaluators()


def test_built_evaluator_runs(registry: EvaluatorRegistry, trace: Trace, case: Case) -> None:
    registry.register("max_steps", MaxSteps)
    result = registry.get({"max_steps": 0}).evaluate(trace, case)
    assert result.passed
