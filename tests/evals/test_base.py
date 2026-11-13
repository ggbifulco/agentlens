"""Tests for the Evaluator protocol and result normalisation."""

from __future__ import annotations

import math

import pytest

from agentlens.evals import AsyncEvaluator, Evaluator, error_result, to_eval_result
from agentlens.schema import Case, EvalResult, Trace


class PropertyName:
    @property
    def name(self) -> str:
        return "prop"

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        return EvalResult(evaluator=self.name, score=1.0, passed=True)


class AttrName:
    name = "attr"

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        return EvalResult(evaluator=self.name, score=1.0, passed=True)

    async def aevaluate(self, trace: Trace, case: Case) -> EvalResult:
        return self.evaluate(trace, case)


def test_protocol_accepts_attribute_or_property_name() -> None:
    assert isinstance(PropertyName(), Evaluator)
    assert isinstance(AttrName(), Evaluator)
    assert not isinstance(object(), Evaluator)
    assert isinstance(AttrName(), AsyncEvaluator)
    assert not isinstance(PropertyName(), AsyncEvaluator)


def test_eval_result_is_returned_unchanged() -> None:
    result = EvalResult(evaluator="other", score=0.3, passed=True)
    assert to_eval_result("x", result) is result


@pytest.mark.parametrize(("value", "score"), [(True, 1.0), (False, 0.0)])
def test_bool_maps_to_full_or_zero_score(value: bool, score: float) -> None:
    result = to_eval_result("x", value, reason="why")
    assert result == EvalResult(evaluator="x", score=score, passed=value, reason="why")


@pytest.mark.parametrize(
    ("value", "threshold", "passed"),
    [(0.5, 0.5, True), (0.49, 0.5, False), (1, 0.5, True), (0, 0.0, True), (0.8, 0.9, False)],
)
def test_number_is_a_score_compared_to_threshold(
    value: float, threshold: float, passed: bool
) -> None:
    result = to_eval_result("x", value, threshold=threshold)
    assert result.score == float(value)
    assert result.passed is passed
    assert isinstance(result.score, float)


@pytest.mark.parametrize("value", [1.5, -0.1, math.nan, 2])
def test_out_of_range_score_is_rejected(value: float) -> None:
    with pytest.raises(ValueError, match=r"score must be in \[0, 1\]"):
        to_eval_result("x", value)


def test_tuple_carries_a_reason() -> None:
    assert to_eval_result("x", (0.75, "close")) == EvalResult(
        evaluator="x", score=0.75, passed=True, reason="close"
    )
    assert to_eval_result("x", (False, None)).reason is None


@pytest.mark.parametrize("value", [(1.0,), (1.0, "a", "b"), (1.0, 3)])
def test_malformed_tuple_is_rejected(value: tuple[object, ...]) -> None:
    with pytest.raises(TypeError, match="tuple result"):
        to_eval_result("x", value)


@pytest.mark.parametrize("value", [None, "yes", [1.0], object()])
def test_unsupported_types_are_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="expected EvalResult"):
        to_eval_result("x", value)


def test_mapping_fills_missing_score_or_passed() -> None:
    assert to_eval_result("x", {"passed": True}) == EvalResult(
        evaluator="x", score=1.0, passed=True
    )
    assert to_eval_result("x", {"passed": False}).score == 0.0
    result = to_eval_result("x", {"score": 0.4, "details": {"n": 1}}, threshold=0.3, reason="r")
    assert result == EvalResult(evaluator="x", score=0.4, passed=True, reason="r", details={"n": 1})
    explicit = to_eval_result("x", {"score": 0.9, "passed": False, "reason": "own"}, reason="r")
    assert (explicit.passed, explicit.reason) == (False, "own")
    assert to_eval_result("x", {"score": 1, "evaluator": "named"}).evaluator == "named"


def test_mapping_errors() -> None:
    with pytest.raises(ValueError, match="needs 'score' and/or 'passed'"):
        to_eval_result("x", {"reason": "?"})
    with pytest.raises(TypeError, match="'score' must be a number"):
        to_eval_result("x", {"score": "high"})
    with pytest.raises(TypeError, match="'score' must be a number"):
        to_eval_result("x", {"score": True})
    with pytest.raises(ValueError, match="score must be"):
        to_eval_result("x", {"score": 3})
    with pytest.raises(ValueError, match="invalid result mapping"):
        to_eval_result("x", {"score": 0.5, "passed": True, "details": "not a dict"})


def test_error_result_describes_the_exception() -> None:
    result = error_result("x", KeyError("missing"))
    assert result.score == 0.0
    assert not result.passed
    assert result.reason == "evaluator error: KeyError: 'missing'"
    assert result.details == {"error": {"type": "KeyError", "message": "'missing'"}}
