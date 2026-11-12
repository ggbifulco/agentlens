"""Tests for Case and the EvaluatorSpec YAML shorthands."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from agentlens.schema import Case, EvaluatorSpec


@pytest.mark.parametrize(
    ("raw", "name", "params"),
    [
        ("no_error", "no_error", {}),
        ({"max_steps": 8}, "max_steps", {"value": 8}),
        ({"tool_sequence": {"mode": "ordered"}}, "tool_sequence", {"mode": "ordered"}),
        ({"no_error": None}, "no_error", {}),
        ({"name": "regex"}, "regex", {}),
        ({"name": "regex", "params": {"pattern": "^ok"}}, "regex", {"pattern": "^ok"}),
    ],
)
def test_evaluator_spec_accepts_yaml_shorthands(
    raw: object, name: str, params: dict[str, object]
) -> None:
    spec = EvaluatorSpec.model_validate(raw)
    assert spec.name == name
    assert spec.params == params


@pytest.mark.parametrize(
    ("spec", "serialised"),
    [
        (EvaluatorSpec(name="no_error"), "no_error"),
        (EvaluatorSpec(name="max_steps", params={"value": 8}), {"max_steps": 8}),
        (EvaluatorSpec(name="t", params={"mode": "ordered"}), {"t": {"mode": "ordered"}}),
        (EvaluatorSpec(name="t", params={"value": None}), {"t": {"value": None}}),
        (EvaluatorSpec(name="t", params={"value": {"a": 1}}), {"t": {"value": {"a": 1}}}),
    ],
)
def test_evaluator_spec_serialises_to_shortest_form(
    spec: EvaluatorSpec, serialised: object
) -> None:
    assert spec.model_dump() == serialised
    assert EvaluatorSpec.model_validate(serialised) == spec


@pytest.mark.parametrize("raw", ["", {"params": {"x": 1}}, {"name": 3}, {"a": 1, "b": 2}])
def test_evaluator_spec_rejects_malformed_input(raw: object) -> None:
    with pytest.raises(ValidationError):
        EvaluatorSpec.model_validate(raw)


def test_case_from_suite_yaml_shape() -> None:
    case = Case.model_validate(
        {
            "id": "refund-simple",
            "input": "I want a refund for order 1234",
            "expected": {"tools": ["lookup_order", "issue_refund"]},
            "evaluators": [{"tool_sequence": {"mode": "ordered"}}, "no_error"],
            "tags": ["refund"],
        }
    )
    assert [e.name for e in case.evaluators] == ["tool_sequence", "no_error"]
    assert case.expected == {"tools": ["lookup_order", "issue_refund"]}


def test_case_defaults() -> None:
    case = Case(id="c1", input={"q": "hi"})
    assert case.expected is None
    assert case.evaluators == []
    assert case.tags == []


def test_case_requires_non_empty_id() -> None:
    with pytest.raises(ValidationError):
        Case(id="", input="x")
