"""Built-ins are registered on import and buildable from suite-style specs."""

from __future__ import annotations

import asyncio
from typing import Any

import pytest
import yaml

from agentlens.evals import (
    Contains,
    CostBudget,
    EvaluatorConfigError,
    EvaluatorRegistry,
    ExactMatch,
    JsonSchema,
    LatencyBudget,
    MaxSteps,
    NoError,
    Regex,
    ToolSequence,
    aevaluate_case,
    available_evaluators,
    default_registry,
    evaluate_case,
    get_evaluator,
    register_builtins,
)
from agentlens.evals.builtin import BUILTIN_EVALUATORS
from agentlens.schema import Case, Span, SpanKind, Trace
from tests.evals.builtin.conftest import build_trace


def test_builtins_are_in_the_default_registry() -> None:
    assert {
        "exact_match",
        "contains",
        "regex",
        "json_schema",
        "no_error",
        "tool_sequence",
        "max_steps",
        "cost_budget",
        "latency_budget",
    } <= set(available_evaluators())
    for name, (_factory, primary) in BUILTIN_EVALUATORS.items():
        assert default_registry.entry(name).primary == primary


@pytest.mark.parametrize(
    ("spec", "expected"),
    [
        ("no_error", NoError()),
        ({"no_error": False}, NoError(include_spans=False)),
        ({"no_error": {"include_spans": False}}, NoError(include_spans=False)),
        ("exact_match", ExactMatch()),
        ({"exact_match": None}, ExactMatch()),
        ({"exact_match": "Paris"}, ExactMatch("Paris")),
        ({"exact_match": {"value": None}}, ExactMatch(None)),
        ({"exact_match": {"value": {"a": 1}, "path": "data"}}, ExactMatch({"a": 1}, path="data")),
        ({"contains": "refund"}, Contains("refund")),
        ({"contains": ["a", "b"]}, Contains(["a", "b"])),
        ({"contains": {"text": ["a"], "mode": "any"}}, Contains(["a"], mode="any")),
        ({"regex": "order \\d+"}, Regex("order \\d+")),
        ({"regex": {"pattern": "^ok$", "ignore_case": True}}, Regex("^ok$", ignore_case=True)),
        ({"json_schema": {"schema": {"type": "object"}}}, JsonSchema({"type": "object"})),
        ("json_schema", JsonSchema()),
        ("tool_sequence", ToolSequence()),
        ({"tool_sequence": ["a", "b"]}, ToolSequence(["a", "b"])),
        ({"tool_sequence": {"mode": "ordered"}}, ToolSequence(mode="ordered")),
        (
            {"tool_sequence": {"tools": ["a"], "mode": "strict", "forbidden": ["rm"]}},
            ToolSequence(["a"], mode="strict", forbidden=["rm"]),
        ),
        ({"max_steps": 8}, MaxSteps(8)),
        ({"max_steps": {"value": 8}}, MaxSteps(8)),
        ({"cost_budget": 0.05}, CostBudget(0.05)),
        ({"cost_budget": {"max_usd": 1, "on_unknown": "pass"}}, CostBudget(1, on_unknown="pass")),
        ({"latency_budget": 5000}, LatencyBudget(5000)),
        ({"latency_budget": {"max_ms": 250.5}}, LatencyBudget(250.5)),
    ],
)
def test_build_from_suite_specs(spec: Any, expected: object) -> None:
    built = get_evaluator(spec)
    assert type(built) is type(expected)
    assert repr(built) == repr(expected)


@pytest.mark.parametrize(
    ("spec", "message"),
    [
        ({"regex": "("}, "invalid regex"),
        ({"regex": {"patern": "x"}}, "invalid parameters for evaluator 'regex'"),
        ("regex", "missing a required argument: 'pattern'"),
        ({"contains": {"text": "x", "mode": "every"}}, "mode must be one of"),
        ({"contains": []}, "non-empty"),
        ({"json_schema": {"schema": {"type": 3}}}, "not a valid JSON schema"),
        ({"no_error": "yes"}, "include_spans must be true or false"),
        ({"exact_match": {"value": "x", "strip": "yes"}}, "strip must be true or false"),
        ({"tool_sequence": {"mode": "exact"}}, "mode must be one of"),
        ({"tool_sequence": "lookup_order"}, "tools must be a list of tool names"),
        ("max_steps", "missing a required argument: 'value'"),
        ({"max_steps": "8"}, "must be an integer >= 0"),
        ({"cost_budget": -1}, "max_usd must be a finite number >= 0"),
        ({"cost_budget": {"max_usd": 1, "on_unknown": "skip"}}, "on_unknown must be one of"),
        ({"latency_budget": {"max_s": 5}}, "invalid parameters for evaluator 'latency_budget'"),
    ],
)
def test_bad_specs_fail_at_build_time(spec: Any, message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        get_evaluator(spec)


SUITE_CASE = """
id: refund
input: "I want a refund for order 1234"
expected: "Refund issued"
evaluators:
  - no_error
  - {contains: refund}
  - {contains: {text: [Refund, "order 1234"], mode: all}}
  - {regex: {pattern: "order \\\\d{4}"}}
  - exact_match
  - {exact_match: {value: "refund issued", case_sensitive: false}}
  - {json_schema: {schema: {type: object, required: [status]}, path: data}}
"""


def _agent_trace() -> Trace:
    span = Span(trace_id="t1", kind=SpanKind.TOOL_CALL, name="issue_refund")
    return Trace(
        id="t1",
        name="agent",
        input="I want a refund for order 1234",
        output="Refund issued",
        spans=[span],
    )


def test_evaluate_a_case_written_like_a_suite_file() -> None:
    case = Case.model_validate(yaml.safe_load(SUITE_CASE))
    trace = _agent_trace()
    results = evaluate_case(trace, case)
    summary = [(r.evaluator, r.passed, round(r.score, 2)) for r in results]
    assert summary == [
        ("no_error", True, 1.0),
        ("contains", False, 0.0),  # case-sensitive by default
        ("contains", False, 0.5),
        ("regex", False, 0.0),  # the output does not mention the order
        ("exact_match", True, 1.0),
        ("exact_match", True, 1.0),
        ("json_schema", False, 0.0),  # plain-text output has no 'data'
    ]
    assert results[6].reason == "output is a non-JSON string, it has no 'data'"
    assert asyncio.run(aevaluate_case(trace, case)) == results


def test_register_builtins_into_a_fresh_registry() -> None:
    registry = EvaluatorRegistry()
    register_builtins(registry)
    assert registry.names() == sorted(BUILTIN_EVALUATORS)
    register_builtins(registry)  # idempotent
    assert isinstance(registry.get({"contains": "x"}), Contains)


TRAJECTORY_SUITE_CASE = """
id: refund-simple
input: "I want a refund for order 1234"
expected: {tools: [lookup_order, issue_refund]}
evaluators:
  - {tool_sequence: {mode: ordered}}
  - {tool_sequence: {mode: strict}}
  - {tool_sequence: {mode: subset, forbidden: [delete_account]}}
  - {max_steps: 8}
  - {max_steps: 2}
  - {cost_budget: 0.05}
  - {latency_budget: 5000}
"""


def test_trajectory_checks_from_a_suite_file() -> None:
    case = Case.model_validate(yaml.safe_load(TRAJECTORY_SUITE_CASE))
    trace = build_trace(
        ["lookup_order", "search_kb", "issue_refund"], costs=[0.01, 0.02, 0.01], latency_ms=3200
    )
    summary = [(r.evaluator, r.passed, round(r.score, 2)) for r in evaluate_case(trace, case)]
    assert summary == [
        ("tool_sequence", True, 1.0),
        ("tool_sequence", False, 0.67),  # search_kb is an extra call
        ("tool_sequence", True, 1.0),
        ("max_steps", True, 1.0),
        ("max_steps", False, 0.67),  # 3 LLM calls > 2
        ("cost_budget", True, 1.0),
        ("latency_budget", True, 1.0),
    ]
