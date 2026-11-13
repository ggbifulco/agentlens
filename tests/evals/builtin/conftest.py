"""Helpers for built-in evaluator tests."""

from __future__ import annotations

from pydantic import JsonValue

from agentlens.schema import Case, Trace


def make_trace(output: JsonValue = None, **fields: object) -> Trace:
    """A finished trace with the given output."""
    return Trace.model_validate({"name": "agent", "input": "q", "output": output, **fields})


def make_case(expected: JsonValue = None) -> Case:
    """A case with the given expectation."""
    return Case(id="c1", input="q", expected=expected)
