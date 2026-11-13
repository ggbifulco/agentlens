"""Tests for the trace checks: no_error, tool_sequence, max_steps, cost/latency budgets."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from typing import Any

import pytest
from pydantic import JsonValue

from agentlens.evals import (
    CostBudget,
    EvaluatorConfigError,
    LatencyBudget,
    MaxSteps,
    NoError,
    ToolSequence,
)
from agentlens.evals.builtin.trace_checks import MAX_LISTED_SPANS, MAX_LISTED_TOOLS, _lcs_matches
from agentlens.schema import Span, SpanKind, Trace
from tests.evals.builtin.conftest import build_trace, make_case

T0 = datetime(2026, 1, 1, tzinfo=UTC)


def _trace(error: str | None = None, span_errors: list[str | None] | None = None) -> Trace:
    spans = [
        Span(
            id=f"s{i}",
            trace_id="t1",
            kind=SpanKind.TOOL_CALL,
            name=f"tool{i}",
            start_time=T0 + timedelta(seconds=10 - i),  # reverse order: sorting matters
            error=err,
        )
        for i, err in enumerate(span_errors or [])
    ]
    return Trace(id="t1", name="agent", input="q", output="a", error=error, spans=spans)


@pytest.mark.parametrize(
    ("trace_error", "span_errors", "include_spans", "passed"),
    [
        (None, [], True, True),
        (None, [None, None], True, True),
        ("ValueError: boom", [], True, False),
        ("ValueError: boom", [], False, False),
        (None, ["KeyError: 'x'"], True, False),
        (None, ["KeyError: 'x'"], False, True),
        ("RuntimeError: x", ["RuntimeError: x"], True, False),
    ],
)
def test_no_error(
    trace_error: str | None,
    span_errors: list[str | None],
    include_spans: bool,
    passed: bool,
) -> None:
    result = NoError(include_spans).evaluate(_trace(trace_error, span_errors), make_case())
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert result.evaluator == "no_error"


def test_no_error_defaults_to_including_spans() -> None:
    assert NoError().include_spans is True
    assert repr(NoError()) == "NoError(include_spans=True)"


def test_no_error_reasons_and_details() -> None:
    ok = NoError().evaluate(_trace(), make_case())
    assert ok.reason == "no errors in the run and its spans"
    assert ok.details == {
        "include_spans": True,
        "trace_error": None,
        "span_error_count": 0,
        "span_errors": [],
    }
    assert NoError(False).evaluate(_trace(), make_case()).reason == "no errors in the run"
    assert NoError(False).evaluate(_trace(), make_case()).details == {
        "include_spans": False,
        "trace_error": None,
    }

    both = NoError().evaluate(_trace("ValueError: boom", ["A: 1", None, "B: 2"]), make_case())
    # spans are listed in start-time order: s2 started before s0
    assert both.reason == (
        "the run raised ValueError: boom; 2 spans recorded an error "
        "(first: tool2 [tool_call]: B: 2)"
    )
    assert both.details["span_error_count"] == 2
    assert both.details["span_errors"] == [
        {"id": "s2", "name": "tool2", "kind": "tool_call", "error": "B: 2"},
        {"id": "s0", "name": "tool0", "kind": "tool_call", "error": "A: 1"},
    ]
    one = NoError().evaluate(_trace(None, ["KeyError: 'x'"]), make_case())
    assert one.reason == "1 span recorded an error (first: tool0 [tool_call]: KeyError: 'x')"


def test_no_error_caps_listed_spans() -> None:
    result = NoError().evaluate(_trace(None, ["E: x"] * 15), make_case())
    assert result.details["span_error_count"] == 15
    assert isinstance(result.details["span_errors"], list)
    assert len(result.details["span_errors"]) == MAX_LISTED_SPANS


def test_no_error_rejects_non_bool() -> None:
    with pytest.raises(EvaluatorConfigError, match="include_spans"):
        NoError("false")  # type: ignore[arg-type]


# -- tool_sequence -------------------------------------------------------------

REFUND = ["lookup_order", "issue_refund"]


@pytest.mark.parametrize(
    ("mode", "actual", "passed", "score"),
    [
        # strict: exact list equality; score = LCS / max(len(expected), len(actual))
        ("strict", REFUND, True, 1.0),
        ("strict", ["lookup_order"], False, 0.5),
        ("strict", ["lookup_order", "search", "issue_refund"], False, 2 / 3),
        ("strict", ["issue_refund", "lookup_order"], False, 0.5),
        ("strict", [], False, 0.0),
        # ordered: expected is a subsequence; score = LCS / len(expected)
        ("ordered", REFUND, True, 1.0),
        ("ordered", ["search", "lookup_order", "search", "issue_refund", "email"], True, 1.0),
        ("ordered", ["issue_refund", "lookup_order"], False, 0.5),
        ("ordered", ["lookup_order"], False, 0.5),
        ("ordered", [], False, 0.0),
        # subset: every expected call happened, any order
        ("subset", ["issue_refund", "lookup_order"], True, 1.0),
        ("subset", ["issue_refund", "email", "lookup_order"], True, 1.0),
        ("subset", ["issue_refund"], False, 0.5),
        ("subset", ["search"], False, 0.0),
    ],
)
def test_tool_sequence_modes(mode: str, actual: list[str], passed: bool, score: float) -> None:
    result = ToolSequence(REFUND, mode=mode).evaluate(  # type: ignore[arg-type]
        build_trace(actual), make_case()
    )
    assert result.evaluator == "tool_sequence"
    assert result.passed is passed
    assert result.score == pytest.approx(score)
    assert result.details["actual"] == actual
    assert result.details["expected"] == REFUND


def test_tool_sequence_counts_repeats() -> None:
    expected = ["search", "search", "answer"]
    once = build_trace(["search", "answer"])
    twice = build_trace(["search", "answer", "search"])
    sub = ToolSequence(expected, mode="subset")
    assert sub.evaluate(once, make_case()).score == pytest.approx(2 / 3)
    assert sub.evaluate(once, make_case()).details["missing"] == ["search"]
    assert sub.evaluate(twice, make_case()).passed
    ordered = ToolSequence(expected, mode="ordered")
    assert not ordered.evaluate(twice, make_case()).passed  # 2nd search came after answer
    assert ordered.evaluate(build_trace(["search", "x", "search", "answer"]), make_case()).passed


def test_tool_sequence_ordered_reports_unmatched_expected_tools() -> None:
    trace = build_trace(["b", "c", "a"])
    result = ToolSequence(["a", "b", "c", "d"]).evaluate(trace, make_case())
    assert result.details["missing"] == ["a", "d"]  # LCS keeps b, c
    assert result.score == 0.5
    assert result.reason == ("missing [a, d] (ordered): expected [a, b, c, d], actual [b, c, a]")


def test_tool_sequence_uses_actual_call_order_not_span_list_order() -> None:
    trace = build_trace(REFUND)
    reordered = trace.model_copy(update={"spans": list(reversed(trace.spans))})
    assert ToolSequence(REFUND, mode="strict").evaluate(reordered, make_case()).passed


def test_tool_sequence_ignores_non_tool_spans() -> None:
    trace = build_trace(REFUND, llm_calls=3, agent_steps=2)
    assert ToolSequence(REFUND, mode="strict").evaluate(trace, make_case()).passed


def test_tool_sequence_expected_from_case() -> None:
    case = make_case({"tools": REFUND, "answer": "done"})
    result = ToolSequence().evaluate(build_trace(REFUND), case)
    assert result.passed
    assert result.details["source"] == "case.expected.tools"
    override = ToolSequence(["lookup_order"]).evaluate(build_trace(REFUND), case)
    assert override.details["source"] == "tools"
    assert override.details["expected"] == ["lookup_order"]


@pytest.mark.parametrize("expected", [None, "lookup_order", {"answer": "x"}])
def test_tool_sequence_without_expected_tools_is_a_config_error(expected: JsonValue) -> None:
    with pytest.raises(EvaluatorConfigError, match="no expected tools for case 'c1'"):
        ToolSequence().evaluate(build_trace(REFUND), make_case(expected))


def test_tool_sequence_rejects_bad_expected_tools_in_case() -> None:
    with pytest.raises(EvaluatorConfigError, match=r"case 'c1' expected\.tools must be a list"):
        ToolSequence().evaluate(build_trace(), make_case({"tools": "lookup_order"}))


def test_tool_sequence_empty_expectation() -> None:
    nothing, some = build_trace([]), build_trace(["search"])
    assert ToolSequence([], mode="strict").evaluate(nothing, make_case()).score == 1.0
    strict = ToolSequence([], mode="strict").evaluate(some, make_case())
    assert (strict.passed, strict.score) == (False, 0.0)
    assert strict.reason == "unexpected [search] (strict): expected [], actual [search]"
    assert ToolSequence([], mode="ordered").evaluate(some, make_case()).passed
    assert not ToolSequence([], allow_extra=False).evaluate(some, make_case()).passed


@pytest.mark.parametrize("mode", ["ordered", "subset"])
def test_tool_sequence_allow_extra_false(mode: str) -> None:
    check = ToolSequence(REFUND, mode=mode, allow_extra=False)  # type: ignore[arg-type]
    retry = build_trace(["lookup_order", "lookup_order", "issue_refund"])
    assert check.evaluate(retry, make_case()).passed  # repeats of expected tools are fine
    extra = check.evaluate(build_trace(["lookup_order", "search", "issue_refund"]), make_case())
    assert not extra.passed
    assert extra.score == pytest.approx(2 / 3)
    assert extra.details["unexpected"] == ["search"]
    assert extra.reason.startswith(f"unexpected [search] ({mode}):")


def test_tool_sequence_extras_are_fine_by_default_but_listed() -> None:
    result = ToolSequence(REFUND).evaluate(
        build_trace(["lookup_order", "search", "issue_refund"]), make_case()
    )
    assert result.passed
    assert result.details["unexpected"] == ["search"]
    assert result.reason == (
        "expected tools called in order (ordered): expected [lookup_order, issue_refund], "
        "actual [lookup_order, search, issue_refund]"
    )


def test_tool_sequence_forbidden_tools() -> None:
    check = ToolSequence(REFUND, forbidden=["delete_account"])
    bad = check.evaluate(
        build_trace(["lookup_order", "delete_account", "issue_refund"]), make_case()
    )
    assert (bad.passed, bad.score) == (False, 0.0)
    assert bad.details["forbidden_called"] == ["delete_account"]
    assert bad.reason.startswith("forbidden tools called: [delete_account] (ordered)")
    assert check.evaluate(build_trace(REFUND), make_case()).passed


def test_tool_sequence_reasons() -> None:
    def reason(mode: str, actual: list[str]) -> str:
        check = ToolSequence(REFUND, mode=mode)  # type: ignore[arg-type]
        return check.evaluate(build_trace(actual), make_case()).reason or ""

    assert reason("strict", REFUND).startswith("tool calls match exactly (strict)")
    assert reason("subset", REFUND[::-1]).startswith("all expected tools called (subset)")
    assert reason("strict", REFUND[::-1]).startswith("missing [issue_refund] (strict)")
    same = ToolSequence(["a", "b"], mode="strict").evaluate(build_trace(["b", "a"]), make_case())
    assert same.reason is not None
    assert same.reason.startswith("missing [b] (strict)")  # ties keep the earlier expected tool
    dup = ToolSequence(["a"], mode="strict").evaluate(build_trace(["a", "a"]), make_case())
    assert dup.reason == (
        "calls differ from the expected sequence (strict): expected [a], actual [a, a]"
    )


def test_tool_sequence_reason_truncates_long_lists() -> None:
    calls = [f"t{i}" for i in range(MAX_LISTED_TOOLS + 5)]
    result = ToolSequence(["x"]).evaluate(build_trace(calls), make_case())
    assert result.reason is not None
    assert "t19, ... +5 more]" in result.reason
    assert result.details["actual"] == calls


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"tools": "lookup_order"}, "tools must be a list of tool names"),
        ({"tools": ["ok", ""]}, "tools must be a list of tool names"),
        ({"tools": [1]}, "tools must be a list of tool names"),
        ({"mode": "exact"}, "mode must be one of 'strict', 'ordered', 'subset'"),
        ({"allow_extra": "no"}, "allow_extra must be true or false"),
        ({"forbidden": "rm"}, "forbidden must be a list of tool names"),
    ],
)
def test_tool_sequence_bad_params(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        ToolSequence(**kwargs)


def test_tool_sequence_repr() -> None:
    assert repr(ToolSequence(("a",), mode="subset", forbidden=["x"])) == (
        "ToolSequence(tools=['a'], mode='subset', allow_extra=True, forbidden=['x'])"
    )


def test_lcs_matches() -> None:
    assert _lcs_matches([], ["a"]) == []
    assert _lcs_matches(["a", "b", "c"], ["a", "c"]) == [0, 2]
    assert _lcs_matches(["a", "a"], ["a"]) == [0]  # ties keep the earliest expected item


# -- max_steps -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("trace", "limit", "passed", "score", "counted"),
    [
        (build_trace(llm_calls=3), 8, True, 1.0, "llm_call"),
        (build_trace(llm_calls=8), 8, True, 1.0, "llm_call"),
        (build_trace(llm_calls=10), 8, False, 0.8, "llm_call"),
        (build_trace(llm_calls=10, agent_steps=4), 5, True, 1.0, "agent_step"),
        (build_trace(), 0, True, 1.0, "llm_call"),
        (build_trace(llm_calls=2), 0, False, 0.0, "llm_call"),
    ],
)
def test_max_steps(trace: Trace, limit: int, passed: bool, score: float, counted: str) -> None:
    result = MaxSteps(limit).evaluate(trace, make_case())
    assert result.evaluator == "max_steps"
    assert (result.passed, result.score) == (passed, pytest.approx(score))
    assert result.details == {"steps": trace.n_steps, "max_steps": limit, "counted": counted}


def test_max_steps_reason() -> None:
    over = MaxSteps(8).evaluate(build_trace(llm_calls=10), make_case())
    assert over.reason == "10 steps (counting llm_call spans), over the limit of 8"
    one = MaxSteps(8).evaluate(build_trace(agent_steps=1), make_case())
    assert one.reason == "1 step (counting agent_step spans), within the limit of 8"
    assert repr(MaxSteps(8)) == "MaxSteps(value=8)"


@pytest.mark.parametrize("value", [-1, 2.5, "8", True, None])
def test_max_steps_bad_value(value: object) -> None:
    with pytest.raises(EvaluatorConfigError, match="max_steps value must be an integer >= 0"):
        MaxSteps(value)  # type: ignore[arg-type]


# -- cost_budget ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("costs", "budget", "passed", "score"),
    [
        ([0.01, 0.02], 0.05, True, 1.0),
        ([0.03, 0.02], 0.05, True, 1.0),
        ([0.06, 0.04], 0.05, False, 0.5),
        ([0.0], 0.0, True, 1.0),
        ([0.01], 0.0, False, 0.0),
    ],
)
def test_cost_budget(costs: list[float], budget: float, passed: bool, score: float) -> None:
    result = CostBudget(budget).evaluate(build_trace(costs=costs), make_case())
    assert result.evaluator == "cost_budget"
    assert (result.passed, result.score) == (passed, pytest.approx(score))
    assert result.details["cost_usd"] == pytest.approx(sum(costs))
    assert result.details["unpriced_spans"] == 0


def test_cost_budget_reason() -> None:
    result = CostBudget(0.05).evaluate(build_trace(costs=[0.1]), make_case())
    assert result.reason == "cost $0.1 is over the budget of $0.05"
    assert repr(CostBudget(1)) == "CostBudget(max_usd=1.0, on_unknown='fail')"


@pytest.mark.parametrize(
    "trace",
    [build_trace(), build_trace(llm_calls=2), build_trace(costs=[None, None])],
    ids=["no-usage", "no-usage-llm-calls", "unpriced"],
)
def test_cost_budget_unknown_cost(trace: Trace) -> None:
    failed = CostBudget(0.05).evaluate(trace, make_case())
    assert (failed.passed, failed.score) == (False, 0.0)
    assert failed.details["cost_usd"] is None
    assert failed.reason is not None
    assert failed.reason.startswith("cost unknown (no span has a cost")
    assert failed.reason.endswith("budget $0.05 not verified, failing (on_unknown: fail)")
    passed = CostBudget(0.05, on_unknown="pass").evaluate(trace, make_case())
    assert (passed.passed, passed.score) == (True, 1.0)
    assert passed.reason is not None
    assert passed.reason.endswith("passing (on_unknown: pass)")


def test_cost_budget_partial_cost() -> None:
    partial = build_trace(costs=[0.01, None])
    result = CostBudget(0.05).evaluate(partial, make_case())
    assert not result.passed
    assert result.details["unpriced_spans"] == 1
    assert result.reason == (
        "cost incomplete: $0.01 known, 1 spans without a cost; "
        "budget $0.05 not verified, failing (on_unknown: fail)"
    )
    assert CostBudget(0.05, on_unknown="pass").evaluate(partial, make_case()).passed
    # the known part alone exceeds the budget: fails whatever on_unknown says
    over = CostBudget(0.005, on_unknown="pass").evaluate(partial, make_case())
    assert (over.passed, over.score) == (False, pytest.approx(0.5))
    assert over.reason == (
        "cost $0.01 (a lower bound: 1 spans have no cost) is over the budget of $0.005"
    )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"max_usd": -0.1}, "max_usd must be a finite number >= 0"),
        ({"max_usd": float("nan")}, "max_usd must be a finite number >= 0"),
        ({"max_usd": float("inf")}, "max_usd must be a finite number >= 0"),
        ({"max_usd": "0.05"}, "max_usd must be a finite number >= 0"),
        ({"max_usd": 1, "on_unknown": "skip"}, "on_unknown must be one of 'fail', 'pass'"),
    ],
)
def test_cost_budget_bad_params(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        CostBudget(**kwargs)


# -- latency_budget ------------------------------------------------------------


@pytest.mark.parametrize(
    ("latency", "budget", "passed", "score"),
    [
        (1200.0, 5000, True, 1.0),
        (5000.0, 5000, True, 1.0),
        (10000.0, 5000, False, 0.5),
        (0.0, 0, True, 1.0),
    ],
)
def test_latency_budget(latency: float, budget: float, passed: bool, score: float) -> None:
    result = LatencyBudget(budget).evaluate(build_trace(latency_ms=latency), make_case())
    assert result.evaluator == "latency_budget"
    assert (result.passed, result.score) == (passed, pytest.approx(score))
    assert result.details == {"latency_ms": latency, "max_ms": float(budget)}


def test_latency_budget_reason() -> None:
    result = LatencyBudget(5000).evaluate(build_trace(latency_ms=7250.5), make_case())
    assert result.reason == "latency 7250.5 ms is over the budget of 5000 ms"
    assert repr(LatencyBudget(5000)) == "LatencyBudget(max_ms=5000.0)"


def test_latency_budget_running_trace_fails() -> None:
    result = LatencyBudget(5000).evaluate(build_trace(latency_ms=None), make_case())
    assert (result.passed, result.score) == (False, 0.0)
    assert result.reason == "latency unknown: the run has not ended (no end_time)"
    assert result.details == {"latency_ms": None, "max_ms": 5000.0}


@pytest.mark.parametrize("value", [-1, "5s", None, float("inf")])
def test_latency_budget_bad_value(value: object) -> None:
    with pytest.raises(EvaluatorConfigError, match="max_ms must be a finite number >= 0"):
        LatencyBudget(value)  # type: ignore[arg-type]
