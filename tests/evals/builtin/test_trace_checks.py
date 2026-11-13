"""Tests for the no_error evaluator."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from agentlens.evals import NoError
from agentlens.evals.builtin.trace_checks import MAX_LISTED_SPANS
from agentlens.schema import Span, SpanKind, Trace
from tests.evals.builtin.conftest import make_case

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
    from agentlens.evals import EvaluatorConfigError

    with pytest.raises(EvaluatorConfigError, match="include_spans"):
        NoError("false")  # type: ignore[arg-type]
