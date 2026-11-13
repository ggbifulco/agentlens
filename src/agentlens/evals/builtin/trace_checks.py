"""Deterministic checks on *how* the run went, not on its output (FR-E1, ``docs/design.md`` §5).

* ``no_error`` — the run and (by default) its spans raised nothing;
* ``tool_sequence`` — the tools called match an expected list (``strict``,
  ``ordered`` or ``subset``);
* ``max_steps`` — the agent took at most N steps (:attr:`Trace.n_steps`);
* ``cost_budget`` — the run cost at most ``max_usd`` (:attr:`Trace.total_cost`);
* ``latency_budget`` — the run took at most ``max_ms`` (:attr:`Trace.latency_ms`).

Budget checks score ``1`` within the limit and ``limit / actual`` above it, so a
run 10 % over budget scores about ``0.91`` while still failing; ``details``
always hold the measured value and the limit.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from typing import Literal

from pydantic import JsonValue

from agentlens.evals.builtin._common import (
    check_bool,
    check_choice,
    check_number,
    clip,
    params_repr,
)
from agentlens.evals.errors import EvaluatorConfigError
from agentlens.schema import Case, EvalResult, SpanKind, Trace

__all__ = ["CostBudget", "LatencyBudget", "MaxSteps", "NoError", "ToolSequence"]

MAX_LISTED_SPANS = 10
"""How many failed spans ``no_error`` lists in ``details["span_errors"]``."""


class NoError:
    """``no_error``: the agent finished without raising, and (by default) no span failed.

    ``trace.error`` is set when the agent itself raised. With ``include_spans``
    (default ``True``) a span that recorded an error — a tool or LLM call that
    raised, even if the agent caught the exception and carried on — fails the
    check too: in a regression suite a swallowed tool error is usually a bug
    worth seeing. Use ``{no_error: {include_spans: false}}`` to check only the
    run itself. (LLM adapter retries happen inside one span, so a retried and
    recovered HTTP error is not a span error.)

    Score ``1`` without errors, else ``0``; ``details`` holds ``trace_error``,
    ``span_error_count`` and the first failed spans (``id``, ``name``, ``kind``,
    ``error``), in start-time order.

    Args:
        include_spans: Also fail on spans that recorded an error.
    """

    name: str = "no_error"

    def __init__(self, include_spans: bool = True) -> None:
        self.include_spans = check_bool("include_spans", include_spans)

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Look for ``trace.error`` and, if enabled, span errors."""
        failed = (
            sorted((s for s in trace.spans if s.error is not None), key=lambda s: s.start_time)
            if self.include_spans
            else []
        )
        span_errors: list[JsonValue] = [
            {"id": s.id, "name": s.name, "kind": s.kind.value, "error": clip(s.error or "", 500)}
            for s in failed[:MAX_LISTED_SPANS]
        ]
        details: dict[str, JsonValue] = {
            "include_spans": self.include_spans,
            "trace_error": trace.error,
        }
        if self.include_spans:
            details["span_error_count"] = len(failed)
            details["span_errors"] = span_errors
        parts: list[str] = []
        if trace.error is not None:
            parts.append(f"the run raised {clip(trace.error, 200)}")
        if failed:
            first = failed[0]
            noun = "span" if len(failed) == 1 else "spans"
            parts.append(
                f"{len(failed)} {noun} recorded an error (first: {first.name} "
                f"[{first.kind.value}]: {clip(first.error or '', 200)})"
            )
        passed = not parts
        if passed:
            scope = "the run and its spans" if self.include_spans else "the run"
            reason = f"no errors in {scope}"
        else:
            reason = "; ".join(parts)
        return EvalResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=reason,
            details=details,
        )

    def __repr__(self) -> str:
        return f"NoError(include_spans={self.include_spans!r})"


# -- tool_sequence -------------------------------------------------------------

ToolSequenceMode = Literal["strict", "ordered", "subset"]
"""How ``tool_sequence`` compares the expected tools with the calls made."""

MAX_LISTED_TOOLS = 20
"""How many tool names a ``tool_sequence`` reason quotes (``details`` keep them all)."""


def _check_tools(where: str, value: object) -> list[str]:
    if (
        isinstance(value, str)
        or not isinstance(value, Sequence)
        or not all(isinstance(t, str) and t for t in value)
    ):
        raise EvaluatorConfigError(f"{where} must be a list of tool names, got {value!r}")
    return list(value)


def _lcs_matches(expected: Sequence[str], actual: Sequence[str]) -> list[int]:
    """Indexes into ``expected`` of one longest common subsequence with ``actual``.

    Classic dynamic programme, ``O(len(expected) * len(actual))``; ties prefer the
    earliest expected items, so the reported "missing" tools are the later ones.
    """
    n, m = len(expected), len(actual)
    # suffix[i][j] = LCS length of expected[i:] and actual[j:]
    suffix = [[0] * (m + 1) for _ in range(n + 1)]
    for i in range(n - 1, -1, -1):
        for j in range(m - 1, -1, -1):
            if expected[i] == actual[j]:
                suffix[i][j] = suffix[i + 1][j + 1] + 1
            else:
                suffix[i][j] = max(suffix[i + 1][j], suffix[i][j + 1])
    matched: list[int] = []
    i = j = 0
    while i < n and j < m:
        if expected[i] == actual[j]:
            matched.append(i)
            i += 1
            j += 1
        elif suffix[i][j + 1] >= suffix[i + 1][j]:
            j += 1  # skipping this call loses nothing: keep expected[i] in play
        else:
            i += 1
    return matched


def _tool_list(tools: Sequence[str]) -> str:
    shown = ", ".join(tools[:MAX_LISTED_TOOLS])
    more = f", ... +{len(tools) - MAX_LISTED_TOOLS} more" if len(tools) > MAX_LISTED_TOOLS else ""
    return f"[{shown}{more}]"


class ToolSequence:
    """``tool_sequence``: the agent called the expected tools (FR-E1).

    The actual calls are :attr:`Trace.tool_calls` (names of ``tool_call`` spans,
    in start order). The expected tools are the ``tools`` parameter, else
    ``case.expected["tools"]`` (``expected: {tools: [lookup_order, issue_refund]}``
    with ``{tool_sequence: {mode: ordered}}``); with neither, ``evaluate`` raises
    :class:`~agentlens.evals.EvaluatorConfigError`.

    Modes (repeats always count — ``[search, search]`` needs two ``search`` calls):

    * ``strict`` — the calls are exactly the expected list. Score =
      LCS / max(len(expected), len(actual)), so extra, missing and reordered
      calls all lower it, and it is ``1`` only on an exact match.
    * ``ordered`` (default) — the expected list is a *subsequence* of the calls:
      in that order, other calls allowed in between. Score = LCS / len(expected).
    * ``subset`` — every expected call happened, in any order (multiset
      inclusion). Score = matched / len(expected), where each call can match
      one expected entry.

    ``LCS`` is the longest common subsequence of the two lists. An empty expected
    list passes ``ordered``/``subset`` trivially and, in ``strict``, means "no tool
    calls". Optional constraints, which fail the check when violated:

    * ``allow_extra=False`` (``ordered``/``subset``) — no call to a tool *outside*
      the expected list (retries of expected tools are fine); each such call is
      added to the score's denominator. ``strict`` already forbids extras.
    * ``forbidden`` — tools that must never be called; any call scores ``0``.

    ``details`` hold ``mode``, ``source``, ``expected``, ``actual``, ``missing``
    (expected entries not matched), ``unexpected`` (calls to tools not in the
    expected list) and ``forbidden_called``.

    Args:
        tools: Expected tool names; defaults to ``case.expected["tools"]``.
        mode: ``"strict"``, ``"ordered"`` or ``"subset"``.
        allow_extra: Allow calls to tools that are not expected.
        forbidden: Tools that must not be called at all.
    """

    name: str = "tool_sequence"

    def __init__(
        self,
        tools: Sequence[str] | None = None,
        *,
        mode: ToolSequenceMode = "ordered",
        allow_extra: bool = True,
        forbidden: Sequence[str] | None = None,
    ) -> None:
        self.tools = None if tools is None else _check_tools("tools", tools)
        self.mode = check_choice("mode", mode, ("strict", "ordered", "subset"))
        self.allow_extra = check_bool("allow_extra", allow_extra)
        self.forbidden = [] if forbidden is None else _check_tools("forbidden", forbidden)

    def _expected(self, case: Case) -> tuple[list[str], str]:
        if self.tools is not None:
            return self.tools, "tools"
        expected = case.expected
        if isinstance(expected, dict) and "tools" in expected:
            where = f"case {case.id!r} expected.tools"
            return _check_tools(where, expected["tools"]), "case.expected.tools"
        raise EvaluatorConfigError(
            f"tool_sequence has no expected tools for case {case.id!r}: set the 'tools' "
            "parameter or the case's expected.tools"
        )

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Compare :attr:`Trace.tool_calls` with the expected tools (see the class docstring)."""
        expected, source = self._expected(case)
        actual = trace.tool_calls

        if self.mode == "subset":
            remaining = Counter(actual)
            missing: list[str] = []
            for tool in expected:
                if remaining[tool] > 0:
                    remaining[tool] -= 1
                else:
                    missing.append(tool)
            n_matched = len(expected) - len(missing)
        else:
            hits = set(_lcs_matches(expected, actual))
            missing = [t for i, t in enumerate(expected) if i not in hits]
            n_matched = len(hits)

        expected_set = set(expected)
        unexpected = [t for t in actual if t not in expected_set]
        forbidden_set = set(self.forbidden)
        forbidden_called = [t for t in actual if t in forbidden_set]

        if self.mode == "strict":
            denominator = max(len(expected), len(actual))
            in_shape = actual == expected
        else:
            denominator = len(expected)
            in_shape = not missing
            if not self.allow_extra:
                denominator += len(unexpected)
                in_shape = in_shape and not unexpected
        score = n_matched / denominator if denominator else 1.0
        if forbidden_called:
            score = 0.0
        passed = in_shape and not forbidden_called

        reason = self._reason(expected, actual, missing, unexpected, forbidden_called, passed)
        details: dict[str, JsonValue] = {
            "mode": self.mode,
            "source": source,
            "expected": list(expected),
            "actual": list(actual),
            "missing": list(missing),
            "unexpected": list(unexpected),
            "forbidden_called": list(forbidden_called),
        }
        return EvalResult(
            evaluator=self.name, score=score, passed=passed, reason=reason, details=details
        )

    def _reason(
        self,
        expected: list[str],
        actual: list[str],
        missing: list[str],
        unexpected: list[str],
        forbidden_called: list[str],
        passed: bool,
    ) -> str:
        if passed:
            verdict = {
                "strict": "tool calls match exactly",
                "ordered": "expected tools called in order",
                "subset": "all expected tools called",
            }[self.mode]
        else:
            problems: list[str] = []
            if forbidden_called:
                problems.append(f"forbidden tools called: {_tool_list(forbidden_called)}")
            if missing:
                problems.append(f"missing {_tool_list(missing)}")
            if unexpected and (self.mode == "strict" or not self.allow_extra):
                problems.append(f"unexpected {_tool_list(unexpected)}")
            if not problems:  # strict: same multiset of calls, different order or count
                problems.append("calls differ from the expected sequence")
            verdict = "; ".join(problems)
        return (
            f"{verdict} ({self.mode}): expected {_tool_list(expected)}, actual {_tool_list(actual)}"
        )

    def __repr__(self) -> str:
        return params_repr(
            "ToolSequence",
            {
                "tools": self.tools,
                "mode": self.mode,
                "allow_extra": self.allow_extra,
                "forbidden": self.forbidden,
            },
        )


# -- budgets -------------------------------------------------------------------


def _budget_result(
    name: str, *, actual: float, limit: float, reason: str, details: dict[str, JsonValue]
) -> EvalResult:
    """Passed within ``limit``; score ``1`` then, else ``limit / actual``.

    ``reason`` may use ``{verb}`` (``"within"``/``"over"``).
    """
    passed = actual <= limit
    return EvalResult(
        evaluator=name,
        score=1.0 if passed else limit / actual,
        passed=passed,
        reason=reason.format(verb="within" if passed else "over"),
        details=details,
    )


class MaxSteps:
    """``max_steps``: the agent took at most ``value`` steps (``{max_steps: 8}``).

    Steps are :attr:`Trace.n_steps`: ``agent_step`` spans when the agent records
    them, otherwise the number of LLM calls. Score ``1`` within the limit, else
    ``value / steps`` (``0`` for ``value=0``); passes only within the limit.

    Args:
        value: Maximum number of steps (an integer ``>= 0``).
    """

    name: str = "max_steps"

    def __init__(self, value: int) -> None:
        self.value = int(check_number("max_steps value", value, integer=True))

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Compare :attr:`Trace.n_steps` with the limit."""
        steps = trace.n_steps
        counted = "agent_step" if trace.spans_of(SpanKind.AGENT_STEP) else "llm_call"
        noun = "step" if steps == 1 else "steps"
        return _budget_result(
            self.name,
            actual=steps,
            limit=self.value,
            reason=f"{steps} {noun} (counting {counted} spans), {{verb}} the limit of {self.value}",
            details={"steps": steps, "max_steps": self.value, "counted": counted},
        )

    def __repr__(self) -> str:
        return f"MaxSteps(value={self.value!r})"


UnknownCostPolicy = Literal["fail", "pass"]
"""What ``cost_budget`` does when the cost cannot be known."""


def _usd(value: float) -> str:
    return f"${value:.6g}"


class CostBudget:
    """``cost_budget``: the run cost at most ``max_usd`` (``{cost_budget: 0.05}``).

    The cost is :attr:`Trace.total_cost`, the sum of ``usage.cost_usd`` over the
    spans (filled from the pricing table; :class:`~agentlens.llm.MockLLM` reports
    ``0``). Score ``1`` within budget, else ``max_usd / cost``.

    **Unknown cost.** When no span has a cost (an unpriced model), or some spans
    with usage lack one, the budget cannot be verified. If the known part already
    exceeds the budget the check fails anyway; otherwise ``on_unknown`` decides:
    ``"fail"`` (default — a budget you cannot check is not a budget you met, and
    a model missing from the pricing table should be noticed) or ``"pass"``
    (score ``1``, with the reason saying the cost is unknown).

    Args:
        max_usd: Budget in US dollars (``>= 0``).
        on_unknown: ``"fail"`` or ``"pass"`` when the cost is unknown or partial.
    """

    name: str = "cost_budget"

    def __init__(self, max_usd: float, *, on_unknown: UnknownCostPolicy = "fail") -> None:
        self.max_usd = float(check_number("max_usd", max_usd))
        self.on_unknown = check_choice("on_unknown", on_unknown, ("fail", "pass"))

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Compare :attr:`Trace.total_cost` with the budget."""
        cost = trace.total_cost
        unpriced = sum(1 for s in trace.spans if s.usage is not None and s.usage.cost_usd is None)
        details: dict[str, JsonValue] = {
            "cost_usd": cost,
            "max_usd": self.max_usd,
            "unpriced_spans": unpriced,
            "on_unknown": self.on_unknown,
        }
        if cost is not None and (unpriced == 0 or cost > self.max_usd):
            partial = f" (a lower bound: {unpriced} spans have no cost)" if unpriced else ""
            return _budget_result(
                self.name,
                actual=cost,
                limit=self.max_usd,
                reason=f"cost {_usd(cost)}{partial} is {{verb}} the budget of {_usd(self.max_usd)}",
                details=details,
            )

        if cost is None:
            what = "cost unknown (no span has a cost; is the model in the pricing table?)"
        else:
            what = f"cost incomplete: {_usd(cost)} known, {unpriced} spans without a cost"
        passed = self.on_unknown == "pass"
        outcome = "passing (on_unknown: pass)" if passed else "failing (on_unknown: fail)"
        return EvalResult(
            evaluator=self.name,
            score=1.0 if passed else 0.0,
            passed=passed,
            reason=f"{what}; budget {_usd(self.max_usd)} not verified, {outcome}",
            details=details,
        )

    def __repr__(self) -> str:
        return f"CostBudget(max_usd={self.max_usd!r}, on_unknown={self.on_unknown!r})"


class LatencyBudget:
    """``latency_budget``: the run took at most ``max_ms`` ms (``{latency_budget: 5000}``).

    The latency is :attr:`Trace.latency_ms` (wall clock from start to end of the
    run). A run that has not ended (``latency_ms is None``) fails with score ``0``.
    Score ``1`` within budget, else ``max_ms / latency``.

    Args:
        max_ms: Budget in milliseconds (``>= 0``).
    """

    name: str = "latency_budget"

    def __init__(self, max_ms: float) -> None:
        self.max_ms = float(check_number("max_ms", max_ms))

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Compare :attr:`Trace.latency_ms` with the budget."""
        latency = trace.latency_ms
        details: dict[str, JsonValue] = {"latency_ms": latency, "max_ms": self.max_ms}
        if latency is None:
            return EvalResult(
                evaluator=self.name,
                score=0.0,
                passed=False,
                reason="latency unknown: the run has not ended (no end_time)",
                details=details,
            )
        return _budget_result(
            self.name,
            actual=latency,
            limit=self.max_ms,
            reason=f"latency {latency:.1f} ms is {{verb}} the budget of {self.max_ms:g} ms",
            details=details,
        )

    def __repr__(self) -> str:
        return f"LatencyBudget(max_ms={self.max_ms!r})"
