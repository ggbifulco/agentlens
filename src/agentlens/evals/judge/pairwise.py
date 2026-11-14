"""Pairwise A/B judge: which of two outputs for the same case is better (FR-E2).

A pairwise judge compares two runs of the same case — typically a *baseline*
trace (A) and a *candidate* trace (B) — instead of scoring one trace, so it is
not an :class:`~agentlens.evals.Evaluator`. It has its own small API::

    from agentlens.evals import PairwiseJudge

    judge = PairwiseJudge("Answers the question and confirms the refund amount",
                          model="openai:gpt-4o-mini")
    result = judge.compare(baseline_trace, candidate_trace, case)
    result.winner        # "a", "b" or "tie"
    result.preference    # 1.0 = A better, 0.0 = B better, 0.5 = tie
    result.consistent    # did both presentation orders agree?
    result.to_eval_result(candidate="b")   # passed if B is not worse than A

**Position bias.** LLM judges prefer the answer shown first (or second)
regardless of quality. Every comparison therefore asks the judge twice: once
with A shown as output 1 (order ``"ab"``) and once with B shown first (order
``"ba"``). The positional answer of the swapped order is mapped back to A/B and
the two preferences (A better = 1, tie = 0.5, B better = 0) are averaged. When
the orders disagree the result is ``consistent=False`` with a lower
``confidence``; opposite answers average to exactly 0.5, a tie.

Each order is an ordinary judge call of :class:`~agentlens.evals.judge.BaseJudge`
(temperature 0, JSON mode, parse + retry) cached under its own key. The cached
``output`` is the *ordered* pair ``[output 1, output 2]`` as shown to the judge,
so the two orders of a pair have different keys, while the same ordered pair
reuses its verdict — also when ``compare`` is later called with A and B swapped.
"""

from __future__ import annotations

import asyncio
import json
from dataclasses import dataclass
from typing import Literal, TypeAlias

from pydantic import JsonValue

from agentlens.evals.errors import EvaluatorConfigError
from agentlens.evals.judge.base import RAW_REPLY_CHARS, BaseJudge, JudgePrompt, JudgeRun, value_text
from agentlens.evals.judge.cache import CacheSpec
from agentlens.evals.judge.parse import JudgeParseError, parse_reasoning
from agentlens.evals.judge.prompts import check_template, load_template, render_template
from agentlens.llm import LLMClient
from agentlens.schema import Case, EvalResult, Trace

__all__ = [
    "ORDERS",
    "PAIRWISE_NAME",
    "PAIRWISE_PLACEHOLDERS",
    "Order",
    "OrderVerdict",
    "PairwiseJudge",
    "PairwiseResult",
    "PairwiseVerdict",
    "Position",
    "Winner",
    "parse_position",
]

Winner: TypeAlias = Literal["a", "b", "tie"]
"""Which trace is better: ``"a"`` (the first argument of ``compare``), ``"b"`` or a tie."""

Position: TypeAlias = Literal["1", "2", "tie"]
"""The judge's raw answer: the output shown first (``"1"``), second (``"2"``) or a tie."""

Order: TypeAlias = Literal["ab", "ba"]
"""Presentation order: ``"ab"`` shows A as output 1, ``"ba"`` shows B as output 1."""

ORDERS: tuple[Order, Order] = ("ab", "ba")

PAIRWISE_NAME = "pairwise"
"""The judge's name: ``evaluator`` of :meth:`PairwiseResult.to_eval_result`."""

PAIRWISE_PLACEHOLDERS = ("criteria", "input", "output_1", "output_2", "reference")
"""Placeholders a custom pairwise ``template`` may use."""

_WINNER_KEYS = ("winner", "choice", "better", "preference", "verdict")
_FIRST = frozenset({"1", "output 1", "output_1", "output1", "response 1", "first", "#1"})
_SECOND = frozenset({"2", "output 2", "output_2", "output2", "response 2", "second", "#2"})
_TIE = frozenset({"tie", "draw", "equal", "same", "both", "neither", "none", "0"})

_PREFERENCE: dict[Winner, float] = {"a": 1.0, "tie": 0.5, "b": 0.0}


def parse_position(data: dict[str, JsonValue]) -> Position:
    """Read the judge's choice (``winner``: ``"1"``, ``"2"`` or ``"tie"``) from its JSON.

    Keys ``winner``/``choice``/``better``/``preference``/``verdict`` are matched
    case-insensitively; values ``1``/``2`` (numbers or strings), ``"output 1"``,
    ``"first"``/``"second"`` and ``"tie"`` (also draw/equal/neither) are accepted.

    Raises:
        JudgeParseError: if no choice key is present or its value is not recognised.
    """
    lowered = {key.lower(): key for key in data}
    key = next((lowered[k] for k in _WINNER_KEYS if k in lowered), None)
    if key is None:
        raise JudgeParseError("the judge's JSON has no 'winner' field")
    raw = data[key]
    if isinstance(raw, int) and not isinstance(raw, bool) and raw in (1, 2):
        return "1" if raw == 1 else "2"
    if isinstance(raw, str):
        word = raw.strip().strip(".!\"'").strip().lower()
        if word in _FIRST:
            return "1"
        if word in _SECOND:
            return "2"
        if word in _TIE:
            return "tie"
    raise JudgeParseError(f"{key!r} must be '1', '2' or 'tie', got {raw!r}")


def _to_winner(position: Position, order: Order) -> Winner:
    """Map a positional answer back to A/B for the given presentation order."""
    if position == "tie":
        return "tie"
    first_is_a = order == "ab"
    return "a" if (position == "1") == first_is_a else "b"


@dataclass(frozen=True)
class PairwiseVerdict:
    """A parsed pairwise reply: the positional choice and the judge's explanation."""

    position: Position
    reasoning: str | None


@dataclass(frozen=True)
class OrderVerdict:
    """The judge's answer for one presentation order, mapped back to A/B.

    ``position``/``winner`` are ``None`` when the reply could not be parsed
    (``error`` and ``raw_reply`` then say why).
    """

    order: Order
    position: Position | None
    winner: Winner | None
    reasoning: str | None
    judge: dict[str, JsonValue]
    """Call metadata: ``judge_model``, ``cached``, ``attempts``, ``cache_key``, ``judge_usage``."""
    error: str | None = None
    raw_reply: str | None = None

    @property
    def preference(self) -> float | None:
        """This order's preference for A: 1 (A better), 0.5 (tie), 0 (B better)."""
        return None if self.winner is None else _PREFERENCE[self.winner]

    def to_dict(self) -> dict[str, JsonValue]:
        """JSON form, as stored in ``details["orders"]``."""
        data: dict[str, JsonValue] = {
            "order": self.order,
            "position": self.position,
            "winner": self.winner,
            "reasoning": self.reasoning,
            **self.judge,
        }
        if self.error is not None:
            data["error"] = self.error
            data["raw_reply"] = self.raw_reply
        return data

    @classmethod
    def from_run(cls, order: Order, run: JudgeRun[PairwiseVerdict]) -> OrderVerdict:
        """Build from a finished judge call for ``order``."""
        if run.verdict is None:
            error = run.error or JudgeParseError("the judge gave no verdict")
            raw = run.raw_reply or ""
            if len(raw) > RAW_REPLY_CHARS:
                raw = raw[:RAW_REPLY_CHARS] + "…"
            return cls(order, None, None, None, run.details(), error=str(error), raw_reply=raw)
        verdict = run.verdict
        return cls(
            order,
            verdict.position,
            _to_winner(verdict.position, order),
            verdict.reasoning,
            run.details(),
        )


@dataclass(frozen=True)
class PairwiseResult:
    """The outcome of comparing trace A with trace B over both presentation orders.

    Attributes:
        winner: ``"a"``, ``"b"`` or ``"tie"`` (from :attr:`preference`).
        preference: Mean preference for A over the two orders: 1 = A better in
            both, 0.5 = tie (or opposite answers), 0 = B better in both.
        consistent: Both orders gave the same (mapped) winner.
        confidence: Agreement between the orders, ``1 - |pref_ab - pref_ba|``:
            1 when consistent, 0.5 when one order says tie, 0 when they are opposite.
        orders: The two :class:`OrderVerdict` (``"ab"`` then ``"ba"``).
        criteria: What the judge compared on.
        error: Set when a reply could not be parsed; the result is then a
            tie with confidence 0 and :meth:`to_eval_result` fails.
    """

    winner: Winner
    preference: float
    consistent: bool
    confidence: float
    orders: tuple[OrderVerdict, OrderVerdict]
    criteria: str
    error: str | None = None

    @classmethod
    def combine(cls, ab: OrderVerdict, ba: OrderVerdict, criteria: str) -> PairwiseResult:
        """Average the two orders' preferences (position-bias mitigation)."""
        orders = (ab, ba)
        if ab.preference is None or ba.preference is None:
            errors = "; ".join(f"order {o.order}: {o.error}" for o in orders if o.error)
            return cls("tie", 0.5, False, 0.0, orders, criteria, error=errors)
        preference = (ab.preference + ba.preference) / 2
        winner: Winner = "a" if preference > 0.5 else "b" if preference < 0.5 else "tie"
        return cls(
            winner=winner,
            preference=preference,
            consistent=ab.winner == ba.winner,
            confidence=1.0 - abs(ab.preference - ba.preference),
            orders=orders,
            criteria=criteria,
        )

    @property
    def ok(self) -> bool:
        """Both orders gave a parseable verdict."""
        return self.error is None

    @property
    def reasons(self) -> list[str]:
        """The judge's explanation for each order that gave one (``"ab: ..."``, ``"ba: ..."``)."""
        return [f"{o.order}: {o.reasoning}" for o in self.orders if o.reasoning]

    @property
    def reason(self) -> str:
        """One-line summary, e.g. ``"B is better (preference for A 0.00, consistent)"``."""
        if self.error is not None:
            return f"pairwise judge reply could not be parsed: {self.error}"
        head = "tie" if self.winner == "tie" else f"{self.winner.upper()} is better"
        if self.consistent:
            return f"{head} (preference for A {self.preference:.2f}, consistent across orders)"
        answers = ", ".join(f"{o.order} → {o.winner}" for o in self.orders)
        return (
            f"{head} (preference for A {self.preference:.2f}; the orders disagree: {answers}, "
            f"confidence {self.confidence:.2f} — possible position bias)"
        )

    def details(self) -> dict[str, JsonValue]:
        """JSON-ready details: winner, preference, consistency and both orders."""
        data: dict[str, JsonValue] = {
            "winner": self.winner,
            "preference": self.preference,
            "consistent": self.consistent,
            "confidence": self.confidence,
            "criteria": self.criteria,
            "orders": [o.to_dict() for o in self.orders],
        }
        if self.error is not None:
            data["error"] = {"type": "JudgeParseError", "message": self.error}
        return data

    def to_eval_result(
        self, candidate: Literal["a", "b"] = "b", *, name: str = PAIRWISE_NAME
    ) -> EvalResult:
        """The comparison as an :class:`EvalResult` for the ``candidate`` trace.

        Score = the candidate's preference (``1 - preference`` for B, ``preference``
        for A); passed when the candidate is not worse (it wins or ties). An
        unparseable comparison fails with score 0 and ``details["error"]``.

        Raises:
            ValueError: if ``candidate`` is not ``"a"`` or ``"b"``.
        """
        if candidate not in ("a", "b"):
            raise ValueError(f"candidate must be 'a' or 'b', got {candidate!r}")
        details = self.details()
        details["candidate"] = candidate
        if self.error is not None:
            return EvalResult(
                evaluator=name, score=0.0, passed=False, reason=self.reason, details=details
            )
        score = self.preference if candidate == "a" else 1.0 - self.preference
        passed = self.winner in (candidate, "tie")
        outcome = "wins" if self.winner == candidate else "ties" if passed else "loses"
        return EvalResult(
            evaluator=name,
            score=score,
            passed=passed,
            reason=f"candidate {candidate.upper()} {outcome}: {self.reason}",
            details=details,
        )


class PairwiseJudge(BaseJudge[PairwiseVerdict]):
    """An LLM judge that compares two traces of the same case on ``criteria`` (FR-E2).

    Not an evaluator (it needs two traces): call :meth:`compare` / :meth:`acompare`
    and, where an :class:`EvalResult` is needed, :meth:`PairwiseResult.to_eval_result`.
    Every comparison makes two judge calls (orders ``"ab"`` and ``"ba"``) — see the
    module docstring for how they are combined.

    Args:
        criteria: What makes one output better than the other.
        include_expected: Show ``case.expected`` to the judge as a reference.
        template: Custom prompt with ``$criteria``, ``$input``, ``$output_1``,
            ``$output_2`` and ``$reference`` (it must ask for ``winner``/``reasoning`` JSON).

    ``model``, ``client``, ``cache``, ``retries`` and ``max_tokens`` are as in
    :class:`~agentlens.evals.judge.BaseJudge`.

    Raises:
        EvaluatorConfigError: for invalid parameters.
    """

    name: str = PAIRWISE_NAME

    def __init__(
        self,
        criteria: str,
        *,
        include_expected: bool = True,
        template: str | None = None,
        model: str | None = None,
        client: LLMClient | None = None,
        cache: CacheSpec = None,
        retries: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model, client=client, cache=cache, retries=retries, max_tokens=max_tokens
        )
        if not isinstance(criteria, str) or not criteria.strip():
            raise EvaluatorConfigError(
                f"criteria must be a non-empty string describing what to compare, got {criteria!r}"
            )
        if not isinstance(include_expected, bool):
            raise EvaluatorConfigError(
                f"include_expected must be true or false, got {include_expected!r}"
            )
        self.criteria = criteria.strip()
        self.include_expected = include_expected
        self.template = (
            None if template is None else check_template(template, PAIRWISE_PLACEHOLDERS)
        )

    def build_prompt(
        self, trace_a: Trace, trace_b: Trace, case: Case | None, order: Order
    ) -> JudgePrompt:
        """Render the prompt for one presentation order.

        The input is ``trace_a.input`` (else ``trace_b.input``, else ``case.input``).
        The cached ``output`` is the ordered pair ``[output 1, output 2]``: the
        presentation order is part of the cache key.
        """
        first, second = (trace_a, trace_b) if order == "ab" else (trace_b, trace_a)
        agent_input = next(
            (t.input for t in (trace_a, trace_b) if t.input is not None),
            case.input if case is not None else None,
        )
        output_1, output_2 = value_text(first.output), value_text(second.output)
        reference = ""
        if self.include_expected and case is not None and case.expected is not None:
            reference = render_template(
                load_template("reference"), {"expected": value_text(case.expected)}
            )
        user = render_template(
            self.template if self.template is not None else load_template("pairwise"),
            {
                "criteria": self.criteria,
                "input": value_text(agent_input),
                "output_1": output_1,
                "output_2": output_2,
                "reference": reference,
            },
        )
        keyed = json.dumps([output_1, output_2], ensure_ascii=False)
        return JudgePrompt(system=load_template("judge_system"), user=user, output=keyed)

    def parse_reply(self, data: dict[str, JsonValue]) -> PairwiseVerdict:
        """Read ``winner`` (``"1"``/``"2"``/``"tie"``) and ``reasoning``."""
        return PairwiseVerdict(position=parse_position(data), reasoning=parse_reasoning(data))

    def compare(self, trace_a: Trace, trace_b: Trace, case: Case | None = None) -> PairwiseResult:
        """Compare ``trace_a`` (e.g. the baseline) with ``trace_b`` (the candidate).

        Asks the judge in both orders (or reads the cache) and averages. A reply that
        cannot be parsed gives a result with ``error`` set; LLM errors propagate.
        """
        ab, ba = (
            OrderVerdict.from_run(
                order, self.run_judge(self.build_prompt(trace_a, trace_b, case, order))
            )
            for order in ORDERS
        )
        return PairwiseResult.combine(ab, ba, self.criteria)

    async def acompare(
        self, trace_a: Trace, trace_b: Trace, case: Case | None = None
    ) -> PairwiseResult:
        """Async :meth:`compare`: the two orders are judged concurrently."""
        runs = await asyncio.gather(
            *(self.arun_judge(self.build_prompt(trace_a, trace_b, case, o)) for o in ORDERS)
        )
        ab, ba = (OrderVerdict.from_run(o, run) for o, run in zip(ORDERS, runs, strict=True))
        return PairwiseResult.combine(ab, ba, self.criteria)

    def __repr__(self) -> str:
        template = ", template='custom'" if self.template is not None else ""
        return (
            f"PairwiseJudge(criteria={self.criteria!r}, include_expected={self.include_expected}"
            f"{template}, model={self.model!r}, cache={self.cache!r}, retries={self.retries})"
        )
