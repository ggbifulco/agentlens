"""The :class:`Evaluator` protocol and the normalisation of evaluator return values."""

from __future__ import annotations

import math
from collections.abc import Awaitable, Mapping
from typing import Any, Protocol, runtime_checkable

from pydantic import ValidationError

from agentlens.schema import Case, EvalResult, Trace

__all__ = [
    "DEFAULT_THRESHOLD",
    "AsyncEvaluator",
    "Evaluator",
    "error_result",
    "to_eval_result",
]

DEFAULT_THRESHOLD = 0.5
"""Pass threshold for evaluators that return a bare score: ``passed = score >= threshold``."""


@runtime_checkable
class Evaluator(Protocol):
    """Anything that scores one trace of one case (``docs/design.md`` §5).

    ``name`` is what ends up in :attr:`EvalResult.evaluator`; it is declared as a
    read-only property so both a plain attribute and a ``@property`` satisfy the
    protocol. ``evaluate`` should return an :class:`EvalResult`; exceptions are
    turned into failed results by :func:`~agentlens.evals.run_evaluator`.
    """

    @property
    def name(self) -> str:
        """Evaluator name, as used in suite files and results."""
        ...

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Score ``trace`` (the agent's run on ``case``)."""
        ...


@runtime_checkable
class AsyncEvaluator(Evaluator, Protocol):
    """An evaluator with a native async path (e.g. an LLM judge doing network I/O).

    :func:`~agentlens.evals.arun_evaluator` awaits ``aevaluate`` when an evaluator
    has it and otherwise runs the sync ``evaluate`` in a worker thread.
    """

    def aevaluate(self, trace: Trace, case: Case) -> Awaitable[EvalResult]:
        """Async version of :meth:`evaluate`."""
        ...


def to_eval_result(
    name: str,
    value: Any,
    *,
    threshold: float = DEFAULT_THRESHOLD,
    reason: str | None = None,
) -> EvalResult:
    """Normalise what a custom evaluator returned into an :class:`EvalResult`.

    Accepted values:

    * an :class:`EvalResult` — returned unchanged;
    * ``bool`` — score ``1.0``/``0.0``, ``passed`` = the value;
    * ``int``/``float`` in ``[0, 1]`` — the score, ``passed = score >= threshold``;
    * ``(value, reason)`` — any of the above plus a reason string (or ``None``);
    * a mapping with ``score`` and/or ``passed`` (plus optional ``reason``,
      ``details``) — a missing ``passed`` is derived from ``score`` and
      ``threshold``, a missing ``score`` from ``passed``.

    Raises:
        TypeError: for any other type (including ``None``).
        ValueError: for a score outside ``[0, 1]`` or ``NaN``, or an invalid mapping.
    """
    if isinstance(value, EvalResult):
        return value
    if isinstance(value, bool):
        return EvalResult(evaluator=name, score=float(value), passed=value, reason=reason)
    if isinstance(value, int | float):
        score = _check_score(value)
        return EvalResult(evaluator=name, score=score, passed=score >= threshold, reason=reason)
    if isinstance(value, tuple):
        if len(value) != 2 or not (value[1] is None or isinstance(value[1], str)):
            raise TypeError("a tuple result must be (score_or_bool, reason: str | None)")
        return to_eval_result(name, value[0], threshold=threshold, reason=value[1])
    if isinstance(value, Mapping):
        return _from_mapping(name, value, threshold=threshold, reason=reason)
    raise TypeError(
        f"evaluator returned {type(value).__name__}; expected EvalResult, bool, a score "
        "in [0, 1], (score, reason) or a mapping with 'score'/'passed'"
    )


def _check_score(value: float) -> float:
    score = float(value)
    if math.isnan(score) or not 0.0 <= score <= 1.0:
        raise ValueError(f"score must be in [0, 1], got {value!r}")
    return score


def _from_mapping(
    name: str, value: Mapping[Any, Any], *, threshold: float, reason: str | None
) -> EvalResult:
    data = dict(value)
    if "score" not in data and "passed" not in data:
        raise ValueError("a mapping result needs 'score' and/or 'passed'")
    if "score" not in data:
        data["score"] = 1.0 if data["passed"] else 0.0
    elif "passed" not in data:
        score = data["score"]
        if isinstance(score, bool) or not isinstance(score, int | float):
            raise TypeError(f"'score' must be a number, got {type(score).__name__}")
        data["passed"] = _check_score(score) >= threshold
    data.setdefault("evaluator", name)
    if reason is not None:
        data.setdefault("reason", reason)
    try:
        return EvalResult.model_validate(data)
    except ValidationError as exc:
        raise ValueError(f"invalid result mapping: {exc}") from exc


def error_result(name: str, exc: BaseException) -> EvalResult:
    """The failed :class:`EvalResult` that stands in for an evaluator that raised.

    Score ``0``, ``passed=False``, ``reason`` = ``"evaluator error: <Type>: <message>"``
    and ``details["error"] = {"type": ..., "message": ...}`` so reports can tell an
    evaluator crash from a genuine failure.
    """
    message = str(exc)
    return EvalResult(
        evaluator=name,
        score=0.0,
        passed=False,
        reason=f"evaluator error: {type(exc).__name__}: {message}",
        details={"error": {"type": type(exc).__name__, "message": message}},
    )
