"""Errors raised while registering, configuring or building evaluators.

Configuration problems (unknown name, bad parameters, duplicate registration)
are raised eagerly so a broken suite file fails at load time. Errors raised
*while evaluating* a trace never propagate: they become a failed
:class:`~agentlens.schema.EvalResult` (see :func:`agentlens.evals.run_evaluator`).
"""

from __future__ import annotations

__all__ = [
    "DuplicateEvaluatorError",
    "EvaluatorConfigError",
    "EvaluatorError",
    "UnknownEvaluatorError",
]


class EvaluatorError(Exception):
    """Base class for every evaluator configuration error."""


class UnknownEvaluatorError(EvaluatorError, LookupError):
    """No evaluator is registered under the requested name."""

    def __init__(self, name: str, available: list[str], suggestions: list[str]) -> None:
        self.name = name
        self.available = available
        self.suggestions = suggestions
        hint = f" Did you mean {' or '.join(repr(s) for s in suggestions)}?" if suggestions else ""
        listed = ", ".join(available) if available else "none"
        super().__init__(f"unknown evaluator {name!r}.{hint} Available: {listed}.")


class DuplicateEvaluatorError(EvaluatorError, ValueError):
    """An evaluator with this name is already registered (pass ``replace=True`` to override)."""


class EvaluatorConfigError(EvaluatorError, ValueError):
    """An evaluator spec is malformed or its parameters do not fit the factory."""
