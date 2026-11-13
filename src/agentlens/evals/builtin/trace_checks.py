"""Deterministic checks on how the run went (not on its output): ``no_error`` (FR-E1)."""

from __future__ import annotations

from pydantic import JsonValue

from agentlens.evals.builtin._common import check_bool, clip
from agentlens.schema import Case, EvalResult, Trace

__all__ = ["NoError"]

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
