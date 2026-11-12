"""Trace and span models: the recorded shape of one agent run."""

from __future__ import annotations

from enum import StrEnum
from typing import Self

from pydantic import Field, JsonValue, NonNegativeFloat, NonNegativeInt, model_validator

from agentlens.schema._base import SchemaModel, UtcDatetime, utc_now
from agentlens.schema.ids import new_id

__all__ = ["Span", "SpanKind", "Trace", "Usage"]


class SpanKind(StrEnum):
    """What a span represents (FR-T2)."""

    AGENT_STEP = "agent_step"
    LLM_CALL = "llm_call"
    TOOL_CALL = "tool_call"
    RETRIEVAL = "retrieval"
    CUSTOM = "custom"


class Usage(SchemaModel):
    """Token usage and estimated cost of an LLM call."""

    input_tokens: NonNegativeInt = 0
    output_tokens: NonNegativeInt = 0
    cost_usd: NonNegativeFloat | None = None
    """Estimated cost; ``None`` when the model is missing from the pricing table."""

    @property
    def total_tokens(self) -> int:
        """Input plus output tokens."""
        return self.input_tokens + self.output_tokens


class Span(SchemaModel):
    """One timed unit of work inside a trace: an LLM call, a tool call, a step...

    Spans form a tree through ``parent_id``; root spans have ``parent_id=None``.
    """

    id: str = Field(default_factory=new_id)
    trace_id: str
    parent_id: str | None = None
    kind: SpanKind
    name: str
    """For ``TOOL_CALL`` spans this is the tool name; for ``LLM_CALL`` usually the model."""
    start_time: UtcDatetime = Field(default_factory=utc_now)
    end_time: UtcDatetime | None = None
    """``None`` while the span is still running."""
    input: JsonValue = None
    output: JsonValue = None
    error: str | None = None
    usage: Usage | None = None
    attributes: dict[str, JsonValue] = Field(default_factory=dict)
    """Free-form metadata: model, temperature, tool arguments schema, ..."""

    @model_validator(mode="after")
    def _check_times(self) -> Self:
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("span end_time is before start_time")
        if self.parent_id == self.id:
            raise ValueError("span cannot be its own parent")
        return self

    @property
    def latency_ms(self) -> float | None:
        """Wall-clock duration in milliseconds, or ``None`` if the span is unfinished."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000


class Trace(SchemaModel):
    """A complete agent run: its input, final output and the tree of spans.

    Derived metrics (``total_cost``, ``total_tokens``, ``latency_ms``,
    ``tool_calls``, ``n_steps``) are plain properties, *not* serialised fields:
    they are always recomputed from the spans, so a stored trace can never hold
    stale totals.
    """

    id: str = Field(default_factory=new_id)
    name: str
    input: JsonValue = None
    output: JsonValue = None
    error: str | None = None
    """Set when the agent raised; the run is still recorded (FR-T5)."""
    start_time: UtcDatetime = Field(default_factory=utc_now)
    end_time: UtcDatetime | None = None
    spans: list[Span] = Field(default_factory=list)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Run context: git sha, agent version, case id, repetition index, ..."""

    @model_validator(mode="after")
    def _check_consistency(self) -> Self:
        if self.end_time is not None and self.end_time < self.start_time:
            raise ValueError("trace end_time is before start_time")
        span_ids: set[str] = set()
        for span in self.spans:
            if span.trace_id != self.id:
                raise ValueError(f"span {span.id} has trace_id {span.trace_id}, expected {self.id}")
            if span.id in span_ids:
                raise ValueError(f"duplicate span id {span.id}")
            span_ids.add(span.id)
        for span in self.spans:
            if span.parent_id is not None and span.parent_id not in span_ids:
                raise ValueError(f"span {span.id} references unknown parent {span.parent_id}")
        return self

    # -- tree navigation -----------------------------------------------------

    def roots(self) -> list[Span]:
        """Spans without a parent, in start-time order."""
        return sorted((s for s in self.spans if s.parent_id is None), key=lambda s: s.start_time)

    def children(self, span_id: str) -> list[Span]:
        """Direct children of ``span_id``, in start-time order."""
        return sorted((s for s in self.spans if s.parent_id == span_id), key=lambda s: s.start_time)

    def spans_of(self, kind: SpanKind) -> list[Span]:
        """All spans of a given kind, in start-time order."""
        return sorted((s for s in self.spans if s.kind == kind), key=lambda s: s.start_time)

    # -- derived metrics -----------------------------------------------------

    @property
    def total_tokens(self) -> int:
        """Sum of input and output tokens over all spans with usage."""
        return sum(s.usage.total_tokens for s in self.spans if s.usage is not None)

    @property
    def total_cost(self) -> float | None:
        """Sum of known span costs in USD.

        ``None`` when no span carries a cost (unknown is not the same as free).
        """
        usages = [s.usage for s in self.spans if s.usage is not None]
        costs = [u.cost_usd for u in usages if u.cost_usd is not None]
        return sum(costs) if costs else None

    @property
    def latency_ms(self) -> float | None:
        """Wall-clock duration of the whole run, or ``None`` if it has not ended."""
        if self.end_time is None:
            return None
        return (self.end_time - self.start_time).total_seconds() * 1000

    @property
    def tool_calls(self) -> list[str]:
        """Names of the tools called, in call order (used by ``tool_sequence``)."""
        return [s.name for s in self.spans_of(SpanKind.TOOL_CALL)]

    @property
    def n_steps(self) -> int:
        """Number of agent steps.

        Counts ``AGENT_STEP`` spans when the agent records them; otherwise falls
        back to the number of LLM calls, since each LLM call is one turn of the
        agent loop.
        """
        steps = len(self.spans_of(SpanKind.AGENT_STEP))
        return steps if steps else len(self.spans_of(SpanKind.LLM_CALL))
