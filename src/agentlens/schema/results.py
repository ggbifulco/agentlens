"""Evaluation results and experiments."""

from __future__ import annotations

from pydantic import Field, JsonValue

from agentlens.schema._base import SchemaModel, UtcDatetime, utc_now
from agentlens.schema.ids import new_id

__all__ = ["EvalResult", "Experiment"]


class EvalResult(SchemaModel):
    """The verdict of one evaluator on one trace."""

    evaluator: str
    score: float = Field(ge=0.0, le=1.0)
    """Normalised to ``[0, 1]`` so different evaluators can be averaged and compared
    (a 1-5 rubric score of 4 becomes 0.75)."""
    passed: bool
    reason: str | None = None
    details: dict[str, JsonValue] = Field(default_factory=dict)


class Experiment(SchemaModel):
    """One execution of a suite: the unit that gets compared against a baseline."""

    id: str = Field(default_factory=new_id)
    name: str
    suite: str
    created_at: UtcDatetime = Field(default_factory=utc_now)
    config: dict[str, JsonValue] = Field(default_factory=dict)
    """What was run: model, prompt version, git sha, ..."""
    is_baseline: bool = False
