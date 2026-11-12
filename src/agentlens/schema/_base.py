"""Shared base model and field types for the schema package."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Annotated

from pydantic import AfterValidator, AwareDatetime, BaseModel, ConfigDict


def _to_utc(value: datetime) -> datetime:
    return value.astimezone(UTC)


UtcDatetime = Annotated[AwareDatetime, AfterValidator(_to_utc)]
"""A timezone-aware datetime, normalised to UTC. Naive datetimes are rejected,
because "local time on the machine that recorded the trace" is ambiguous once a
trace is exported and compared elsewhere."""


def utc_now() -> datetime:
    """Current time as an aware UTC datetime."""
    return datetime.now(UTC)


class SchemaModel(BaseModel):
    """Base class for every AgentLens schema model.

    * ``extra="forbid"`` turns typos in YAML/JSON (``evalutors:``) into errors
      instead of silently dropping data.
    * ``validate_assignment=True`` keeps invariants when the recorder mutates a
      model after creation (e.g. setting ``span.end_time``).
    """

    model_config = ConfigDict(extra="forbid", validate_assignment=True)
