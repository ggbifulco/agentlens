"""Dataset models: test cases and the evaluator specs attached to them."""

from __future__ import annotations

from typing import Any

from pydantic import Field, JsonValue, model_serializer, model_validator

from agentlens.schema._base import SchemaModel

__all__ = ["Case", "EvaluatorSpec"]


class EvaluatorSpec(SchemaModel):
    """A reference to an evaluator plus its parameters.

    Accepts the three spellings used in suite YAML (see ``docs/design.md`` §6)::

        no_error                          # bare name
        {max_steps: 8}                    # name -> single scalar parameter
        {tool_sequence: {mode: ordered}}  # name -> parameter mapping

    as well as the canonical ``{name: ..., params: {...}}`` form. A scalar
    parameter is stored as ``params={"value": 8}``. Specs serialise back to the
    shortest YAML-style form, so a suite file round-trips unchanged.
    """

    name: str = Field(min_length=1)
    params: dict[str, JsonValue] = Field(default_factory=dict)

    @model_validator(mode="before")
    @classmethod
    def _parse_shorthand(cls, data: Any) -> Any:
        if isinstance(data, str):
            return {"name": data}
        if isinstance(data, dict) and len(data) == 1:
            ((key, value),) = data.items()
            if key == "name" and isinstance(value, str):
                return data  # canonical form without params
            if key in {"name", "params"}:
                return data  # malformed canonical form: let field validation report it
            if isinstance(value, dict):
                return {"name": key, "params": value}
            if value is None:
                return {"name": key}
            return {"name": key, "params": {"value": value}}
        return data

    @model_serializer(mode="plain")
    def _serialise_short(self) -> str | dict[str, JsonValue]:
        if not self.params:
            return self.name
        if set(self.params) == {"value"}:
            value = self.params["value"]
            # Mappings and None would parse back differently, so keep them explicit.
            if value is not None and not isinstance(value, dict):
                return {self.name: value}
        return {self.name: dict(self.params)}


class Case(SchemaModel):
    """One dataset row: an input to feed the agent and what we expect back."""

    id: str = Field(min_length=1)
    input: JsonValue
    expected: JsonValue = None
    evaluators: list[EvaluatorSpec] = Field(default_factory=list)
    """Evaluators for this case, applied on top of the suite defaults."""
    tags: list[str] = Field(default_factory=list)
