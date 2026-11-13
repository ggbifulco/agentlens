"""Provider-neutral request/response models for LLM calls (``docs/design.md`` §3).

Every adapter (mock, OpenAI-compatible, Anthropic) translates to and from these
models, so judges, example agents and replay only ever see one shape (FR-M1).
All models are JSON-serialisable pydantic models; :meth:`LLMRequest.canonical_json`
gives a stable encoding that cassettes hash to recognise a request (ADR 0004).
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Literal, Self

from pydantic import (
    Field,
    JsonValue,
    NonNegativeFloat,
    PositiveInt,
    field_validator,
    model_validator,
)

from agentlens.schema import Usage
from agentlens.schema._base import SchemaModel

__all__ = ["LLMRequest", "LLMResponse", "Message", "Role", "ToolCall", "ToolSpec"]

Role = Literal["system", "user", "assistant", "tool"]
"""Who authored a :class:`Message`."""


class ToolCall(SchemaModel):
    """A request from the model to call one tool."""

    id: str
    """Provider-assigned id; the matching ``tool`` message echoes it as ``tool_call_id``."""
    name: str
    arguments: dict[str, JsonValue] = Field(default_factory=dict)
    """Parsed (not string-encoded) JSON arguments."""


def _empty_object_schema() -> dict[str, JsonValue]:
    return {"type": "object", "properties": {}}


class ToolSpec(SchemaModel):
    """A tool the model may call, described by a JSON Schema for its arguments."""

    name: str = Field(min_length=1)
    description: str = ""
    parameters: dict[str, JsonValue] = Field(default_factory=_empty_object_schema)
    """JSON Schema of the arguments object."""


class Message(SchemaModel):
    """One chat message.

    Invariants: only ``assistant`` messages carry ``tool_calls``; ``tool`` messages
    must name the call they answer in ``tool_call_id`` (and only they may);
    every role except ``assistant`` needs ``content`` (an assistant message may be
    tool calls only).
    """

    role: Role
    content: str | None = None
    tool_calls: list[ToolCall] = Field(default_factory=list)
    tool_call_id: str | None = None

    @model_validator(mode="after")
    def _check_role_fields(self) -> Self:
        if self.tool_calls and self.role != "assistant":
            raise ValueError("only assistant messages can have tool_calls")
        if (self.role == "tool") != (self.tool_call_id is not None):
            raise ValueError("tool_call_id is required on tool messages and only allowed there")
        if self.content is None and self.role != "assistant":
            raise ValueError(f"{self.role} messages need content")
        return self

    @classmethod
    def system(cls, content: str) -> Message:
        """A system prompt."""
        return cls(role="system", content=content)

    @classmethod
    def user(cls, content: str) -> Message:
        """A user turn."""
        return cls(role="user", content=content)

    @classmethod
    def assistant(
        cls, content: str | None = None, tool_calls: list[ToolCall] | None = None
    ) -> Message:
        """An assistant turn: text, tool calls, or both."""
        return cls(role="assistant", content=content, tool_calls=tool_calls or [])

    @classmethod
    def tool(cls, content: str, tool_call_id: str) -> Message:
        """The result of the tool call ``tool_call_id``."""
        return cls(role="tool", content=content, tool_call_id=tool_call_id)


class LLMRequest(SchemaModel):
    """Everything needed to make one LLM call, independent of the provider.

    ``messages`` also accepts a plain string, shorthand for a single user message.
    ``None`` values mean "provider/client default".
    """

    messages: list[Message] = Field(min_length=1)
    model: str | None = None
    """``None`` lets the client use its configured model."""
    temperature: NonNegativeFloat | None = None
    max_tokens: PositiveInt | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    response_format: dict[str, JsonValue] | None = None
    """E.g. ``{"type": "json_object"}`` or ``{"type": "json_schema", "json_schema": {...}}``."""

    @field_validator("messages", mode="before")
    @classmethod
    def _str_to_user_message(cls, value: Any) -> Any:
        if isinstance(value, str):
            return [Message.user(value)]
        return value

    @property
    def last_user_message(self) -> str | None:
        """Content of the most recent ``user`` message, or ``None`` if there is none."""
        for message in reversed(self.messages):
            if message.role == "user":
                return message.content
        return None

    def canonical_json(self) -> str:
        """Stable, compact JSON encoding: sorted keys, no whitespace, unicode kept.

        Two requests that are equal as models always encode to the same string,
        whatever the dict insertion order of tool schemas or ``response_format``.
        """
        return json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
        )

    def fingerprint(self) -> str:
        """SHA-256 hex digest of :meth:`canonical_json` (the cassette request key)."""
        return hashlib.sha256(self.canonical_json().encode()).hexdigest()


class LLMResponse(SchemaModel):
    """The normalised result of one LLM call."""

    content: str | None = None
    """Text output; ``None`` when the model only returned tool calls."""
    tool_calls: list[ToolCall] = Field(default_factory=list)
    usage: Usage | None = None
    """``None`` when the provider did not report usage."""
    raw: JsonValue = None
    """The provider's raw JSON payload, kept for debugging; not used for logic."""
    model: str | None = None
    """The model that actually answered, as reported by the provider."""

    def to_message(self) -> Message:
        """This response as an ``assistant`` message, to append to the conversation."""
        return Message.assistant(self.content, list(self.tool_calls))
