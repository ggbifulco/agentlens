"""Shared helpers for replay tests: a tiny tool-using agent and trace normalisation."""

from __future__ import annotations

import json
from collections.abc import Awaitable, Callable
from typing import Any

from pydantic import JsonValue

from agentlens.llm import LLMClient, LLMRequest, LLMResponse, Message, ToolCall, ToolSpec
from agentlens.schema import Trace

TOOLS = [ToolSpec(name="lookup_order", description="Look up an order")]

SCRIPT: list[LLMResponse | str | ToolCall] = [
    ToolCall(id="call_1", name="lookup_order", arguments={"order_id": "1234"}),
    "Order 1234 shipped yesterday.",
]


def lookup_order(order_id: str) -> dict[str, str]:
    """The agent's tool."""
    return {"order_id": order_id, "status": "shipped"}


async def alookup_order(order_id: str) -> dict[str, str]:
    """Async variant of :func:`lookup_order`."""
    return lookup_order(order_id)


def agent(llm: LLMClient, lookup: Callable[..., Any], question: str) -> str | None:
    """Ask the LLM, run requested tools, loop until a text answer."""
    messages = [Message.user(question)]
    while True:
        response = llm.complete(LLMRequest(messages=messages, tools=TOOLS))
        messages.append(response.to_message())
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            result = lookup(**call.arguments)
            messages.append(Message.tool(json.dumps(result), call.id))


async def aagent(
    llm: LLMClient, lookup: Callable[..., Awaitable[Any]], question: str
) -> str | None:
    """Async variant of :func:`agent`."""
    messages = [Message.user(question)]
    while True:
        response = await llm.acomplete(LLMRequest(messages=messages, tools=TOOLS))
        messages.append(response.to_message())
        if not response.tool_calls:
            return response.content
        for call in response.tool_calls:
            result = await lookup(**call.arguments)
            messages.append(Message.tool(json.dumps(result), call.id))


class FailingLLM:
    """A client that must never be reached (no ``model``: names come from the cassette)."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("the real client was called")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        raise AssertionError("the real client was called")


def normalise(trace: Trace) -> JsonValue:
    """Trace JSON without the volatile fields (ids, timestamps); parents by position."""
    positions = {span.id: n for n, span in enumerate(trace.spans)}
    data = trace.model_dump(mode="json", exclude={"id", "start_time", "end_time", "spans"})
    data["spans"] = [
        {
            **span.model_dump(
                mode="json", exclude={"id", "trace_id", "parent_id", "start_time", "end_time"}
            ),
            "parent": positions.get(span.parent_id) if span.parent_id else None,
        }
        for span in trace.spans
    ]
    return data
