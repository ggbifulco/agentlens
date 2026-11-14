"""The support desk's tools, their JSON-schema specs, and how the agent runs them.

:func:`make_tools` binds the tools to one run's :class:`~.data.Database` and
wraps each one with :func:`agentlens.wrap_tool`, so every call becomes a
``tool_call`` span named after the tool (what ``tool_sequence`` checks).

The tools only enforce hard invariants of the data (the order exists, a refund
needs a delivered order and cannot exceed its total, an address can change only
before shipping). Business *policy* — the refund limit and window, when to
escalate — lives in the system prompt (``prompts/``), so a prompt change can
cause a behaviour regression that the golden dataset catches.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable, Mapping
from typing import Any

import agentlens
from agentlens.llm import ToolCall, ToolSpec

from .data import FAQ, Database

__all__ = ["TOOL_NAMES", "TOOL_SPECS", "Tool", "call_tool", "make_tools"]

Tool = Callable[..., dict[str, Any]]
"""A tool: keyword arguments in, a JSON-serialisable dict out."""


def _object(properties: dict[str, Any], required: list[str]) -> dict[str, Any]:
    return {"type": "object", "properties": properties, "required": required}


_ORDER_ID = {"type": "string", "description": "The order number, e.g. '1001'."}

TOOL_SPECS: list[ToolSpec] = [
    ToolSpec(
        name="lookup_order",
        description="Look up an order: items, total, status, delivery date, customer.",
        parameters=_object({"order_id": _ORDER_ID}, ["order_id"]),
    ),
    ToolSpec(
        name="issue_refund",
        description="Refund a delivered order to the original payment method.",
        parameters=_object(
            {
                "order_id": _ORDER_ID,
                "amount": {"type": "number", "description": "Amount to refund, in USD."},
                "reason": {"type": "string", "description": "Why the customer wants a refund."},
            },
            ["order_id", "amount"],
        ),
    ),
    ToolSpec(
        name="track_order",
        description="Shipping status of an order: carrier, tracking number, ETA or delivery date.",
        parameters=_object({"order_id": _ORDER_ID}, ["order_id"]),
    ),
    ToolSpec(
        name="update_address",
        description="Change the shipping address of an order that has not shipped yet.",
        parameters=_object(
            {
                "order_id": _ORDER_ID,
                "new_address": {"type": "string", "description": "The full new address."},
            },
            ["order_id", "new_address"],
        ),
    ),
    ToolSpec(
        name="escalate_to_human",
        description="Open a ticket for a human support agent.",
        parameters=_object(
            {
                "reason": {"type": "string", "description": "Short summary for the human agent."},
                "order_id": _ORDER_ID,
            },
            ["reason"],
        ),
    ),
    ToolSpec(
        name="search_faq",
        description="Search the help centre for store policies (returns, shipping, payment...).",
        parameters=_object(
            {"query": {"type": "string", "description": "The customer's question."}},
            ["query"],
        ),
    ),
]
"""The tools as the LLM sees them (JSON Schema parameters)."""

TOOL_NAMES: tuple[str, ...] = tuple(spec.name for spec in TOOL_SPECS)


def _order_key(order_id: object) -> str:
    return str(order_id).strip().lstrip("#").strip()


def _not_found(order_id: str) -> dict[str, Any]:
    return {"error": "order_not_found", "message": f"order {order_id} not found"}


def make_tools(db: Database) -> dict[str, Tool]:
    """The tools bound to ``db``, each wrapped with :func:`agentlens.wrap_tool`."""

    def lookup_order(order_id: str) -> dict[str, Any]:
        key = _order_key(order_id)
        order = db.orders.get(key)
        if order is None:
            return _not_found(key)
        customer = db.customers[order["customer_id"]]
        return {**order, "customer": {"name": customer["name"], "email": customer["email"]}}

    def issue_refund(order_id: str, amount: float, reason: str = "") -> dict[str, Any]:
        key = _order_key(order_id)
        order = db.orders.get(key)
        if order is None:
            return _not_found(key)
        if order["status"] != "delivered":
            return {"error": "not_refundable", "message": f"order {key} is {order['status']}"}
        if order["refunded"]:
            return {"error": "already_refunded", "message": f"order {key} was already refunded"}
        value = float(amount)
        if not 0 < value <= order["total"]:
            return {
                "error": "invalid_amount",
                "message": f"amount must be between 0 and the order total {order['total']:.2f}",
            }
        order["refunded"] = True
        refund = {
            "refund_id": f"R-{key}",
            "order_id": key,
            "amount": round(value, 2),
            "reason": reason,
            "status": "issued",
        }
        db.refunds.append(refund)
        return refund

    def track_order(order_id: str) -> dict[str, Any]:
        key = _order_key(order_id)
        order = db.orders.get(key)
        if order is None:
            return _not_found(key)
        fields = ("status", "carrier", "tracking_number", "eta", "delivered_on")
        return {"order_id": key, **{name: order[name] for name in fields if name in order}}

    def update_address(order_id: str, new_address: str) -> dict[str, Any]:
        key = _order_key(order_id)
        order = db.orders.get(key)
        if order is None:
            return _not_found(key)
        if order["status"] != "processing":
            return {
                "error": "already_shipped",
                "message": f"order {key} is {order['status']}; the address can no longer change",
            }
        address = new_address.strip()
        if not address:
            return {"error": "invalid_address", "message": "the new address is empty"}
        order["shipping_address"] = address
        return {"order_id": key, "shipping_address": address, "status": "updated"}

    def escalate_to_human(reason: str, order_id: str | None = None) -> dict[str, Any]:
        ticket = {
            "ticket_id": f"ESC-{1001 + len(db.tickets)}",
            "reason": reason,
            "order_id": None if order_id is None else _order_key(order_id),
            "queue": "tier-2",
            "response_time": "within 24 hours",
        }
        db.tickets.append(ticket)
        return ticket

    def search_faq(query: str) -> dict[str, Any]:
        text = query.lower()
        scored = []
        for rank, article in enumerate(FAQ):
            hits = sum(1 for word in article.keywords if re.search(rf"\b{word}\b", text))
            if hits:
                scored.append((-hits, rank, article))
        results = [
            {"id": article.id, "title": article.title, "answer": article.answer}
            for _, _, article in sorted(scored)[:2]
        ]
        return {"query": query, "results": results}

    functions: list[Tool] = [
        lookup_order,
        issue_refund,
        track_order,
        update_address,
        escalate_to_human,
        search_faq,
    ]
    return {fn.__name__: agentlens.wrap_tool(fn, name=fn.__name__) for fn in functions}


def call_tool(tools: Mapping[str, Tool], call: ToolCall) -> str:
    """Run one tool call and return its result as the JSON text of a ``tool`` message.

    Mistakes a real model can make (an unknown tool, wrong or missing arguments)
    and exceptions raised by a tool become an ``{"error": ...}`` result the
    model can read and recover from, instead of crashing the run. A tool
    exception is still recorded on its span by ``wrap_tool``.
    """
    tool = tools.get(call.name)
    if tool is None:
        result: dict[str, Any] = {
            "error": "unknown_tool",
            "message": f"no tool named {call.name!r}; available: {', '.join(tools)}",
        }
    else:
        try:
            result = tool(**call.arguments)
        except TypeError as exc:
            result = {"error": "invalid_arguments", "message": str(exc)}
        except Exception as exc:
            result = {"error": "tool_failed", "message": f"{type(exc).__name__}: {exc}"}
    return json.dumps(result, sort_keys=True)
