"""A scripted "policy" that plays the LLM offline: ``mock:support``.

:func:`support_policy` is a :class:`~agentlens.llm.MockLLM` responder
(``request -> tool call(s) | final text``). On every call it re-reads the whole
conversation, like a real model would:

1. **Intents** come from the customer's message: it is split into clauses
   (sentences, "and"/"also"), and each clause is matched against keyword rules
   (refund, tracking, address change, store-policy question); an order number
   is carried over to the next clause ("My order 1004 arrived broken. I want my
   money back."). A request for a person escalates the whole conversation.
2. **Progress** comes from the tool results already in the conversation. Each
   intent is a small state machine ("look the order up, then refund it if the
   policy allows"); the policy replays them over the results so far and emits
   the first tool call that has no result yet.
3. When every intent is done it writes the **final answer**, one sentence per
   intent, from the tool results.

**Policy rules are read from the system prompt** (the refund limit "up to
$200" and the window "within 30 days of delivery"), so editing the prompt
changes the mock's behaviour the way it would change a real model's: drop the
limit and the over-limit refund is issued instead of escalated, which the
golden dataset reports as a regression. Responses are a pure function of the
request, hence deterministic.
"""

from __future__ import annotations

import json
import re
from collections.abc import Generator, Sequence
from dataclasses import dataclass
from datetime import date
from typing import Any

from agentlens.llm import LLMRequest, Message, MockLLM, ScriptItem, ToolCall

from .data import TODAY

__all__ = ["Intent", "Rules", "detect_intents", "make_mock_llm", "support_policy"]

_ORDER_ID = re.compile(r"#?\b(\d{4})\b")
_ESCALATE = re.compile(
    r"\b(human|real person|manager|supervisor|someone else|speak to someone|talk to someone)\b"
)
_REFUND = re.compile(r"\brefund|\bmoney back\b|\breimburse")
_TRACK = re.compile(
    r"\bwhere(?:'s| is)\b|\btrack|\bstatus\b|\bwhen will\b"
    r"|\b(?:hasn't|has not|not yet|never|didn't|did not)\s+(?:\w+\s+)?arriv"
)
_ADDRESS = re.compile(r"\baddress\b")
_NEW_ADDRESS = re.compile(r"\bto:?\s+(\d+\s.*?)[.?!]*$")
_FAQ_TOPICS = re.compile(
    r"\b(polic(?:y|ies)|returns?|ship(?:ping)?|deliver(?:y)?|international|abroad|countr(?:y|ies)"
    r"|canada|usa|uk|pay(?:ment)?s?|paypal|card|warranty|guarantee|cancel(?:lation)?"
    r"|gift ?cards?|voucher|hours|open|contact|phone|email)\b"
)
_CLAUSES = re.compile(r"(?<=[.?!])\s+|\s+(?:and also|and then|and|also)\s+")


@dataclass(frozen=True)
class Rules:
    """The policy numbers found in the system prompt (``None``: no such rule)."""

    refund_limit: float | None = None
    refund_window_days: int | None = None

    @classmethod
    def from_prompt(cls, prompt: str) -> Rules:
        """Read ``up to $<limit>`` and ``within <n> days of delivery`` from ``prompt``."""
        limit = re.search(r"up to \$(\d+(?:\.\d+)?)", prompt)
        window = re.search(r"within (\d+) days of delivery", prompt)
        return cls(
            refund_limit=float(limit.group(1)) if limit else None,
            refund_window_days=int(window.group(1)) if window else None,
        )


@dataclass(frozen=True)
class Intent:
    """One thing the customer asked for."""

    kind: str  # refund | track | address | faq | escalate | decline
    order_id: str | None = None
    text: str = ""
    new_address: str | None = None


def _key(intent: Intent) -> tuple[str, str | None, str | None, str]:
    """What makes two intents the same request ("track 1002 ... track 1002" is one)."""
    text = intent.text.lower() if intent.kind == "faq" else ""
    return intent.kind, intent.order_id, intent.new_address, text


def detect_intents(message: str) -> list[Intent]:
    """The intents of a customer message, in the order they were asked."""
    lowered = message.lower()
    if _ESCALATE.search(lowered):
        ids = _ORDER_ID.findall(message)
        return [Intent("escalate", ids[0] if ids else None, message.strip())]
    intents: list[Intent] = []
    order_id: str | None = None
    for clause in filter(None, (c.strip() for c in _CLAUSES.split(message))):
        low = clause.lower()
        ids = _ORDER_ID.findall(clause)
        order_id = ids[0] if ids else order_id
        if "policy" in low or "policies" in low:
            intent = Intent("faq", text=clause)
        elif _ADDRESS.search(low):
            found = _NEW_ADDRESS.search(clause)
            intent = Intent("address", order_id, clause, found.group(1) if found else None)
        elif _REFUND.search(low):
            intent = Intent("refund", order_id, clause)
        elif _TRACK.search(low):
            intent = Intent("track", order_id, clause)
        elif _FAQ_TOPICS.search(low):
            intent = Intent("faq", text=clause)
        else:
            continue
        if all(_key(intent) != _key(seen) for seen in intents):
            intents.append(intent)
    return intents or [Intent("decline", text=message.strip())]


# --- conversation state ----------------------------------------------------------


@dataclass
class _Step:
    """A tool call the policy wants, or the sentence an intent ends with."""

    call: tuple[str, dict[str, Any]] | None = None
    sentence: str | None = None


_Steps = Generator[_Step, dict[str, Any], None]
"""An intent's state machine: yields steps, receives each call's result."""


def _tool_results(messages: Sequence[Message]) -> list[dict[str, Any]]:
    """The results of the tool calls so far, in call order."""
    by_id: dict[str, dict[str, Any]] = {}
    for message in messages:
        if message.role == "tool" and message.tool_call_id is not None:
            try:
                value = json.loads(message.content or "")
            except json.JSONDecodeError:
                value = {"error": "invalid_result", "message": message.content}
            if not isinstance(value, dict):
                value = {"error": "invalid_result", "message": message.content}
            by_id[message.tool_call_id] = value
    calls = [call for m in messages if m.role == "assistant" for call in m.tool_calls]
    return [by_id.get(call.id, {}) for call in calls]


def _money(value: float) -> str:
    return f"${value:,.2f}"


def _limit(value: float) -> str:
    return f"${value:,.0f}" if value == int(value) else _money(value)


def _not_found(order_id: str) -> str:
    return f"I couldn't find order {order_id}. Could you double-check the order number?"


def _ask_order(topic: str) -> str:
    return f"Could you give me your order number so I can help with your {topic}?"


def _refund_steps(intent: Intent, rules: Rules) -> _Steps:
    if intent.order_id is None:
        yield _Step(sentence=_ask_order("refund"))
        return
    oid = intent.order_id
    order = yield _Step(call=("lookup_order", {"order_id": oid}))
    if "error" in order:
        yield _Step(sentence=_not_found(oid))
        return
    total = float(order["total"])
    if order["status"] == "cancelled":
        yield _Step(
            sentence=f"Order {oid} was cancelled, so there is nothing to refund: "
            "your payment was already returned."
        )
        return
    if order["status"] != "delivered":
        yield _Step(
            sentence=f"Order {oid} has not been delivered yet (status: {order['status']}), "
            "so I can't refund it yet; you can ask for a refund once it arrives."
        )
        return
    if order["refunded"]:
        yield _Step(sentence=f"Order {oid} has already been refunded.")
        return
    delivered = date.fromisoformat(order["delivered_on"])
    window = rules.refund_window_days
    if window is not None and (TODAY - delivered).days > window:
        yield _Step(
            sentence=f"Order {oid} was delivered on {order['delivered_on']}, more than "
            f"{window} days ago, so it is outside our refund window and I can't refund it."
        )
        return
    limit = rules.refund_limit
    if limit is not None and total > limit:
        reason = f"Refund of {_money(total)} requested for order {oid} (above {_limit(limit)})"
        ticket = yield _Step(call=("escalate_to_human", {"reason": reason, "order_id": oid}))
        yield _Step(
            sentence=f"Refunds above {_limit(limit)} need a human agent, so I can't issue the "
            f"{_money(total)} refund for order {oid} myself. I have escalated it "
            f"(ticket {ticket.get('ticket_id', '?')}) and a colleague will contact you "
            f"{ticket.get('response_time', 'soon')}."
        )
        return
    refund = yield _Step(
        call=("issue_refund", {"order_id": oid, "amount": total, "reason": intent.text})
    )
    if "error" in refund:
        yield _Step(sentence=f"I couldn't refund order {oid}: {refund.get('message', 'error')}.")
        return
    yield _Step(
        sentence=f"I have issued a refund of {_money(float(refund['amount']))} for order {oid} "
        f"(refund {refund['refund_id']}); it will reach your original payment method within "
        "5 business days."
    )


def _track_steps(intent: Intent) -> _Steps:
    if intent.order_id is None:
        yield _Step(sentence=_ask_order("delivery"))
        return
    oid = intent.order_id
    info = yield _Step(call=("track_order", {"order_id": oid}))
    if "error" in info:
        yield _Step(sentence=_not_found(oid))
        return
    status = info["status"]
    carrier = info.get("carrier")
    tracking = info.get("tracking_number")
    if status == "delivered":
        text = f"Order {oid} was delivered on {info['delivered_on']} by {carrier}."
    elif status == "shipped":
        text = (
            f"Order {oid} has shipped with {carrier} (tracking number {tracking}) and should "
            f"arrive on {info['eta']}."
        )
    elif status == "delayed":
        text = (
            f"Order {oid} is delayed with {carrier} (tracking number {tracking}); the new "
            f"expected delivery date is {info['eta']}. Sorry for the wait!"
        )
    elif status == "processing":
        text = f"Order {oid} is still being prepared and has not shipped yet."
    else:
        text = f"Order {oid} is {status}, so it will not be shipped."
    yield _Step(sentence=text)


def _address_steps(intent: Intent) -> _Steps:
    if intent.order_id is None:
        yield _Step(sentence=_ask_order("address change"))
        return
    oid = intent.order_id
    if intent.new_address is None:
        yield _Step(sentence=f"What is the new shipping address for order {oid}?")
        return
    order = yield _Step(call=("lookup_order", {"order_id": oid}))
    if "error" in order:
        yield _Step(sentence=_not_found(oid))
        return
    if order["status"] != "processing":
        yield _Step(
            sentence=f"Order {oid} is already {order['status']}, so its shipping address can "
            "no longer be changed."
        )
        return
    updated = yield _Step(
        call=("update_address", {"order_id": oid, "new_address": intent.new_address})
    )
    if "error" in updated:
        yield _Step(sentence=f"I couldn't update order {oid}: {updated.get('message', 'error')}.")
        return
    yield _Step(
        sentence=f"I have updated the shipping address of order {oid} to "
        f"{updated['shipping_address']}."
    )


def _faq_steps(intent: Intent) -> _Steps:
    found = yield _Step(call=("search_faq", {"query": intent.text}))
    answers = [r["answer"] for r in found.get("results", [])]
    if not answers:
        yield _Step(
            sentence="I couldn't find an answer to that in our help centre; "
            "please write to support@example.com."
        )
        return
    yield _Step(sentence=" ".join(answers))


def _escalate_steps(intent: Intent) -> _Steps:
    arguments: dict[str, Any] = {"reason": intent.text}
    if intent.order_id is not None:
        arguments["order_id"] = intent.order_id
    ticket = yield _Step(call=("escalate_to_human", arguments))
    yield _Step(
        sentence=f"I have passed your request to a human agent (ticket "
        f"{ticket.get('ticket_id', '?')}); they will get back to you "
        f"{ticket.get('response_time', 'soon')}."
    )


def _steps(intent: Intent, rules: Rules) -> _Steps:
    if intent.kind == "refund":
        return _refund_steps(intent, rules)
    if intent.kind == "track":
        return _track_steps(intent)
    if intent.kind == "address":
        return _address_steps(intent)
    if intent.kind == "faq":
        return _faq_steps(intent)
    if intent.kind == "escalate":
        return _escalate_steps(intent)
    return _decline_steps()


def _decline_steps() -> _Steps:
    yield _Step(
        sentence="I'm sorry, I can only help with questions about our store and your orders."
    )


def _next(intents: Sequence[Intent], rules: Rules, results: Sequence[dict[str, Any]]) -> ScriptItem:
    """Replay every intent over ``results``; the first call without a result is next."""
    sentences: list[str] = []
    used = 0
    for intent in intents:
        machine = _steps(intent, rules)
        step = next(machine)
        while step.call is not None:
            if used == len(results):
                name, arguments = step.call
                return ToolCall(id=f"call_{used + 1}", name=name, arguments=arguments)
            step = machine.send(results[used])
            used += 1
        if step.sentence:
            sentences.append(step.sentence)
    return " ".join(sentences)


def support_policy(request: LLMRequest) -> ScriptItem:
    """The next response of the support agent for ``request`` (a ``MockLLM`` responder)."""
    system = "\n".join(m.content or "" for m in request.messages if m.role == "system")
    question = next((m.content or "" for m in request.messages if m.role == "user"), "")
    return _next(
        detect_intents(question), Rules.from_prompt(system), _tool_results(request.messages)
    )


def make_mock_llm(model: str = "support") -> MockLLM:
    """A :class:`~agentlens.llm.MockLLM` playing the support agent (``mock:<model>``)."""
    return MockLLM(support_policy, model=model)
