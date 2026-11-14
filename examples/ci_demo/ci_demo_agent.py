"""A tiny deterministic agent for the AgentLens CI demo (no API keys).

A "shop assistant" answers questions from a small FAQ through a scripted
:class:`~agentlens.llm.MockLLM`: the first pattern that matches the question
gives the answer. Every call is traced (``agentlens.wrap_llm``), so the suite
gets steps, latency and cost like a real agent.

Edit ``FAQ`` (e.g. break an answer) and open a pull request to see the
AgentLens comment report a regression. The full demo agent replaces this one
in a later release.
"""

from __future__ import annotations

import agentlens
from agentlens.llm import LLMRequest, Message, MockLLM

FAQ: dict[str, str] = {
    r"(?i)opening hours|open": "We are open 9:00-18:00, Monday to Saturday.",
    r"(?i)refund": "Refunds are issued within 14 days of purchase.",
    r"(?i)shipping|deliver": "Standard shipping takes 3-5 business days.",
    r"(?i)track": "Track your order from the link in the confirmation email.",
    r"(?i)password": "Reset your password from the login page.",
    r"(?i)gift card": "Gift cards never expire.",
    r"(?i)contact|phone|email": "Write to support@example.com.",
}
FALLBACK = "Sorry, I can only answer questions about the shop."

_llm = agentlens.wrap_llm(MockLLM(FAQ, default=FALLBACK))


def answer(question: str) -> str:
    """Answer one customer question (the suite's agent)."""
    request = LLMRequest(
        messages=[Message.system("You are a shop assistant."), Message.user(question)]
    )
    return _llm.complete(request).content or ""
