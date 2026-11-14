"""The support-desk agent: a plain tool-calling loop over any :class:`~agentlens.llm.LLMClient`.

.. code-block:: python

    from examples.support_agent import run_agent

    run_agent("I want a refund for order 1001")                       # offline: mock:support
    run_agent("Where is order 1002?", model="openai:gpt-4o-mini")     # a real model

The loop: send the conversation and the tool specs to the model; if it asks
for tools, run them and append their results; stop at the first answer without
tool calls, or after :data:`MAX_ITERATIONS` model calls. The model is wrapped
with :func:`agentlens.wrap_llm` and the tools with :func:`agentlens.wrap_tool`,
so inside :func:`agentlens.run` (e.g. under ``agentlens run``) every step is
traced.

The model comes from ``model=`` (passed by ``agentlens run --model`` / the
suite), else ``$AGENTLENS_MODEL``, else :data:`DEFAULT_MODEL`. Any ``mock``
spec (``mock``, ``mock:support``) runs the scripted policy of
:mod:`.mock_policy`, so the agent works with no API keys.

The system prompt lives in ``prompts/system_<version>.txt``. The version is
``prompt_version=``, else ``$SUPPORT_PROMPT_VERSION``, else :data:`PROMPT_VERSION`:
``SUPPORT_PROMPT_VERSION=v2 agentlens run examples/support_agent/suite.yaml`` runs
the regressed ``v2`` prompt of ``docs/regression-story.md`` without editing code.
"""

from __future__ import annotations

import os
from functools import cache
from pathlib import Path

import agentlens
from agentlens.llm import MODEL_ENV, LLMClient, LLMRequest, Message, get_client, parse_spec

from .data import new_database
from .mock_policy import make_mock_llm
from .tools import TOOL_SPECS, call_tool, make_tools

__all__ = [
    "DEFAULT_MODEL",
    "GIVE_UP_MESSAGE",
    "MAX_ITERATIONS",
    "PROMPT_VERSION",
    "PROMPT_VERSION_ENV",
    "build_llm",
    "resolve_prompt_version",
    "run_agent",
    "system_prompt",
]

PROMPT_VERSION = "v1"
"""Which ``prompts/system_<version>.txt`` the agent uses by default."""

PROMPT_VERSION_ENV = "SUPPORT_PROMPT_VERSION"
"""Environment variable overriding :data:`PROMPT_VERSION` (e.g. ``v2``) without editing code."""

PROMPTS_DIR = Path(__file__).parent / "prompts"

DEFAULT_MODEL = "mock:support"
"""Used when neither ``model=`` nor ``$AGENTLENS_MODEL`` is given: offline and deterministic."""

MAX_ITERATIONS = 8
"""Most model calls per run: the guard against a model that never stops calling tools."""

GIVE_UP_MESSAGE = (
    "I'm sorry, I couldn't complete your request. Please write to support@example.com."
)
"""The answer when the model is still calling tools after ``max_iterations`` calls."""


@cache
def system_prompt(version: str = PROMPT_VERSION) -> str:
    """The text of ``prompts/system_<version>.txt``."""
    return (PROMPTS_DIR / f"system_{version}.txt").read_text(encoding="utf-8").strip()


def resolve_prompt_version(version: str | None = None) -> str:
    """``version``, else ``$SUPPORT_PROMPT_VERSION``, else :data:`PROMPT_VERSION`."""
    return version or os.environ.get(PROMPT_VERSION_ENV) or PROMPT_VERSION


def build_llm(model: str | None = None) -> LLMClient:
    """The client for ``model`` (else ``$AGENTLENS_MODEL``, else :data:`DEFAULT_MODEL`).

    ``mock`` specs give the scripted support policy; anything else goes through
    :func:`agentlens.llm.get_client` (``openai:...``, ``anthropic:...``, ``ollama:...``).
    """
    spec = model or os.environ.get(MODEL_ENV) or DEFAULT_MODEL
    provider, name = parse_spec(spec)
    if provider == "mock":
        return make_mock_llm(name)
    return get_client(spec)


def run_agent(
    question: str,
    model: str | None = None,
    *,
    llm: LLMClient | None = None,
    prompt_version: str | None = None,
    max_iterations: int = MAX_ITERATIONS,
) -> str:
    """Answer one customer message, using the tools as needed (the suite's agent).

    Args:
        question: The customer's message.
        model: ``"provider:model"`` spec (see :func:`build_llm`); ``agentlens run``
            passes the suite's/CLI's model here.
        llm: A ready client, instead of ``model`` (tests, custom setups).
        prompt_version: Which system prompt to use (see :func:`resolve_prompt_version`).
        max_iterations: Most model calls before giving up with :data:`GIVE_UP_MESSAGE`.

    Returns:
        The agent's final answer.
    """
    client = agentlens.wrap_llm(llm if llm is not None else build_llm(model))
    tools = make_tools(new_database())
    messages = [
        Message.system(system_prompt(resolve_prompt_version(prompt_version))),
        Message.user(question),
    ]
    for _ in range(max_iterations):
        request = LLMRequest(messages=messages, tools=TOOL_SPECS, temperature=0)
        response = client.complete(request)
        messages.append(response.to_message())
        if not response.tool_calls:
            return response.content or ""
        for call in response.tool_calls:
            messages.append(Message.tool(call_tool(tools, call), call.id))
    return GIVE_UP_MESSAGE
