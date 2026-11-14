"""Example support-desk agent: a tool-calling loop, mock tools and a golden dataset (FR-D1).

``run_agent(question, model=None)`` answers a customer message with the tools
``lookup_order``, ``issue_refund``, ``track_order``, ``update_address``,
``escalate_to_human`` and ``search_faq`` over an in-memory database. It runs
offline with the scripted ``mock:support`` policy (the default) and with any
real provider (``model="openai:gpt-4o-mini"`` or ``$AGENTLENS_MODEL``).

Run the golden dataset from the repository root::

    agentlens run examples/support_agent/suite.yaml

See ``README.md`` in this folder.
"""

from .agent import (
    DEFAULT_MODEL,
    GIVE_UP_MESSAGE,
    MAX_ITERATIONS,
    PROMPT_VERSION,
    PROMPT_VERSION_ENV,
    build_llm,
    resolve_prompt_version,
    run_agent,
    system_prompt,
)
from .mock_policy import detect_intents, make_mock_llm, support_policy
from .tools import TOOL_NAMES, TOOL_SPECS, call_tool, make_tools

__all__ = [
    "DEFAULT_MODEL",
    "GIVE_UP_MESSAGE",
    "MAX_ITERATIONS",
    "PROMPT_VERSION",
    "PROMPT_VERSION_ENV",
    "TOOL_NAMES",
    "TOOL_SPECS",
    "build_llm",
    "call_tool",
    "detect_intents",
    "make_mock_llm",
    "make_tools",
    "resolve_prompt_version",
    "run_agent",
    "support_policy",
    "system_prompt",
]
