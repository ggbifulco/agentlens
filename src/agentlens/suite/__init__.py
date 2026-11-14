"""Suites: an agent, a dataset of cases and how to run them (``docs/design.md`` §6).

Load a suite YAML file (errors point at ``file:line:column``)::

    from agentlens.suite import load_suite

    suite = load_suite("suites/support.yaml")
    agent = suite.load_agent()            # imports "examples.support_agent:run_agent"
    for case in suite.resolved_cases():   # suite defaults merged into each case
        output = agent(case.input)

or define one in Python (FR-S1)::

    from agentlens.suite import Suite

    suite = Suite(name="support", agent=run_agent,
                  cases=[{"id": "refund", "input": "refund order 1234"}]).check()

Datasets can live in their own file (``cases: data/support.jsonl`` in the suite,
or :func:`load_cases` directly).
"""

from agentlens.suite.agent import (
    check_import_path,
    import_module,
    is_async_agent,
    resolve_agent,
)
from agentlens.suite.errors import AgentImportError, SuiteError, SuiteIssue, format_loc
from agentlens.suite.loader import DATASET_SUFFIXES, load_cases, load_suite
from agentlens.suite.model import Suite, SuiteDefaults, merge_evaluators

__all__ = [
    "DATASET_SUFFIXES",
    "AgentImportError",
    "Suite",
    "SuiteDefaults",
    "SuiteError",
    "SuiteIssue",
    "check_import_path",
    "format_loc",
    "import_module",
    "is_async_agent",
    "load_cases",
    "load_suite",
    "merge_evaluators",
    "resolve_agent",
]
