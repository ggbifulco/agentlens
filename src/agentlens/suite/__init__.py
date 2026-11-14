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

Run it (FR-S2/S3): every case ``repetitions`` times with bounded concurrency,
traces and eval results saved under a new experiment::

    from agentlens.store import SQLiteStore
    from agentlens.suite import run_suite, set_baseline

    with SQLiteStore() as store:
        result = run_suite(suite, store=store, name="main")
        print(f"{result.pass_rate:.0%}")
        set_baseline(store, result.experiment)
"""

from agentlens.suite.agent import (
    check_import_path,
    import_module,
    is_async_agent,
    resolve_agent,
)
from agentlens.suite.errors import AgentImportError, SuiteError, SuiteIssue, format_loc
from agentlens.suite.experiment import (
    ExperimentNotFoundError,
    experiment_config,
    find_experiment,
    get_baseline,
    git_info,
    resolve_model,
    set_baseline,
    unset_baseline,
)
from agentlens.suite.loader import DATASET_SUFFIXES, load_cases, load_suite
from agentlens.suite.model import Suite, SuiteDefaults, merge_evaluators
from agentlens.suite.runner import (
    CaseResult,
    ExperimentRun,
    ProgressCallback,
    RunResult,
    RunTimeoutError,
    arun_suite,
    default_experiment_name,
    run_suite,
)

__all__ = [
    "DATASET_SUFFIXES",
    "AgentImportError",
    "CaseResult",
    "ExperimentNotFoundError",
    "ExperimentRun",
    "ProgressCallback",
    "RunResult",
    "RunTimeoutError",
    "Suite",
    "SuiteDefaults",
    "SuiteError",
    "SuiteIssue",
    "arun_suite",
    "check_import_path",
    "default_experiment_name",
    "experiment_config",
    "find_experiment",
    "format_loc",
    "get_baseline",
    "git_info",
    "import_module",
    "is_async_agent",
    "load_cases",
    "load_suite",
    "merge_evaluators",
    "resolve_agent",
    "resolve_model",
    "run_suite",
    "set_baseline",
    "unset_baseline",
]
