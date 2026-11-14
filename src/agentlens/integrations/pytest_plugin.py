"""pytest plugin: agent suites as pytest tests (FR-S6).

Installed with AgentLens and loaded automatically by pytest (entry point
``pytest11: agentlens``); it does nothing unless its options or fixture are used.

**Suite files.** Every case of a suite becomes one test item::

    pytest --agentlens suites/support.yaml --agentlens suites/billing.yaml

collects ``suites/support.yaml::<case_id>`` items next to the normal tests (select
them with ``-k``, by case id, suite name or tag). The agent runs every
repetition of the selected cases (one experiment per suite, honouring the
suite's ``concurrency``), the case's evaluators score each run, and a case
passes when **every** repetition passed. A failure lists, per failing
repetition, the agent error and each failed evaluator with its score and
reason. Options: ``--agentlens-repetitions N`` overrides the suites'
repetitions, ``--agentlens-db PATH`` saves the experiments (traces and scores)
to a SQLite database, e.g. for ``agentlens compare``.

**Python tests.** The ``agentlens_run`` fixture runs an agent on one case::

    from agentlens.integrations.pytest_plugin import assert_passed

    def test_refund(agentlens_run):
        result = agentlens_run(my_agent, {"id": "refund", "input": "refund 1234",
                                          "expected": "Refunded"},
                               evaluators=["no_error", "contains"])
        assert_passed(result)                 # AssertionError listing evaluator reasons
        assert result.mean_scores()["contains"] == 1.0
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pytest

if TYPE_CHECKING:
    from _pytest._code.code import TerminalRepr

    from agentlens.schema import Case
    from agentlens.store import SQLiteStore
    from agentlens.suite import CaseResult, ExperimentRun, RunResult, Suite

__all__ = [
    "AgentlensCase",
    "AgentlensSuiteFile",
    "CaseFailedError",
    "CaseRunner",
    "SuiteRunError",
    "agentlens_run",
    "assert_passed",
    "format_case_result",
]

_MAX_VALUE = 200
"""Inputs, outputs and expected values longer than this are truncated in reports."""


class CaseFailedError(AssertionError):
    """A case had at least one failing repetition; ``str()`` is the readable report."""

    def __init__(self, result: CaseResult, suite: str | None = None) -> None:
        super().__init__(format_case_result(result, suite=suite))
        self.result = result


class SuiteRunError(Exception):
    """A suite could not run at all (agent import failure, bad evaluator spec, store error)."""


# ---------------------------------------------------------------------- report


def _short(value: object) -> str:
    text = json.dumps(value, ensure_ascii=False, default=str)
    return text if len(text) <= _MAX_VALUE else text[: _MAX_VALUE - 3] + "..."


def _format_run(run: RunResult) -> list[str]:
    lines = [f"  repetition {run.repetition}: FAILED (trace {run.trace.id})"]
    if run.error is not None:
        lines.append(f"    error: {run.error}")
    else:
        lines.append(f"    output: {_short(run.trace.output)}")
    for res in run.eval_results:
        if not res.passed:
            reason = res.reason or "(no reason given)"
            lines.append(f"    {res.evaluator}: score {res.score:.2f}: {reason}")
    return lines


def format_case_result(result: CaseResult, *, suite: str | None = None) -> str:
    """A readable report of ``result``: the case, then every failing repetition with
    its error or output and each failed evaluator (``name: score S: reason``).

    Passing repetitions are only counted. Used for failed suite items and by
    :func:`assert_passed`.
    """
    where = f" of suite {suite!r}" if suite else ""
    total = len(result.runs)
    lines = [
        f"agentlens case {result.case_id!r}{where}: "
        f"{result.passed}/{total} repetition{'s' if total != 1 else ''} passed",
        f"  input: {_short(result.case.input)}",
    ]
    if result.case.expected is not None:
        lines.append(f"  expected: {_short(result.case.expected)}")
    for run in result.runs:
        if not run.passed:
            lines += _format_run(run)
    return "\n".join(lines)


def assert_passed(result: CaseResult, *, suite: str | None = None) -> None:
    """Fail the current test unless every repetition of ``result`` passed.

    Raises:
        CaseFailedError: an :class:`AssertionError` whose message is
            :func:`format_case_result` (evaluator, score and reason of each failure).
    """
    __tracebackhide__ = True
    if result.passed != len(result.runs):
        raise CaseFailedError(result, suite=suite)


# ---------------------------------------------------------------------- state


class _State:
    """Per-session plugin state (kept in ``config.stash``, never in globals)."""

    def __init__(self, suites: set[Path], db: Path | None, repetitions: int | None) -> None:
        self.suites = suites
        self.db = db
        self.repetitions = repetitions
        self._store: SQLiteStore | None = None

    def store(self) -> SQLiteStore | None:
        """The ``--agentlens-db`` store, opened on first use (``None`` without the option)."""
        if self.db is None:
            return None
        if self._store is None:
            from agentlens.store import SQLiteStore

            self._store = SQLiteStore(self.db)
        return self._store

    def close(self) -> None:
        if self._store is not None:
            self._store.close()
            self._store = None


_STATE = pytest.StashKey[_State]()


def _state(config: pytest.Config) -> _State:
    return config.stash[_STATE]


# ---------------------------------------------------------------------- hooks


def pytest_addoption(parser: pytest.Parser) -> None:
    """Register ``--agentlens``, ``--agentlens-db`` and ``--agentlens-repetitions``."""
    group = parser.getgroup("agentlens", "AgentLens agent suites")
    group.addoption(
        "--agentlens",
        action="append",
        default=[],
        metavar="SUITE",
        dest="agentlens_suites",
        help="Collect every case of this AgentLens suite YAML file as a test (repeatable).",
    )
    group.addoption(
        "--agentlens-db",
        default=None,
        metavar="PATH",
        dest="agentlens_db",
        help="Save AgentLens experiments (traces, eval results) to this SQLite database.",
    )
    group.addoption(
        "--agentlens-repetitions",
        type=int,
        default=None,
        metavar="N",
        dest="agentlens_repetitions",
        help="Run every AgentLens case N times (overrides the suites' repetitions).",
    )


def pytest_configure(config: pytest.Config) -> None:
    """Validate the options and add the suite files to the paths pytest collects."""
    base = config.invocation_params.dir
    # Files already named by a positional argument (``suite.yaml::case_id``) are not
    # added again, so node ids keep narrowing the selection.
    named = {(base / arg.split("::")[0]).resolve() for arg in config.args}
    suites: set[Path] = set()
    for raw in config.getoption("agentlens_suites"):
        path = (base / raw).resolve()
        if not path.is_file():
            raise pytest.UsageError(f"--agentlens: suite file not found: {raw}")
        if path not in suites:
            suites.add(path)
            if path not in named:
                config.args.append(str(path))
    repetitions: int | None = config.getoption("agentlens_repetitions")
    if repetitions is not None and repetitions < 1:
        raise pytest.UsageError("--agentlens-repetitions must be >= 1")
    db: str | None = config.getoption("agentlens_db")
    config.stash[_STATE] = _State(suites, base / db if db else None, repetitions)


def pytest_unconfigure(config: pytest.Config) -> None:
    """Close the ``--agentlens-db`` store, if it was opened."""
    state = config.stash.get(_STATE, None)
    if state is not None:
        state.close()


def pytest_collect_file(file_path: Path, parent: pytest.Collector) -> AgentlensSuiteFile | None:
    """Collect the files passed with ``--agentlens`` (and nothing else)."""
    if file_path.resolve() in _state(parent.config).suites:
        return AgentlensSuiteFile.from_parent(parent, path=file_path)
    return None


# ---------------------------------------------------------------------- collection


class AgentlensSuiteFile(pytest.File):
    """A suite YAML file: loads the suite and yields one :class:`AgentlensCase` per case.

    The suite runs once, when its first selected case runs: only the selected
    cases (after ``-k``/``-m``/node-id selection), as one experiment.
    """

    suite: Suite

    def collect(self) -> Iterator[AgentlensCase]:
        """Load the suite (a :class:`~agentlens.suite.SuiteError` is a collection error)."""
        from agentlens.suite import load_suite

        self.suite = load_suite(self.path)
        self._outcome: ExperimentRun | SuiteRunError | None = None
        for case in self.suite.resolved_cases():
            item = AgentlensCase.from_parent(self, name=case.id, case=case)
            item.extra_keyword_matches.update({self.suite.name, *case.tags})
            yield item

    def repr_failure(  # type: ignore[override]
        self, excinfo: pytest.ExceptionInfo[BaseException]
    ) -> str | TerminalRepr:
        """Suite errors are shown as their ``file:line:column`` issues, without a traceback."""
        from agentlens.suite import SuiteError

        if isinstance(excinfo.value, SuiteError):
            return str(excinfo.value)
        return super().repr_failure(excinfo)

    def _run(self) -> ExperimentRun | SuiteRunError:
        from agentlens.suite import run_suite

        selected = {
            item.case.id
            for item in self.session.items
            if isinstance(item, AgentlensCase) and item.parent is self
        }
        suite = self.suite.model_copy(
            update={"cases": [case for case in self.suite.cases if case.id in selected]}
        )
        state = _state(self.config)
        try:
            return run_suite(suite, store=state.store(), repetitions=state.repetitions)
        except Exception as exc:
            return SuiteRunError(f"suite {self.suite.name!r} could not run: {_describe(exc)}")

    def experiment_run(self) -> ExperimentRun:
        """The suite's run, executing the selected cases on first call.

        Raises:
            SuiteRunError: if the suite could not run.
        """
        if self._outcome is None:
            self._outcome = self._run()
        if isinstance(self._outcome, SuiteRunError):
            raise SuiteRunError(str(self._outcome))
        return self._outcome


def _describe(exc: BaseException) -> str:
    text = str(exc)
    return f"{type(exc).__name__}: {text}" if text else type(exc).__name__


class AgentlensCase(pytest.Item):
    """One case of a suite file (all its repetitions)."""

    def __init__(self, *, case: Case, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self.case = case

    @property
    def suite_file(self) -> AgentlensSuiteFile:
        """The suite file this case belongs to."""
        assert isinstance(self.parent, AgentlensSuiteFile)
        return self.parent

    def runtest(self) -> None:
        """Run (or reuse) the suite's experiment and fail unless every repetition passed."""
        suite_file = self.suite_file
        run = suite_file.experiment_run()
        result = run.case(self.case.id)
        self.user_properties.append(("agentlens_experiment", run.experiment.id))
        self.user_properties.append(("agentlens_pass_rate", result.pass_rate))
        for name, score in sorted(result.mean_scores().items()):
            self.user_properties.append((f"agentlens_score.{name}", score))
        if result.passed != len(result.runs):
            raise CaseFailedError(result, suite=suite_file.suite.name)

    def repr_failure(
        self,
        excinfo: pytest.ExceptionInfo[BaseException],
        style: Any = None,
    ) -> str | TerminalRepr:
        """The evaluator report for failed cases, the reason for suites that could not run."""
        if isinstance(excinfo.value, CaseFailedError | SuiteRunError):
            return str(excinfo.value)
        return super().repr_failure(excinfo, style)  # pragma: no cover - plugin bugs only

    def reportinfo(self) -> tuple[Path, int | None, str]:
        """``(suite file, None, "agentlens <suite>::<case>")`` for report headers."""
        return self.path, None, f"agentlens {self.suite_file.suite.name}::{self.case.id}"


# ---------------------------------------------------------------------- fixture


class CaseRunner:
    """What the ``agentlens_run`` fixture returns: call it to run an agent on one case."""

    def __init__(self, *, name: str, store: SQLiteStore | None, repetitions: int | None) -> None:
        self.name = name
        self.store = store
        self.repetitions = repetitions

    def __call__(
        self,
        agent: Callable[..., Any] | str,
        case: Case | Mapping[str, Any],
        *,
        evaluators: Sequence[str | Mapping[str, Any]] = (),
        repetitions: int | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> CaseResult:
        """Run ``agent`` on ``case`` and evaluate it (like a one-case suite).

        Args:
            agent: The agent callable, or an import path ``"pkg.module:callable"``.
            case: A :class:`~agentlens.schema.Case` or the same fields as a mapping
                (``{"id": ..., "input": ..., "expected": ..., "evaluators": [...]}``).
            evaluators: Evaluator specs applied on top of the case's own (a case
                evaluator with the same name wins), as in a suite's ``defaults``.
            repetitions: How many runs (default ``--agentlens-repetitions``, else 1).
            model: Passed to agents that take a ``model`` parameter.
            timeout_s: Per-run timeout in seconds.

        Returns:
            The :class:`~agentlens.suite.CaseResult`; check it with :func:`assert_passed`.
            With ``--agentlens-db`` the run is also saved as an experiment of the
            suite named after the test.
        """
        from agentlens.suite import Suite, run_suite

        reps = repetitions if repetitions is not None else (self.repetitions or 1)
        suite = Suite.model_validate(
            {
                "name": self.name,
                "agent": agent,
                "repetitions": reps,
                "timeout_s": timeout_s,
                "defaults": {"evaluators": list(evaluators)},
                "cases": [case],
            }
        )
        return run_suite(suite, store=self.store, model=model).cases[0]


@pytest.fixture
def agentlens_run(request: pytest.FixtureRequest) -> CaseRunner:
    """Run an agent on one case and evaluate it: ``agentlens_run(agent, case, evaluators=...)``.

    Returns a :class:`CaseRunner`; see :func:`assert_passed` for a readable failure.
    """
    state = _state(request.config)
    return CaseRunner(name=request.node.name, store=state.store(), repetitions=state.repetitions)
