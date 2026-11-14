from __future__ import annotations

import asyncio
import inspect
import logging
import threading
import time
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import pytest

import agentlens
from agentlens.evals import EvaluatorRegistry, UnknownEvaluatorError
from agentlens.llm import LLMRequest, Message, get_client
from agentlens.schema import Case, EvalResult, Experiment, SpanKind, Trace
from agentlens.store import SQLiteStore, StoreError
from agentlens.suite import (
    CaseResult,
    ExperimentRun,
    RunResult,
    RunTimeoutError,
    Suite,
    arun_suite,
    default_experiment_name,
    run_suite,
)
from agentlens.suite import runner as runner_module
from agentlens.suite.runner import REDACTION_FAILED
from tests.suite.conftest import WriteFile

CASES: list[dict[str, Any]] = [
    {"id": "hello", "input": "hello", "expected": "hello"},
    {"id": "world", "input": "world", "expected": "world"},
]
EVALUATORS: list[Any] = ["no_error", "exact_match"]


def echo_agent(question: str, model: str | None = None) -> str:
    """Toy sync agent: one traced LLM call to the echoing mock client."""
    llm = agentlens.wrap_llm(get_client(model or "mock"))
    return llm.complete(LLMRequest(messages=[Message.user(question)])).content or ""


async def async_echo_agent(question: str, model: str | None = None) -> str:
    llm = agentlens.wrap_llm(get_client(model or "mock"))
    response = await llm.acomplete(LLMRequest(messages=[Message.user(question)]))
    return response.content or ""


def make_suite(agent: Any = echo_agent, **kwargs: Any) -> Suite:
    data: dict[str, Any] = {
        "name": "toy",
        "agent": agent,
        "cases": CASES,
        "defaults": {"evaluators": EVALUATORS},
    }
    data.update(kwargs)
    return Suite.model_validate(data)


@pytest.fixture
def store(tmp_path: Path) -> Iterator[SQLiteStore]:
    with SQLiteStore(tmp_path / "runs.db") as db:
        yield db


@pytest.fixture(autouse=True)
def _no_model_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)


# -- running and persisting -------------------------------------------------------


@pytest.mark.parametrize("agent", [echo_agent, async_echo_agent])
def test_runs_every_case_and_persists_experiment(
    agent: Any, store: SQLiteStore, registry: EvaluatorRegistry
) -> None:
    result = run_suite(make_suite(agent), store=store, registry=registry, name="exp-1")

    assert isinstance(result, ExperimentRun)
    assert result.total == 2
    assert result.passed == 2
    assert result.pass_rate == 1.0
    assert result.errors == 0
    assert [c.case_id for c in result.cases] == ["hello", "world"]

    experiment = store.get_experiment(result.experiment.id)
    assert experiment == result.experiment
    assert experiment.name == "exp-1"
    assert experiment.suite == "toy"
    assert not experiment.is_baseline

    traces = store.list_traces(experiment_id=experiment.id)
    assert len(traces) == 2
    stored_results = store.list_eval_results(experiment.id)
    for run in result.runs:
        assert run.trace.output == run.case_id
        assert store.get_trace(run.trace.id) == run.trace
        assert stored_results[run.trace.id] == run.eval_results
        assert [r.evaluator for r in run.eval_results] == ["no_error", "exact_match"]
        # The LLM span recorded inside the agent belongs to this run's trace:
        # tracing context reached the worker thread / task.
        llm_spans = [s for s in run.trace.spans if s.kind is SpanKind.LLM_CALL]
        assert len(llm_spans) == 1
        assert llm_spans[0].trace_id == run.trace.id
        assert run.trace.metadata == {
            "case_id": run.case_id,
            "repetition": 0,
            "suite": "toy",
            "experiment": experiment.id,
        }


def test_repetitions_run_each_case_n_times(store: SQLiteStore, registry: EvaluatorRegistry) -> None:
    result = run_suite(make_suite(repetitions=3), store=store, registry=registry)

    assert result.total == 6
    for case in result.cases:
        assert [run.repetition for run in case.runs] == [0, 1, 2]
        assert case.pass_rate == 1.0
        assert case.mean_scores() == {"no_error": 1.0, "exact_match": 1.0}
    assert [(r.case_id, r.repetition) for r in result.runs] == [
        ("hello", 0),
        ("hello", 1),
        ("hello", 2),
        ("world", 0),
        ("world", 1),
        ("world", 2),
    ]
    found = store.list_traces(metadata={"case_id": "world", "repetition": 2})
    assert len(found) == 1
    assert result.experiment.config["repetitions"] == 3


def test_overrides_win_over_suite_settings(registry: EvaluatorRegistry) -> None:
    result = run_suite(make_suite(repetitions=3), repetitions=1, concurrency=4, registry=registry)
    assert result.total == 2
    assert result.experiment.config["repetitions"] == 1
    assert result.experiment.config["concurrency"] == 4


def test_without_store_results_are_only_returned(registry: EvaluatorRegistry) -> None:
    result = run_suite(make_suite(), registry=registry)
    assert result.pass_rate == 1.0


def test_suite_without_evaluators_passes_when_no_error() -> None:
    suite = Suite(name="bare", agent=echo_agent, cases=[{"id": "a", "input": "x"}])
    result = run_suite(suite)
    assert result.runs[0].eval_results == []
    assert result.runs[0].passed


def test_failed_evaluator_fails_the_run(registry: EvaluatorRegistry) -> None:
    suite = make_suite(cases=[{"id": "a", "input": "x", "expected": "y"}])
    result = run_suite(suite, registry=registry)
    run = result.runs[0]
    assert run.error is None
    assert not run.passed
    assert result.pass_rate == 0.0


def test_suite_loaded_from_path(write: WriteFile, registry: EvaluatorRegistry) -> None:
    write(
        "toy_agent.py",
        """
        def run_agent(text):
            return text.upper()
        """,
    )
    path = write(
        "suite.yaml",
        """
        name: from-file
        agent: toy_agent:run_agent
        repetitions: 2
        cases:
          - {id: a, input: abc, expected: ABC, evaluators: [exact_match]}
        """,
    )
    result = run_suite(path, registry=registry)
    assert result.suite.name == "from-file"
    assert result.total == 2
    assert result.pass_rate == 1.0


def test_unserialisable_output_is_converted() -> None:
    class Answer:
        def __repr__(self) -> str:
            return "Answer()"

    suite = Suite(name="obj", agent=lambda _: Answer(), cases=[{"id": "a", "input": "x"}])
    assert run_suite(suite).runs[0].trace.output == "Answer()"


def test_async_entry_point(registry: EvaluatorRegistry) -> None:
    async def main() -> ExperimentRun:
        return await arun_suite(make_suite(async_echo_agent), registry=registry)

    assert asyncio.run(main()).pass_rate == 1.0


# -- concurrency -------------------------------------------------------------------


@pytest.mark.parametrize("limit", [1, 2, 3])
def test_async_concurrency_bound_is_respected(limit: int) -> None:
    active = 0
    peak = 0

    async def agent(_: object) -> str:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return "ok"

    result = run_suite(make_suite(agent, defaults={}, repetitions=4, concurrency=limit))
    assert result.total == 8
    assert peak == limit


def test_sync_concurrency_bound_is_respected() -> None:
    lock = threading.Lock()
    active = 0
    peak = 0

    def agent(_: object) -> str:
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            active -= 1
        return "ok"

    result = run_suite(make_suite(agent, defaults={}, repetitions=3, concurrency=2))
    assert result.total == 6
    assert peak == 2


def test_concurrent_runs_keep_separate_traces(store: SQLiteStore) -> None:
    @agentlens.wrap_tool
    async def tool(text: str) -> str:
        await asyncio.sleep(0.005)
        return text

    async def agent(text: str) -> str:
        await tool(text)
        await tool(text)
        return text

    result = run_suite(make_suite(agent, defaults={}, repetitions=3, concurrency=6), store=store)
    for run in result.runs:
        assert [s.input for s in run.trace.spans] == [{"text": run.case_id}] * 2
        assert {s.trace_id for s in run.trace.spans} == {run.trace.id}


# -- failures ----------------------------------------------------------------------


def test_agent_exception_is_recorded_not_raised(
    store: SQLiteStore, registry: EvaluatorRegistry
) -> None:
    def agent(text: str) -> str:
        if text == "world":
            raise ValueError("boom")
        return text

    result = run_suite(make_suite(agent, repetitions=2), store=store, registry=registry)

    assert result.total == 4
    assert result.errors == 2
    assert result.passed == 2
    assert result.pass_rate == 0.5
    failed = result.case("world")
    assert failed.pass_rate == 0.0
    assert failed.errors == 2
    for run in failed.runs:
        assert run.error == "ValueError: boom"
        assert not run.timed_out
        assert not run.passed
        no_error = run.eval_results[0]
        assert no_error.evaluator == "no_error"
        assert not no_error.passed
        assert store.get_trace(run.trace.id) is not None
    assert result.case("hello").pass_rate == 1.0


def test_async_timeout_cancels_the_run(registry: EvaluatorRegistry) -> None:
    cancelled = []

    async def agent(text: str) -> str:
        if text == "world":
            try:
                await asyncio.sleep(5)
            except asyncio.CancelledError:
                cancelled.append(text)
                raise
        return text

    result = run_suite(make_suite(agent, timeout_s=0.05), registry=registry)

    slow = result.case("world").runs[0]
    assert slow.timed_out
    assert slow.error == "RunTimeoutError: run exceeded timeout_s=0.05"
    assert not slow.passed
    assert cancelled == ["world"]
    assert result.case("hello").runs[0].passed
    assert result.experiment.config["timeout_s"] == 0.05


def test_sync_timeout_is_recorded() -> None:
    def agent(_: object) -> str:
        time.sleep(0.3)
        return "late"

    suite = Suite(name="slow", agent=agent, cases=[{"id": "a", "input": "x"}])
    run = run_suite(suite, timeout_s=0.02).runs[0]
    assert run.timed_out
    assert run.trace.output is None


def test_store_failure_propagates() -> None:
    class BrokenStore(SQLiteStore):
        def save_trace(self, trace: Trace, *, experiment_id: str | None = None) -> None:
            raise StoreError("disk full")

    with BrokenStore(":memory:") as broken, pytest.raises(StoreError, match="disk full"):
        run_suite(make_suite(defaults={}, repetitions=3, concurrency=2), store=broken)


def test_redaction_failure_drops_content(store: SQLiteStore) -> None:
    def exploding(value: Any) -> Any:
        raise RuntimeError("bad redactor")

    suite = Suite(name="r", agent=echo_agent, cases=[{"id": "a", "input": "secret"}])
    run = run_suite(suite, store=store, redactor=exploding).runs[0]
    assert run.error == REDACTION_FAILED
    assert run.trace.input is None
    assert run.trace.spans == []
    assert run.trace.end_time is not None
    assert store.get_trace(run.trace.id) == run.trace


def test_redactor_none_keeps_raw_trace() -> None:
    suite = Suite(name="r", agent=echo_agent, cases=[{"id": "a", "input": "sk-" + "a" * 40}])
    run = run_suite(suite, redactor=None).runs[0]
    assert run.trace.input == "sk-" + "a" * 40


def test_bad_evaluator_fails_before_anything_runs(store: SQLiteStore) -> None:
    calls = []
    suite = Suite.model_validate(
        {
            "name": "bad",
            "agent": calls.append,
            "cases": [{"id": "a", "input": "x", "evaluators": ["nope"]}],
        },
        context={"agentlens.suite.skip_checks": True},
    )
    with pytest.raises(UnknownEvaluatorError):
        run_suite(suite, store=store)
    assert calls == []
    assert store.list_experiments() == []


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"repetitions": 0}, "repetitions and concurrency"),
        ({"concurrency": 0}, "repetitions and concurrency"),
        ({"timeout_s": 0}, "timeout_s"),
    ],
)
def test_invalid_overrides(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        run_suite(make_suite(), **kwargs)


# -- model convention --------------------------------------------------------------


def test_model_passed_to_agents_that_accept_it(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: list[str | None] = []

    def agent(text: str, *, model: str | None = None) -> str:
        seen.append(model)
        return text

    suite = make_suite(agent, defaults={}, cases=[CASES[0]])
    run_suite(suite)
    run_suite(suite.model_copy(update={"model": "mock:suite"}))
    monkeypatch.setenv("AGENTLENS_MODEL", "mock:env")
    run_suite(suite)
    result = run_suite(suite.model_copy(update={"model": "mock:suite"}), model="mock:override")
    assert seen == [None, "mock:suite", "mock:env", "mock:override"]
    assert result.experiment.config["model"] == "mock:override"


def test_model_reaches_the_llm_client() -> None:
    suite = make_suite(echo_agent, defaults={}, cases=[CASES[0]], model="mock:toy-model")
    run = run_suite(suite).runs[0]
    assert run.trace.spans[0].attributes["model"] == "toy-model"


def test_model_not_passed_to_agents_without_the_parameter() -> None:
    def agent(text: str, **kwargs: Any) -> str:
        assert kwargs == {}
        return text

    result = run_suite(make_suite(agent, defaults={}), model="mock:x")
    assert result.pass_rate == 1.0


def test_uninspectable_agent_gets_no_model(monkeypatch: pytest.MonkeyPatch) -> None:
    def broken_signature(_: object) -> inspect.Signature:
        raise ValueError("no signature")

    monkeypatch.setattr(runner_module.inspect, "signature", broken_signature)
    assert not runner_module._accepts_model(echo_agent)


# -- experiment metadata -------------------------------------------------------------


def test_experiment_config_and_default_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner_module,
        "experiment_config",
        lambda suite, **kw: {"captured": True, "extra": dict(kw["extra"] or {})},
    )
    suite = make_suite(defaults={})
    result = run_suite(suite, config={"prompt_version": "v2"})
    experiment = result.experiment
    assert experiment.config == {"captured": True, "extra": {"prompt_version": "v2"}}
    assert experiment.name == default_experiment_name(suite, experiment.created_at)
    assert experiment.name.startswith("toy-")


def test_experiment_config_is_collected() -> None:
    suite = make_suite(defaults={}, metadata={"team": "support"})
    config = run_suite(suite, config={"prompt_version": "v2"}).experiment.config
    assert config["suite_hash"] == suite.content_hash()
    assert config["agentlens_version"] == agentlens.__version__
    assert config["prompt_version"] == "v2"
    assert config["suite_metadata"] == {"team": "support"}
    assert "git_sha" in config


# -- progress ------------------------------------------------------------------------


def test_progress_callback(caplog: pytest.LogCaptureFixture) -> None:
    calls: list[tuple[str, int, int]] = []

    def progress(result: RunResult, completed: int, total: int) -> None:
        calls.append((result.case_id, completed, total))
        if completed == 1:
            raise RuntimeError("progress bar broke")

    with caplog.at_level(logging.ERROR, logger="agentlens.suite"):
        result = run_suite(make_suite(defaults={}, repetitions=2), progress=progress)
    assert result.total == 4
    assert [c[1:] for c in calls] == [(1, 4), (2, 4), (3, 4), (4, 4)]
    assert "progress callback failed" in caplog.text


# -- result objects ------------------------------------------------------------------


def test_result_objects_edge_cases() -> None:
    case = Case(id="a", input="x")
    empty = CaseResult(case, [])
    assert empty.pass_rate == 0.0
    assert empty.mean_scores() == {}
    run = ExperimentRun(
        experiment=Experiment(name="e", suite="s"),
        suite=make_suite(),
        cases=[empty],
    )
    assert run.pass_rate == 0.0
    assert run.case("a") is empty
    with pytest.raises(KeyError):
        run.case("missing")

    trace = Trace(name="s")
    scored = CaseResult(
        case,
        [
            RunResult("a", 0, trace, [EvalResult(evaluator="e", score=1.0, passed=True)]),
            RunResult("a", 1, trace, [EvalResult(evaluator="e", score=0.0, passed=False)]),
        ],
    )
    assert scored.mean_scores() == {"e": 0.5}
    assert scored.passed == 1


def test_timeout_error_type() -> None:
    assert issubclass(RunTimeoutError, TimeoutError)
