"""The pytest plugin, exercised in isolated pytest runs (``pytester``) with a MockLLM toy agent.

The repository's own test run blocks the plugin (``-p no:agentlens`` in pyproject, so
coverage sees every import); the inner ``pytester`` runs load it from the entry point
like any project that installs AgentLens.
"""

from __future__ import annotations

import subprocess
import sys

import pytest

from agentlens.integrations.pytest_plugin import (
    CaseFailedError,
    assert_passed,
    format_case_result,
)
from agentlens.schema import Case, EvalResult, Trace
from agentlens.store import SQLiteStore
from agentlens.suite import CaseResult, RunResult

pytest_plugins = ["pytester"]

AGENT = """
import agentlens
from agentlens.llm import LLMRequest, Message, MockLLM


def shout(question: str) -> str:
    llm = agentlens.wrap_llm(MockLLM(lambda request: request.messages[-1].content.upper()))
    return llm.complete(LLMRequest(messages=[Message.user(question)])).content


def explode(question: str) -> str:
    raise ValueError("boom")
"""

SUITE = """\
name: toy
agent: toy_agent:{agent}
defaults:
  evaluators: [no_error, exact_match]
cases:
  - {{id: ok, input: "hi", expected: "HI", tags: [smoke]}}
  - {{id: bad, input: "hi", expected: "nope"}}
"""


@pytest.fixture
def project(pytester: pytest.Pytester) -> pytest.Pytester:
    """A project with the toy agent, ``suite.yaml`` (one passing, one failing case) and
    a normal test file."""
    pytester.makepyfile(toy_agent=AGENT)
    pytester.makefile(".yaml", suite=SUITE.format(agent="shout"))
    pytester.makepyfile(test_normal="def test_normal():\n    assert True\n")
    return pytester


# ---------------------------------------------------------------------- inert


def test_plugin_is_inert_without_options(project: pytest.Pytester) -> None:
    result = project.runpytest()
    result.assert_outcomes(passed=1)  # suite.yaml is not collected


def test_help_lists_the_options(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--help")
    result.stdout.fnmatch_lines(
        ["*AgentLens agent suites*", "*--agentlens=SUITE*", "*--agentlens-db=PATH*"]
    )
    assert "--agentlens-repetitions=N" in result.stdout.str()


def test_fixture_is_registered(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--fixtures")
    result.stdout.fnmatch_lines(["agentlens_run*", "*Run an agent on one case*"])


# ---------------------------------------------------------------------- suites


def test_collects_one_item_per_case_next_to_normal_tests(project: pytest.Pytester) -> None:
    result = project.runpytest("--agentlens", "suite.yaml", "--collect-only", "-q")
    result.stdout.fnmatch_lines(["suite.yaml::ok", "suite.yaml::bad", "test_normal.py::*"])


def test_item_location_names_suite_and_case(project: pytest.Pytester) -> None:
    items, _ = project.inline_genitems("--agentlens", "suite.yaml")
    case_items = [item for item in items if item.nodeid.startswith("suite.yaml")]
    assert [item.location for item in case_items] == [
        ("suite.yaml", None, "agentlens toy::ok"),
        ("suite.yaml", None, "agentlens toy::bad"),
    ]


def test_failure_report_lists_evaluator_reasons(project: pytest.Pytester) -> None:
    result = project.runpytest("--agentlens", "suite.yaml")
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(
        [
            "*_ agentlens toy::bad _*",
            "agentlens case 'bad' of suite 'toy': 0/1 repetition passed",
            '  input: "hi"',
            '  expected: "nope"',
            "  repetition 0: FAILED (trace *)",
            '    output: "HI"',
            "    exact_match: score 0.00: *",
        ]
    )
    out = result.stdout.str()
    assert "no_error:" not in out  # passing evaluators are not listed
    assert "Traceback" not in out
    assert "runtest" not in out
    assert "FAILED suite.yaml::bad" in out


def test_agent_errors_are_reported(project: pytest.Pytester) -> None:
    project.makefile(".yaml", suite=SUITE.format(agent="explode"))
    result = project.runpytest("--agentlens", "suite.yaml", "-p", "no:cacheprovider")
    result.assert_outcomes(passed=1, failed=2)
    result.stdout.fnmatch_lines(
        ["    error: ValueError: boom", "    no_error: score 0.00: *ValueError: boom*"]
    )


def test_keyword_selection_runs_only_selected_cases(project: pytest.Pytester) -> None:
    db = project.path / "lens.db"
    result = project.runpytest("--agentlens", "suite.yaml", "--agentlens-db", "lens.db", "-k", "ok")
    result.assert_outcomes(passed=1, deselected=2)
    with SQLiteStore(db) as store:
        (experiment,) = store.list_experiments(suite="toy")
        traces = store.list_traces(experiment_id=experiment.id)
    assert [t.metadata["case_id"] for t in traces] == ["ok"]


@pytest.mark.parametrize(("keyword", "selected"), [("smoke", 1), ("toy", 2), ("bad", 1)])
def test_tags_and_suite_name_are_keywords(
    project: pytest.Pytester, keyword: str, selected: int
) -> None:
    result = project.runpytest("--agentlens", "suite.yaml", "-k", keyword, "--collect-only", "-q")
    assert f"{selected}/3 tests collected" in result.stdout.str()


def test_node_id_selection(project: pytest.Pytester) -> None:
    result = project.runpytest("--agentlens", "suite.yaml", "suite.yaml::ok")
    result.assert_outcomes(passed=1)  # neither the other case nor test_normal.py


def test_repetitions_and_db(project: pytest.Pytester) -> None:
    result = project.runpytest(
        "--agentlens", "suite.yaml", "--agentlens-repetitions", "3", "--agentlens-db", "out/a.db"
    )
    result.assert_outcomes(passed=2, failed=1)
    result.stdout.fnmatch_lines(["agentlens case 'bad' of suite 'toy': 0/3 repetitions passed"])
    with SQLiteStore(project.path / "out" / "a.db") as store:
        (experiment,) = store.list_experiments(suite="toy")
        assert experiment.config["repetitions"] == 3
        assert len(store.list_traces(experiment_id=experiment.id)) == 6
        results = store.list_eval_results(experiment.id)
    assert len(results) == 6


def test_option_is_repeatable_and_deduplicated(project: pytest.Pytester) -> None:
    project.makefile(".yaml", other=SUITE.format(agent="shout").replace("toy", "other", 1))
    result = project.runpytest(
        "--agentlens", "suite.yaml", "--agentlens", "other.yaml", "--agentlens", "./suite.yaml"
    )
    result.assert_outcomes(passed=3, failed=2)


def test_suite_inside_a_collected_directory_is_collected_once(
    project: pytest.Pytester,
) -> None:
    result = project.runpytest(".", "--agentlens", "suite.yaml", "--collect-only", "-q")
    assert "3 tests collected" in result.stdout.str()


def test_junit_properties(project: pytest.Pytester) -> None:
    project.runpytest("--agentlens", "suite.yaml", "--junitxml", "out.xml")
    xml = (project.path / "out.xml").read_text()
    assert 'name="agentlens_pass_rate" value="1.0"' in xml
    assert 'name="agentlens_score.exact_match" value="0.0"' in xml
    assert 'name="agentlens_experiment"' in xml


# ---------------------------------------------------------------------- errors


def test_invalid_suite_is_a_collection_error(project: pytest.Pytester) -> None:
    project.makefile(".yaml", suite="name: toy\nagent: toy_agent:shout\ncases: []\nbogus: 1\n")
    result = project.runpytest("--agentlens", "suite.yaml")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*ERROR collecting suite.yaml*", "invalid suite *", "*bogus*"])
    assert "Traceback" not in result.stdout.str()


def test_unexpected_collection_errors_keep_the_traceback(
    project: pytest.Pytester, monkeypatch: pytest.MonkeyPatch
) -> None:
    def broken(path: object) -> None:
        raise RuntimeError("disk on fire")

    monkeypatch.setattr("agentlens.suite.load_suite", broken)
    result = project.runpytest("--agentlens", "suite.yaml")
    result.assert_outcomes(errors=1)
    result.stdout.fnmatch_lines(["*ERROR collecting suite.yaml*", "*RuntimeError: disk on fire"])


def test_agent_that_cannot_be_imported_fails_every_case(project: pytest.Pytester) -> None:
    project.makefile(".yaml", suite=SUITE.format(agent="missing"))
    result = project.runpytest("--agentlens", "suite.yaml")
    result.assert_outcomes(passed=1, failed=2)
    lines = [line for line in result.stdout.lines if line.startswith("suite 'toy' could not")]
    assert len(lines) == 2
    assert lines[0].startswith("suite 'toy' could not run: AgentImportError: ")
    assert "missing" in lines[0]


def test_missing_suite_file_is_a_usage_error(pytester: pytest.Pytester) -> None:
    result = pytester.runpytest("--agentlens", "nope.yaml")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--agentlens: suite file not found: nope.yaml*"])


def test_repetitions_must_be_positive(project: pytest.Pytester) -> None:
    result = project.runpytest("--agentlens", "suite.yaml", "--agentlens-repetitions", "0")
    assert result.ret == pytest.ExitCode.USAGE_ERROR
    result.stderr.fnmatch_lines(["*--agentlens-repetitions must be >= 1*"])


# ---------------------------------------------------------------------- fixture


FIXTURE_TESTS = """
from agentlens.integrations.pytest_plugin import assert_passed
from toy_agent import shout


def test_pass(agentlens_run):
    result = agentlens_run(shout, {"id": "hi", "input": "hi", "expected": "HI"},
                           evaluators=["no_error", "exact_match"])
    assert_passed(result)
    assert result.mean_scores() == {"no_error": 1.0, "exact_match": 1.0}


def test_fail(agentlens_run):
    result = agentlens_run("toy_agent:shout",
                           {"id": "hi", "input": "hi", "expected": "nope",
                            "evaluators": [{"contains": {"text": "xyz"}}]},
                           evaluators=["exact_match"], repetitions=2)
    assert_passed(result, suite="mine")
"""


def test_fixture_runs_and_reports(project: pytest.Pytester) -> None:
    project.makepyfile(test_fixture=FIXTURE_TESTS)
    result = project.runpytest("test_fixture.py", "--agentlens-db", "f.db")
    result.assert_outcomes(passed=1, failed=1)
    result.stdout.fnmatch_lines(
        [
            '>*assert_passed(result, suite="mine")',
            "E*CaseFailedError: agentlens case 'hi' of suite 'mine': 0/2 repetitions passed",
            "E*exact_match: score 0.00: *",
            "E*contains: score 0.00: *",
        ]
    )
    with SQLiteStore(project.path / "f.db") as store:
        assert {e.suite for e in store.list_experiments()} == {"test_pass", "test_fail"}


def test_fixture_uses_repetitions_option(project: pytest.Pytester) -> None:
    project.makepyfile(
        test_reps="""
        from toy_agent import shout

        def test_reps(agentlens_run):
            result = agentlens_run(shout, {"id": "a", "input": "x"}, model="mock")
            assert len(result.runs) == 4 and result.pass_rate == 1.0
        """
    )
    project.runpytest("test_reps.py", "--agentlens-repetitions", "4").assert_outcomes(passed=1)


# ---------------------------------------------------------------------- report helpers


def _run(
    rep: int, *, error: str | None = None, results: list[EvalResult] | None = None
) -> RunResult:
    return RunResult("c", rep, Trace(name="t", output="out", error=error), results or [])


def _case(**fields: object) -> Case:
    return Case.model_validate({"id": "c", "input": "x" * 300, **fields})


def test_format_case_result_truncates_and_handles_missing_reasons() -> None:
    failing = EvalResult(evaluator="judge", score=0.25, passed=False)
    result = CaseResult(_case(), [_run(0, results=[failing]), _run(1)])
    text = format_case_result(result)
    assert text.splitlines()[0] == "agentlens case 'c': 1/2 repetitions passed"
    assert f'  input: "{"x" * 196}...' in text
    assert "expected" not in text
    assert "    judge: score 0.25: (no reason given)" in text
    assert "repetition 1" not in text


def test_assert_passed() -> None:
    assert_passed(CaseResult(_case(), [_run(0)]))
    with pytest.raises(CaseFailedError) as info:
        assert_passed(CaseResult(_case(expected={"a": 1}), [_run(0, error="KeyError: 'a'")]))
    assert isinstance(info.value, AssertionError)
    assert info.value.result.case_id == "c"
    assert '  expected: {"a": 1}' in str(info.value)
    assert "    error: KeyError: 'a'" in str(info.value)


def test_loading_the_plugin_does_not_import_the_runner() -> None:
    # pytest imports the plugin at startup in every project that installs AgentLens:
    # the suite runner and the store load lazily, on first use.
    code = (
        "import sys, agentlens.integrations.pytest_plugin; "
        "print(sorted(m for m in ('agentlens.suite', 'agentlens.store') if m in sys.modules))"
    )
    out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == "[]"
