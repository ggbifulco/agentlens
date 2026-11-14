from __future__ import annotations

import importlib
import json
import runpy
import sys

import pytest

from agentlens import __version__
from agentlens.cli import EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE, main
from agentlens.cli._common import auto_format
from agentlens.report import ComparisonReport, ExperimentReport, ReportFormat, load_report
from agentlens.store import SQLiteStore
from agentlens.suite import get_baseline
from tests.cli.conftest import Toy, plain

cli_app = importlib.import_module("agentlens.cli.app")

# -- root ----------------------------------------------------------------------------


def test_help_lists_commands_and_exit_codes(toy: Toy) -> None:
    result = toy("--help", db=False)
    assert result.exit_code == EXIT_OK
    for command in ("run", "compare", "report", "baseline", "experiments", "traces", "export"):
        assert command in result.output
    assert "Exit codes" in result.output
    assert "AGENTLENS_DB" in result.output


@pytest.mark.parametrize(
    "command",
    [
        ["run"],
        ["compare"],
        ["report"],
        ["export"],
        ["import"],
        ["baseline", "set"],
        ["baseline", "unset"],
        ["baseline", "show"],
        ["experiments", "ls"],
        ["traces", "ls"],
        ["traces", "show"],
    ],
)
def test_every_command_has_help(toy: Toy, command: list[str]) -> None:
    result = toy(*command, "--help", db=False)
    assert result.exit_code == EXIT_OK, result.output
    assert "Usage:" in result.output


@pytest.mark.parametrize("group", [[], ["baseline"], ["experiments"], ["traces"]])
def test_groups_without_a_command_show_help(toy: Toy, group: list[str]) -> None:
    result = toy(*group, db=False)
    assert "Usage:" in result.output


def test_version(toy: Toy) -> None:
    result = toy("--version", db=False)
    assert result.exit_code == EXIT_OK
    assert result.output == f"agentlens {__version__}\n"


def test_python_dash_m(monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]) -> None:
    monkeypatch.setattr(sys, "argv", ["agentlens", "--version"])
    # agentlens ships a pytest plugin, so pytest marks the package for assertion rewriting
    # and its import hook (which has no get_code(), needed by runpy) would load __main__.
    monkeypatch.setattr(
        sys, "meta_path", [f for f in sys.meta_path if type(f).__name__ != "AssertionRewritingHook"]
    )
    with pytest.raises(SystemExit) as info:
        runpy.run_module("agentlens", run_name="__main__")
    assert info.value.code == 0
    assert __version__ in capsys.readouterr().out


def test_importing_main_module_does_not_run_the_cli() -> None:
    module = importlib.import_module("agentlens.__main__")
    assert module.main is main


def test_unknown_option_is_a_usage_error(toy: Toy) -> None:
    assert toy("run", "--bogus").exit_code == EXIT_USAGE


# -- run -----------------------------------------------------------------------------


def test_run_prints_summary_and_saves_experiment(toy: Toy) -> None:
    result = toy("run", str(toy.suite), "--name", "main")
    assert result.exit_code == EXIT_OK, result.output
    assert "[1/12] q0 #0 pass" in result.stderr
    assert "x #0 FAIL" in result.stderr
    out = result.stdout
    assert "experiment  main (" in out
    assert "runs        10/12 passed (83%)" in out
    assert "  x     0/2     0%            exact_match" in out
    assert f"saved experiment 'main' to {toy.db}" in out
    with SQLiteStore(toy.db) as store:
        (experiment,) = store.list_experiments()
        assert experiment.name == "main"
        assert not experiment.is_baseline
        assert len(store.list_traces(experiment_id=experiment.id)) == 12


def test_run_overrides_and_quiet(toy: Toy) -> None:
    result = toy(
        "run",
        str(toy.suite),
        "-n",
        "fast",
        "--model",
        "mock:test",
        "--repetitions",
        "1",
        "--concurrency",
        "3",
        "--timeout",
        "5",
        "--quiet",
    )
    assert result.exit_code == EXIT_OK, result.output
    assert result.stderr == ""
    assert "model       mock:test" in result.stdout
    with SQLiteStore(toy.db) as store:
        config = store.list_experiments(name="fast")[0].config
    assert (config["repetitions"], config["concurrency"], config["timeout_s"]) == (1, 3, 5.0)
    assert config["model"] == "mock:test"


def test_run_shows_agent_errors_in_progress(toy: Toy) -> None:
    result = toy("run", str(toy.new_suite), "-r", "1")
    assert result.exit_code == EXIT_OK
    assert "boom #0 FAIL (ValueError: boom)" in result.stderr
    assert "1 error" in result.stdout


def test_run_set_baseline(toy: Toy) -> None:
    result = toy("run", str(toy.suite), "--name", "main", "--set-baseline", "-q")
    assert result.exit_code == EXIT_OK
    assert "[baseline]" in result.stdout
    assert "marked as the baseline of suite 'toy'" in result.stdout
    with SQLiteStore(toy.db) as store:
        baseline = get_baseline(store, "toy")
    assert baseline is not None
    assert baseline.name == "main"


def test_run_gates_pass(toy: Toy) -> None:
    gates = toy.write("gates.yaml", "success_rate: {min: 0.5}\nregressions: {max: 0}\n")
    result = toy("run", str(toy.suite), "--gates", str(gates), "-q")
    assert result.exit_code == EXIT_OK
    assert "gates passed: 1 passed, 0 failed, 1 skipped" in result.stdout
    assert "no baseline comparison" in result.stdout


def test_run_gate_failure_exits_1_and_skips_baseline(toy: Toy) -> None:
    gates = toy.write("gates.yaml", "success_rate: {min: 0.99}\n")
    result = toy("run", str(toy.suite), "-g", str(gates), "--set-baseline", "-q")
    assert result.exit_code == EXIT_GATE_FAILED
    assert "gates FAILED" in result.stdout
    assert "not marked as the baseline: a gate failed" in result.stderr
    with SQLiteStore(toy.db) as store:
        assert get_baseline(store, "toy") is None
        assert len(store.list_experiments()) == 1  # the experiment is still saved


def test_run_invalid_suite_is_a_usage_error(toy: Toy) -> None:
    bad = toy.write(
        "bad.yaml",
        """
        name: toy
        agent: toy_agent:run_agent
        cases:
          - id: a
            input: hi
            evaluators: [no_eror]
        """,
    )
    result = toy("run", str(bad))
    assert result.exit_code == EXIT_USAGE
    assert "error: invalid suite" in result.stderr
    assert "bad.yaml:6:" in result.stderr
    assert "did you mean 'no_error'" in result.stderr
    assert not toy.db.exists()  # nothing ran, nothing created


def test_run_missing_suite_file(toy: Toy) -> None:
    result = toy("run", "nope.yaml")
    assert result.exit_code == EXIT_USAGE
    assert "error:" in result.stderr


def test_run_invalid_gates_fails_before_running(toy: Toy) -> None:
    gates = toy.write("gates.yaml", "succes_rate: {min: 0.9}\n")
    result = toy("run", str(toy.suite), "--gates", str(gates))
    assert result.exit_code == EXIT_USAGE
    assert "gates.yaml:1:1" in result.stderr
    assert not toy.db.exists()


def test_run_agent_import_error(toy: Toy) -> None:
    suite = toy.write(
        "s.yaml", "name: toy\nagent: toy_agent:missing\ncases: [{id: a, input: hi}]\n"
    )
    result = toy("run", str(suite))
    assert result.exit_code == EXIT_USAGE
    assert "missing" in result.stderr


@pytest.mark.parametrize(
    "args", [["--repetitions", "0"], ["--concurrency", "0"], ["--timeout", "0"]]
)
def test_run_rejects_bad_numbers(toy: Toy, args: list[str]) -> None:
    result = toy("run", str(toy.suite), *args)
    assert result.exit_code == EXIT_USAGE
    assert not toy.db.exists()


def test_db_from_environment(toy: Toy, monkeypatch: pytest.MonkeyPatch) -> None:
    env_db = toy.root / "env" / "agentlens.db"
    monkeypatch.setenv("AGENTLENS_DB", str(env_db))
    result = toy("run", str(toy.suite), "-q", db=False)
    assert result.exit_code == EXIT_OK
    assert env_db.exists()


def test_default_db_location(toy: Toy) -> None:
    result = toy("run", str(toy.suite), "-q", db=False)
    assert result.exit_code == EXIT_OK
    assert (toy.root / ".agentlens" / "agentlens.db").exists()


# -- compare -------------------------------------------------------------------------


def test_compare_against_the_suite_baseline(ran: Toy) -> None:
    result = ran("compare", "--candidate", "pr-1", "--resamples", "200")
    assert result.exit_code == EXIT_OK, result.output
    out = result.stdout
    assert "baseline   main (" in out
    assert "[baseline]" in out
    assert "candidate  pr-1 (" in out
    assert "5 paired, 1 added, 1 removed; paired bootstrap, 200 resamples, 95% CI" in out
    assert "case regressions (2)" in out
    assert "  q0    1 -> 0     exact_match" in out
    assert "case improvements (1)" in out
    assert "added cases (1): boom" in out
    assert "removed cases (1): gone" in out
    assert "gates" not in out


def test_compare_explicit_baseline_and_gates_pass(ran: Toy) -> None:
    gates = ran.write("gates.yaml", "success_rate: {min: 0.5}\n")
    result = ran("compare", "-b", "main", "-c", "main", "-g", str(gates), "--resamples", "50")
    assert result.exit_code == EXIT_OK
    assert "gates passed" in result.stdout


def test_compare_gate_failure_exits_1(ran: Toy) -> None:
    gates = ran.write("gates.yaml", "case_regressions: {max: 0}\nsuccess_rate: {min: 0.6}\n")
    result = ran("compare", "-c", "pr-1", "--gates", str(gates), "--resamples", "50")
    assert result.exit_code == EXIT_GATE_FAILED
    assert "FAIL case_regressions max 0" in result.stdout
    # absolute rules see every candidate case (3/6 with "boom"), not only the 5 paired (3/5)
    assert "pass_rate 0.5 is below min 0.6" in result.stdout


def test_compare_writes_json(ran: Toy) -> None:
    gates = ran.write("gates.yaml", "regressions: {max: 5}\n")
    out = ran.root / "reports" / "cmp.json"
    result = ran("compare", "-c", "pr-1", "-g", str(gates), "--json", str(out), "--seed", "7")
    assert result.exit_code == EXIT_OK
    assert "baseline   main" in result.stdout
    report = load_report(out)
    assert isinstance(report, ComparisonReport)
    assert report.comparison.seed == 7
    assert report.gates is not None
    assert report.gates.passed


def test_compare_json_to_stdout(ran: Toy) -> None:
    result = ran("compare", "-c", "pr-1", "--json", "-", "--resamples", "50", "--confidence", "0.9")
    assert result.exit_code == EXIT_OK
    data = json.loads(result.stdout)
    assert data["comparison"]["confidence"] == 0.9
    assert data["gates"] is None


def test_compare_without_a_suite_baseline(toy: Toy) -> None:
    assert toy("run", str(toy.suite), "-n", "main", "-q").exit_code == EXIT_OK
    result = toy("compare", "-c", "main")
    assert result.exit_code == EXIT_USAGE
    assert "suite 'toy' has no baseline" in result.stderr


@pytest.mark.parametrize(
    ("args", "message"),
    [
        (["-c", "nope"], "no experiment with id or name 'nope'"),
        (["-c", "pr-1", "-b", "nope"], "no experiment with id or name 'nope'"),
        (["-c", "pr-1", "--confidence", "1.5"], "between 0 and 1"),
        (["-c", "pr-1", "--resamples", "0"], "--resamples"),
        (["-c", "pr-1", "--gates", "missing.yaml"], "missing.yaml"),
    ],
)
def test_compare_usage_errors(ran: Toy, args: list[str], message: str) -> None:
    result = ran("compare", *args)
    assert result.exit_code == EXIT_USAGE
    assert message in plain(result.stderr)


def test_compare_requires_a_candidate(ran: Toy) -> None:
    assert ran("compare").exit_code == EXIT_USAGE


def test_compare_unwritable_json(ran: Toy) -> None:
    (ran.root / "file").write_text("")
    result = ran("compare", "-c", "pr-1", "--resamples", "10", "--json", "file/sub/cmp.json")
    assert result.exit_code == EXIT_USAGE
    assert "error:" in result.stderr


def test_read_commands_need_an_existing_database(toy: Toy) -> None:
    result = toy("compare", "-c", "pr-1")
    assert result.exit_code == EXIT_USAGE
    assert f"no AgentLens database at {toy.db}" in result.stderr
    assert not toy.db.exists()


def test_corrupt_database_is_a_usage_error(toy: Toy) -> None:
    toy.db.write_text("not a database" * 100)
    result = toy("experiments", "ls")
    assert result.exit_code == EXIT_USAGE
    assert "error:" in result.stderr


# -- report --------------------------------------------------------------------------


def test_report_experiment_text(ran: Toy) -> None:
    result = ran("report", "pr-1")
    assert result.exit_code == EXIT_OK, result.output
    assert "experiment  pr-1 (" in result.stdout
    assert "runs        6/12 passed (50%), 2 errors" in result.stdout
    assert "  boom  0/2     0%    2       exact_match, no_error" in result.stdout


def test_report_experiment_json(ran: Toy) -> None:
    result = ran("report", "main", "--format", "json")
    assert result.exit_code == EXIT_OK
    report = ExperimentReport.model_validate_json(result.stdout)
    assert report.experiment.is_baseline
    assert [case.case_id for case in report.cases] == ["q0", "q1", "q2", "q3", "x", "gone"]


def test_report_output_file_and_rerender(ran: Toy) -> None:
    out = ran.root / "out" / "main.json"
    result = ran("report", "main", "-f", "JSON", "-o", str(out))
    assert result.exit_code == EXIT_OK
    assert result.stdout == ""
    assert f"wrote {out}" in result.stderr
    again = ran("report", str(out))
    assert again.exit_code == EXIT_OK
    assert "experiment  main (" in again.stdout


def test_report_saved_comparison(ran: Toy) -> None:
    cmp = ran.root / "cmp.json"
    assert ran("compare", "-c", "pr-1", "--resamples", "20", "--json", str(cmp)).exit_code == 0
    result = ran("report", str(cmp), db=False)  # no database needed for a file
    assert result.exit_code == EXIT_OK
    assert "candidate  pr-1 (" in result.stdout


def test_report_planned_formats(ran: Toy) -> None:
    result = ran("report", "pr-1", "--format", "html")
    assert result.exit_code == EXIT_USAGE
    assert "not available yet" in result.stderr
    assert "#2" in result.stderr


@pytest.mark.parametrize(
    ("target", "message"),
    [("nope", "no experiment with id or name 'nope'"), ("missing.json", "no such file")],
)
def test_report_not_found(ran: Toy, target: str, message: str) -> None:
    result = ran("report", target)
    assert result.exit_code == EXIT_USAGE
    assert message in result.stderr


def test_report_bad_file(ran: Toy) -> None:
    bad = ran.write("bad.json", "[]")
    result = ran("report", str(bad))
    assert result.exit_code == EXIT_USAGE
    assert "not an AgentLens report" in result.stderr


def test_report_unknown_format(ran: Toy) -> None:
    assert ran("report", "main", "--format", "pdf").exit_code == EXIT_USAGE


# -- rich terminal report (auto format) ----------------------------------------------


class _Stdout:
    def __init__(self, tty: bool | None) -> None:
        self.tty = tty

    def isatty(self) -> bool:
        if self.tty is None:
            raise ValueError("I/O operation on closed file")
        return self.tty


@pytest.mark.parametrize(
    ("tty", "env", "expected"),
    [
        (True, {}, ReportFormat.RICH),
        (False, {}, ReportFormat.TEXT),
        (None, {}, ReportFormat.TEXT),
        (True, {"NO_COLOR": "1"}, ReportFormat.TEXT),
        (True, {"NO_COLOR": ""}, ReportFormat.RICH),  # empty means unset (no-color.org)
        (True, {"TERM": "dumb"}, ReportFormat.TEXT),
    ],
)
def test_auto_format(
    monkeypatch: pytest.MonkeyPatch, tty: bool | None, env: dict[str, str], expected: ReportFormat
) -> None:
    monkeypatch.delenv("NO_COLOR", raising=False)
    monkeypatch.setenv("TERM", "xterm-256color")
    for key, value in env.items():
        monkeypatch.setenv(key, value)
    monkeypatch.setattr(sys, "stdout", _Stdout(tty))
    assert auto_format() is expected


@pytest.fixture
def terminal(monkeypatch: pytest.MonkeyPatch) -> None:
    """Pretend stdout is an interactive terminal."""
    monkeypatch.setattr(cli_app, "auto_format", lambda: ReportFormat.RICH)


@pytest.mark.usefixtures("terminal")
def test_report_is_rich_on_a_terminal(ran: Toy) -> None:
    result = ran("report", "pr-1")
    assert result.exit_code == EXIT_OK, result.output
    assert "mean over cases" in result.stdout
    assert "Failures (3 of 6 cases)" in result.stdout
    assert " \n" not in result.stdout  # no trailing padding


@pytest.mark.usefixtures("terminal")
def test_report_to_a_file_defaults_to_text(ran: Toy) -> None:
    out = ran.root / "report.txt"
    assert ran("report", "pr-1", "-o", str(out)).exit_code == EXIT_OK
    assert out.read_text().startswith("experiment  pr-1 (")


def test_report_format_rich_when_piped(ran: Toy) -> None:
    result = ran("report", "pr-1", "--format", "rich")
    assert result.exit_code == EXIT_OK
    assert "Metrics" in result.stdout
    assert "\x1b[" not in result.stdout  # click strips colours when not on a terminal


@pytest.mark.usefixtures("terminal")
def test_run_and_compare_are_rich_on_a_terminal(ran: Toy) -> None:
    result = ran("compare", "-c", "pr-1", "--resamples", "50")
    assert result.exit_code == EXIT_OK
    assert "Case regressions (2)" in result.stdout
    run_result = ran("run", str(ran.suite), "-q")
    assert run_result.exit_code == EXIT_OK
    assert "✓ all" not in run_result.stdout  # "x" fails in the baseline suite
    assert "Failures (1 of 6 cases)" in run_result.stdout
    assert "saved experiment" in run_result.stdout


# -- run --json / report --format markdown -------------------------------------------


def test_run_writes_json_with_gates(toy: Toy) -> None:
    gates = toy.write("gates.yaml", "success_rate: {min: 0.5}\n")
    out = toy.root / "reports" / "run.json"
    result = toy("run", str(toy.suite), "-n", "main", "-g", str(gates), "-q", "--json", str(out))
    assert result.exit_code == EXIT_OK
    assert "experiment  main (" in result.stdout
    report = load_report(out)
    assert isinstance(report, ExperimentReport)
    assert report.gates is not None
    assert report.gates.passed


def test_run_json_to_stdout(toy: Toy) -> None:
    result = toy("run", str(toy.suite), "-n", "main", "--set-baseline", "-q", "--json", "-")
    assert result.exit_code == EXIT_OK
    report = ExperimentReport.model_validate_json(result.stdout)
    assert report.experiment.is_baseline
    assert "saved experiment 'main'" in result.stderr
    assert "marked as the baseline" in result.stderr


def test_run_unwritable_json(toy: Toy) -> None:
    (toy.root / "file").write_text("")
    result = toy("run", str(toy.suite), "-q", "--json", "file/sub/run.json")
    assert result.exit_code == EXIT_USAGE


def test_report_markdown_comparison_with_marker(ran: Toy) -> None:
    gates = ran.write("gates.yaml", "case_regressions: {max: 0}\n")
    cmp = ran.root / "cmp.json"
    compared = ran(
        "compare", "-c", "pr-1", "-g", str(gates), "--resamples", "20", "--json", str(cmp)
    )
    assert compared.exit_code == EXIT_GATE_FAILED
    out = ran.root / "report.md"
    result = ran(
        "report", str(cmp), "--format", "markdown", "--marker", "agentlens:toy", "-o", str(out)
    )
    assert result.exit_code == EXIT_OK
    text = out.read_text(encoding="utf-8")
    assert text.startswith("<!-- agentlens:toy -->\n### ❌ AgentLens: gates failed")
    assert "<details><summary>Case regressions (2)</summary>" in text
    assert "| ❌ **fail** | case_regressions |" in text


def test_report_markdown_experiment_default_marker(ran: Toy) -> None:
    result = ran("report", "main", "--format", "markdown")
    assert result.exit_code == EXIT_OK
    assert result.stdout.startswith("<!-- agentlens-report -->\n### ❌ AgentLens:")
    assert "Failing cases (1 of 6)" in result.stdout


def test_report_marker_needs_markdown(ran: Toy) -> None:
    result = ran("report", "main", "--format", "text", "--marker", "x")
    assert result.exit_code == EXIT_USAGE
    assert "only applies to --format markdown" in plain(result.stderr)


def test_report_invalid_marker(ran: Toy) -> None:
    result = ran("report", "main", "--format", "markdown", "--marker", "a -->")
    assert result.exit_code == EXIT_USAGE
    assert "invalid marker" in plain(result.stderr)
