"""The regression story of ``docs/regression-story.md``, end to end and offline (FR-D2).

A pull request "simplifies" the support agent's system prompt (``system_v2.txt``)
and drops the "$200" refund limit, so the agent refunds an $899 laptop instead of
escalating: a tool-usage regression. The whole pipeline must catch it:
run the suite on v1 -> mark it as the baseline -> run v2 -> compare -> gates fail
(exit 1) -> reports in every format name the regressed cases. Once through the
Python API, once through the CLI exactly as CI runs it.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from examples.support_agent import PROMPT_VERSION_ENV
from typer.testing import CliRunner

from agentlens.cli import app
from agentlens.compare import (
    EXIT_GATE_FAILED,
    EXIT_OK,
    compare_experiments,
    evaluate_gates,
    experiment_runs,
    load_gates,
)
from agentlens.compare.model import Status
from agentlens.report import ComparisonReport, render, render_html
from agentlens.store import SQLiteStore
from agentlens.suite import get_baseline, load_suite, run_suite, set_baseline

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "support_agent"
SUITE = EXAMPLE / "suite.yaml"
GATES = EXAMPLE / "gates.yaml"

REGRESSED = {"refund-over-limit", "multi-broken-over-limit"}
"""The cases where the agent now calls issue_refund above $200 instead of escalate_to_human."""


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)
    monkeypatch.delenv("AGENTLENS_DB", raising=False)
    monkeypatch.delenv(PROMPT_VERSION_ENV, raising=False)
    monkeypatch.chdir(ROOT)  # `examples.support_agent` is imported from the repository root


def test_pipeline_catches_the_prompt_regression(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = SQLiteStore(tmp_path / "agentlens.db")
    suite = load_suite(SUITE)

    # main: the reviewed v1 prompt, recorded as the suite's baseline.
    baseline = run_suite(suite, store=store, name="main")
    assert baseline.passed == baseline.total == 27
    set_baseline(store, baseline.experiment)
    marked = get_baseline(store, "support-agent")
    assert marked is not None
    assert marked.id == baseline.experiment.id

    # The pull request: the "simplified" v2 prompt.
    monkeypatch.setenv(PROMPT_VERSION_ENV, "v2")
    candidate = run_suite(suite, store=store, name="pr-prompt-v2")
    failed = {run.case_id for run in candidate.runs if not run.passed}
    assert failed == REGRESSED
    for run in candidate.runs:
        if run.case_id in REGRESSED:
            assert "issue_refund" in run.trace.tool_calls
            assert "escalate_to_human" not in run.trace.tool_calls

    comparison = compare_experiments(store, marked, "pr-prompt-v2")
    assert {case.case_id for case in comparison.case_regressions} == REGRESSED
    assert comparison.case("refund-over-limit").status is Status.REGRESSION
    assert "tool_sequence" in comparison.case("refund-over-limit").regressed_evaluators
    assert comparison.metric("pass_rate").delta == pytest.approx(-2 / 27)
    assert comparison.metric("steps").delta == 0  # same number of calls, different tools

    gates = evaluate_gates(load_gates(GATES), comparison=comparison)
    assert not gates.passed
    assert gates.exit_code == EXIT_GATE_FAILED
    assert {(g.metric, g.rule) for g in gates.failures} == {
        ("pass_rate", "min"),
        ("pass_rate", "min_delta"),
        ("case_regressions", "max"),
    }

    report = ComparisonReport(comparison=comparison, gates=gates)
    text = render(report, "text")
    markdown = render(report, "markdown")
    html = render_html(
        report,
        runs=candidate.runs,
        baseline_runs=experiment_runs(store, marked),
    )
    data = json.loads(render(report, "json"))
    for case_id in REGRESSED:
        assert case_id in text
        assert f"`{case_id}`" in markdown
        assert case_id in html
    assert "gates FAILED" in text
    assert "AgentLens: gates failed" in markdown
    assert "case_regressions" in markdown
    assert html.startswith("<!DOCTYPE html>") or html.lower().startswith("<!doctype html>")
    assert data["gates"]["passed"] is False

    # Control: the same PR without the prompt change passes every gate.
    monkeypatch.delenv(PROMPT_VERSION_ENV)
    again = run_suite(suite, store=store, name="pr-no-change")
    clean = evaluate_gates(
        load_gates(GATES), comparison=compare_experiments(store, marked, again.experiment)
    )
    assert clean.passed
    assert clean.exit_code == EXIT_OK


def test_cli_story_exits_1_and_lists_the_regressions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner = CliRunner()
    db = ["--db", str(tmp_path / "agentlens.db")]
    suite, gates = str(SUITE), str(GATES)

    first = runner.invoke(app, [*db, "run", suite, "--name", "main", "--set-baseline", "-q"])
    assert first.exit_code == 0, first.output
    assert "marked as the baseline" in first.output

    # The candidate alone: its absolute gates fail already (92.6% < 95%).
    monkeypatch.setenv(PROMPT_VERSION_ENV, "v2")
    second = runner.invoke(app, [*db, "run", suite, "--name", "candidate", "-q", "--gates", gates])
    assert second.exit_code == EXIT_GATE_FAILED, second.output
    monkeypatch.delenv(PROMPT_VERSION_ENV)

    out = tmp_path / "report.json"
    html = tmp_path / "report.html"
    compared = runner.invoke(
        app,
        [
            *db,
            "compare",
            "--candidate",
            "candidate",
            "--gates",
            gates,
            "--json",
            str(out),
            "--html",
            str(html),
        ],
    )
    assert compared.exit_code == EXIT_GATE_FAILED, compared.output
    assert "case regressions (2)" in compared.output
    assert "gates FAILED" in compared.output
    for case_id in REGRESSED:
        assert case_id in compared.output
        assert case_id in html.read_text(encoding="utf-8")

    markdown = runner.invoke(app, [*db, "report", str(out), "--format", "markdown"])
    assert markdown.exit_code == 0, markdown.output
    assert markdown.output.startswith("<!-- agentlens-report -->")
    assert "### ❌ AgentLens: gates failed" in markdown.output
    assert "2 cases regressed" in markdown.output
    for case_id in REGRESSED:
        assert f"`{case_id}`" in markdown.output
