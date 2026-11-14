"""The Markdown report: snapshot tests plus escaping, marker and truncation rules.

Snapshots live in ``tests/report/snapshots/``. After an intended change of the
layout, regenerate them and review the diff::

    AGENTLENS_UPDATE_SNAPSHOTS=1 uv run pytest tests/report/test_markdown.py
"""

from __future__ import annotations

import pytest

from agentlens.compare import Comparison, GateReport, Gates, compare_runs, evaluate_gates
from agentlens.report import (
    DEFAULT_MARKER,
    ComparisonReport,
    ExperimentReport,
    MarkdownRenderer,
    ReportFormat,
    get_renderer,
    marker_comment,
    render,
)
from agentlens.report.markdown import _esc
from agentlens.schema import Experiment
from tests.compare.helpers import run
from tests.report.test_terminal import (
    CANDIDATE,
    CREATED,
    _check_snapshot,
    _comparison_report,
    _experiment_report,
)


def test_comparison_snapshot() -> None:
    _check_snapshot("comparison.md", render(_comparison_report(), "markdown") + "\n")


def test_experiment_snapshot() -> None:
    _check_snapshot("experiment.md", render(_experiment_report(), "markdown") + "\n")


def test_markdown_is_a_registered_format() -> None:
    assert isinstance(get_renderer(ReportFormat.MARKDOWN), MarkdownRenderer)
    assert get_renderer("markdown").marker == DEFAULT_MARKER  # type: ignore[attr-defined]


def test_marker_comment() -> None:
    assert marker_comment() == "<!-- agentlens-report -->"
    assert marker_comment("agentlens:suite/a_b.c@1") == "<!-- agentlens:suite/a_b.c@1 -->"
    for bad in ("", "a b", "x-->", "<!--", "a\nb"):
        with pytest.raises(ValueError, match="invalid marker"):
            marker_comment(bad)


def test_custom_marker_and_no_marker() -> None:
    report = _comparison_report()
    assert MarkdownRenderer(marker="k1").comparison(report).startswith("<!-- k1 -->\n### ")
    text = MarkdownRenderer(marker=None).experiment(_experiment_report())
    assert text.startswith("### ❌ AgentLens:")
    assert "<!--" not in text


def test_invalid_renderer_options() -> None:
    with pytest.raises(ValueError, match="max_cases"):
        MarkdownRenderer(max_cases=0)
    with pytest.raises(ValueError, match="invalid marker"):
        MarkdownRenderer(marker="has space")


def test_user_text_is_escaped() -> None:
    assert _esc("a|b <script> & *x* [l](u) `c` \\ \n") == (
        "a\\|b &lt;script&gt; &amp; \\*x\\* \\[l\\](u) \\`c\\` \\\\  "
    )
    experiment = Experiment(name="<b>evil|name</b>", suite="s`uite", created_at=CREATED)
    runs = [run("case|<1>", passed=False)]
    text = MarkdownRenderer().experiment(ExperimentReport.from_runs(experiment, runs))
    assert "**&lt;b&gt;evil\\|name&lt;/b&gt;**" in text
    assert "suite `s'uite`" in text
    assert "| `case\\|<1>` | 0/1 |" in text  # code span: no HTML, pipe escaped for the table


def test_comparison_without_gates_or_experiments() -> None:
    same = compare_runs([run("a"), run("b")], [run("a"), run("b")], resamples=10, seed=None)
    text = MarkdownRenderer().comparison(ComparisonReport(comparison=same))
    assert "### ✅ AgentLens: no metric changed" in text
    assert "**candidate** vs **baseline** · 2 paired cases" in text
    assert "Gates" not in text
    assert "<details" not in text
    assert "baseline: in memory · candidate: in memory" in text
    assert "seed" not in text


def test_comparison_without_gates_regressed_is_a_failure() -> None:
    worse = compare_runs(
        [run(f"c{i}") for i in range(6)],
        [run(f"c{i}", passed=False) for i in range(6)],
        resamples=20,
    )
    text = MarkdownRenderer().comparison(ComparisonReport(comparison=worse))
    assert text.splitlines()[1] == ("### ❌ AgentLens: 2 metrics regressed · 6 cases regressed")


def test_empty_comparison_and_passing_gates() -> None:
    empty = Comparison(resamples=10, confidence=0.9)
    report = ComparisonReport(comparison=empty, gates=GateReport(passed=True))
    text = MarkdownRenderer().comparison(report)
    assert "### ✅ AgentLens: gates passed · nothing to compare" in text
    assert "#### Gates: ✅ passed (0 passed, 0 failed, 0 skipped)" in text
    assert "| Metric |" not in text
    assert "| Gate |" not in text


def test_case_improvements_and_only_baseline_experiment() -> None:
    cmp = compare_runs(
        [run("a", passed=False), run("gone")],
        [run("a")],
        resamples=10,
        baseline_experiment=Experiment(name="main", suite="toy", created_at=CREATED),
    )
    text = MarkdownRenderer().comparison(ComparisonReport(comparison=cmp))
    assert "<details><summary>Case improvements (1)</summary>" in text
    assert "**candidate** vs **main** · suite `toy` · 1 paired case (1 removed)" in text
    assert "**Removed cases** (1): `gone`" in text
    assert "**Added cases**" not in text


def test_long_case_lists_are_truncated() -> None:
    base = [run(f"c{i}") for i in range(5)]
    cand = [run(f"c{i}", passed=False) for i in range(5)]
    cand += [run(f"new{i}") for i in range(4)]
    cmp = compare_runs(base, cand, resamples=10)
    text = MarkdownRenderer(max_cases=2).comparison(ComparisonReport(comparison=cmp))
    assert "Case regressions (5)" in text
    assert "`c1`" in text
    assert "`c2`" not in text
    assert "…and 3 more." in text
    assert "**Added cases** (4): `new0`, `new1` …and 2 more" in text

    failing = [run(f"f{i}", passed=False) for i in range(3)]
    exp = MarkdownRenderer(max_cases=1).experiment(ExperimentReport.from_runs(CANDIDATE, failing))
    assert "Failing cases (3 of 3)" in exp
    assert "…and 2 more." in exp


def test_experiment_all_passed_empty_and_errors() -> None:
    experiment = Experiment(name="x", suite="toy", created_at=CREATED)
    ok = MarkdownRenderer().experiment(ExperimentReport.from_runs(experiment, [run("a")]))
    assert "### ✅ AgentLens: 1/1 runs passed (100%)" in ok
    assert "<details" not in ok
    assert "model" not in ok
    assert "· 1 case\n" in ok
    assert "experiment `x`" in ok
    assert " git `" not in ok

    empty = MarkdownRenderer().experiment(ExperimentReport.from_runs(experiment, []))
    assert "### ❌ AgentLens: 0/0 runs passed\n" not in empty  # 0 of 0 is not a failure
    assert "### ✅ AgentLens: 0/0 runs passed\n" in empty
    assert "| Metric |" not in empty

    errors = ExperimentReport.from_runs(experiment, [run("a", error="E"), run("b", error="E")])
    assert "· 2 errors" in MarkdownRenderer().experiment(errors)


def test_skipped_gates_are_shown() -> None:
    runs = [run("a")]
    gates = evaluate_gates(Gates.from_dict({"pass_rate": {"min_delta": -0.1}}), run=runs)
    text = MarkdownRenderer().experiment(ExperimentReport.from_runs(CANDIDATE, runs, gates=gates))
    assert "### ✅ AgentLens: gates passed" in text
    assert "| skip | pass_rate | min_delta -0.1 | - |" in text
    assert "> - no baseline comparison" in text
