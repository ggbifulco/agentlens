"""The HTML report: content, structure, escaping, self-containment and the trace payload."""

from __future__ import annotations

import json
import math
import re
from datetime import timedelta
from html.parser import HTMLParser

import pytest

from agentlens.compare import Comparison, MetricDelta, compare_runs
from agentlens.compare.model import Direction, Status
from agentlens.report import (
    ComparisonReport,
    ExperimentReport,
    HtmlRenderer,
    ReportFormat,
    get_renderer,
    render,
    render_html,
)
from agentlens.report import html as html_module
from agentlens.schema import EvalResult, Experiment, Span, SpanKind, Trace, Usage
from agentlens.suite import RunResult
from tests.compare.helpers import T0, run
from tests.report.test_terminal import (
    BASELINE,
    CREATED,
    _baseline_runs,
    _candidate_runs,
    _comparison_report,
    _experiment_report,
)

EVIL = "</script><script>alert(1)</script>"
VOID = {"meta", "br", "hr", "img", "input", "link", "wbr", "source", "col", "area", "base"}


class _Structure(HTMLParser):
    """Checks that every element is closed in order, and collects tags and attributes."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[str] = []
        self.errors: list[str] = []
        self.tags: list[tuple[str, dict[str, str | None]]] = []
        self.doctype = False
        self.data: dict[str, str] = {}
        self._script_id: str | None = None

    def handle_decl(self, decl: str) -> None:
        self.doctype = decl.lower() == "doctype html"

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        attributes = dict(attrs)
        self.tags.append((tag, attributes))
        if tag not in VOID:
            self.stack.append(tag)
        if tag == "script":
            self._script_id = attributes.get("id") or "inline"

    def handle_endtag(self, tag: str) -> None:
        if not self.stack or self.stack[-1] != tag:
            self.errors.append(f"unexpected </{tag}> (open: {self.stack[-3:]})")
            return
        self.stack.pop()
        if tag == "script":
            self._script_id = None

    def handle_data(self, data: str) -> None:
        if self._script_id is not None:
            self.data[self._script_id] = self.data.get(self._script_id, "") + data


def _parse(html: str) -> _Structure:
    parser = _Structure()
    parser.feed(html)
    parser.close()
    assert parser.doctype
    assert parser.errors == []
    assert parser.stack == []
    return parser


def _payload(html: str) -> dict[str, dict[str, dict[str, object]]]:
    data: dict[str, dict[str, dict[str, object]]] = json.loads(_parse(html).data["agentlens-data"])
    return data


def _assert_self_contained(html: str) -> None:
    parser = _parse(html)
    for tag, attrs in parser.tags:
        assert tag not in {"link", "img", "iframe", "object", "embed"}, tag
        assert "src" not in attrs, (tag, attrs)
        href = attrs.get("href")
        assert href is None or href.startswith("#"), href
    assert not re.search(r"https?:|//[a-z]", html.replace("http-equiv", ""))
    assert "@import" not in html
    assert "url(" not in html


def _comparison_html() -> str:
    return render_html(_comparison_report(), runs=_candidate_runs(), baseline_runs=_baseline_runs())


# -- registry ------------------------------------------------------------------------


def test_html_is_a_registered_format() -> None:
    assert isinstance(get_renderer(ReportFormat.HTML), HtmlRenderer)
    html = render(_comparison_report(), "html")
    assert html.startswith("<!doctype html>")
    assert "Traces are not included" in html
    assert _payload(html) == {"traces": {}}
    assert "Runs not included." in html


# -- comparison ----------------------------------------------------------------------


def test_comparison_key_content() -> None:
    html = _comparison_html()
    assert "<title>AgentLens: pr-42 vs main</title>" in html
    assert "gates failed · 3 metrics regressed · 1 improved · 3 cases regressed" in html
    assert "8 cases paired · 1 added · 1 removed" in html
    assert "git <code>0123456789ab (dirty)</code>" in html
    assert "model <code>openai:gpt-4o-mini</code>" in html
    # metrics table with CI and status
    assert "<th>95% CI</th>" in html
    assert 'class="st-regression"' in html
    assert '>pass rate <span class="better"' in html
    # gates
    assert 'Gates <span class="pill st-fail">failed</span>' in html
    assert "<code>max_delta_pct +20%</code>" in html
    # cases: regressions first, added/removed included, side by side
    order = re.findall(r'data-status="(\w+)" data-case="([^"]+)"', html)
    assert order[0] == ("regression", "case-00")
    assert ("added", "case-new") in order
    assert ("removed", "retired") in order
    assert [status for status, _ in order].index("added") > 2
    assert "<h3>Baseline</h3>" in html
    assert "<h3>Candidate</h3>" in html
    assert "regressed evaluators: exact" in html
    assert "only in the candidate" in html
    assert "TimeoutError: agent call timed out" in html
    assert "View trace" in html


def test_comparison_trace_links_match_the_payload() -> None:
    html = _comparison_html()
    traces = _payload(html)["traces"]
    linked = set(re.findall(r'href="#trace-([^"]+)" data-trace="\1"', html))
    assert linked == set(traces)
    assert len(traces) == len(_baseline_runs()) + len(_candidate_runs())
    sides = {str(trace["side"]) for trace in traces.values()}
    assert sides == {"baseline", "candidate"}


def test_comparison_is_self_contained_and_well_formed() -> None:
    html = _comparison_html()
    _assert_self_contained(html)
    parser = _parse(html)
    scripts = [attrs for tag, attrs in parser.tags if tag == "script"]
    assert scripts == [{"type": "application/json", "id": "agentlens-data"}, {}]
    assert "<style>" in html
    assert "prefers-color-scheme: dark" in html
    assert 'id="trace-viewer"' in html


def test_ok_verdict_without_gates() -> None:
    runs = [run("a"), run("b")]
    report = ComparisonReport(comparison=compare_runs(runs, runs, resamples=10))
    html = render_html(report)
    assert 'class="verdict ok"' in html
    assert "no metric changed" in html
    assert '<dd class="muted">in memory</dd>' in html
    assert "<title>AgentLens: candidate vs baseline</title>" in html


def test_empty_comparison() -> None:
    report = ComparisonReport(comparison=Comparison(resamples=10, confidence=0.9, seed=None))
    html = render_html(report, title="Empty")
    _parse(html)
    assert "<title>Empty</title>" in html
    assert "nothing to compare" in html
    assert "No metric to compare." in html
    assert "No cases." in html
    assert "seed" not in html.split("context muted")[1].split("</p>")[0]
    assert "90% CI" in html


def test_summary_cards_only_for_present_metrics() -> None:
    metric = MetricDelta(
        name="pass_rate",
        direction=Direction.HIGHER_IS_BETTER,
        n=3,
        baseline=0.5,
        candidate=1.0,
        delta=0.5,
        ci_low=0.1,
        ci_high=0.9,
        status=Status.IMPROVEMENT,
    )
    report = ComparisonReport(
        comparison=Comparison(metrics=[metric], resamples=10, confidence=0.95)
    )
    html = render_html(report)
    assert html.count('class="card ') == 1
    assert 'class="card better"' in html
    assert "1 improved" in html


# -- experiment ----------------------------------------------------------------------


def test_experiment_report() -> None:
    html = render_html(_experiment_report(), runs=_candidate_runs())
    _assert_self_contained(html)
    assert "<title>AgentLens: pr-42</title>" in html
    assert "gates failed · 5/9 runs passed · 1 error" in html
    order = re.findall(r'data-status="(\w+)" data-case="([^"]+)"', html)
    failing = [case for status, case in order if status == "fail"]
    assert failing == ["case-00", "case-01", "case-02", "case-new"]
    assert order[: len(failing)] == [("fail", case) for case in failing]
    assert "failed evaluators: exact" in html
    assert "1 run raised or timed out" in html
    assert 'failing <span class="count">4</span>' in html
    assert "<h3>Baseline</h3>" not in html
    traces = _payload(html)["traces"]
    assert {trace["side"] for trace in traces.values()} == {None}


def test_experiment_without_runs() -> None:
    experiment = Experiment(name="empty", suite="s", created_at=CREATED)
    html = render_html(ExperimentReport.from_runs(experiment, []))
    _parse(html)
    assert "No runs." in html
    assert "0/0 runs passed" in html
    assert 'class="verdict ok"' in html


# -- escaping ------------------------------------------------------------------------


def _evil_run(case_id: str) -> RunResult:
    trace_id = "t-evil"
    span = Span(
        trace_id=trace_id,
        kind=SpanKind.TOOL_CALL,
        name=EVIL,
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=5),
        input={"q": EVIL},
        output=EVIL,
        error=EVIL,
        attributes={EVIL: "<!--"},
    )
    trace = Trace(
        id=trace_id,
        name=EVIL,
        input=EVIL,
        output=EVIL,
        error=f"ValueError: {EVIL}",
        start_time=T0,
        end_time=T0 + timedelta(milliseconds=10),
        spans=[span],
        metadata={"case_id": case_id},
    )
    results = [EvalResult(evaluator=EVIL, score=0.0, passed=False, reason=EVIL)]
    return RunResult(case_id=case_id, repetition=0, trace=trace, eval_results=results)


def test_user_data_is_escaped() -> None:
    evil_exp = Experiment(name=EVIL, suite=EVIL, created_at=CREATED, config={"model": EVIL})
    runs = [_evil_run(EVIL)]
    comparison = compare_runs(
        [run(EVIL, passed=True)],
        runs,
        resamples=10,
        baseline_experiment=BASELINE,
        candidate_experiment=evil_exp,
    )
    html = render_html(ComparisonReport(comparison=comparison), runs=runs)
    assert EVIL not in html
    assert "alert(1)</script>" not in html
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in html
    assert html.count("<script") == 2
    assert html.count("</script>") == 2
    assert "<!--" not in html
    _assert_self_contained(html)
    # The JSON payload still carries the original text.
    trace = _payload(html)["traces"]["t-evil"]
    assert trace["name"] == EVIL
    spans = trace["spans"]
    assert isinstance(spans, list)
    assert spans[0]["name"] == EVIL
    assert spans[0]["attributes"] == {EVIL: "<!--"}


def test_script_json() -> None:
    text = str(html_module.script_json({"a": "</script>&<!--\u2028\u2029", "b": [1.5]}))
    assert text == ('{"a":"\\u003c/script\\u003e\\u0026\\u003c!--\\u2028\\u2029","b":[1.5]}')
    assert json.loads(text) == {"a": "</script>&<!--\u2028\u2029", "b": [1.5]}
    with_nan = str(html_module.script_json({"x": [math.nan, (math.inf, 1.0)], "y": "ok"}))
    assert json.loads(with_nan) == {"x": [None, [None, 1.0]], "y": "ok"}


# -- trace payload -------------------------------------------------------------------


def test_span_tree_payload() -> None:
    trace_id = "t-tree"
    step = Span(
        id="step",
        trace_id=trace_id,
        kind=SpanKind.AGENT_STEP,
        name="step",
        start_time=T0 + timedelta(milliseconds=1),
        end_time=T0 + timedelta(milliseconds=30),
    )
    tool = Span(
        id="tool",
        trace_id=trace_id,
        parent_id="step",
        kind=SpanKind.TOOL_CALL,
        name="search",
        start_time=T0 + timedelta(milliseconds=12),
        end_time=T0 + timedelta(milliseconds=20),
        error="KeyError: 'x'",
    )
    llm = Span(
        id="llm",
        trace_id=trace_id,
        parent_id="step",
        kind=SpanKind.LLM_CALL,
        name="gpt",
        start_time=T0 + timedelta(milliseconds=2),
        end_time=T0 + timedelta(milliseconds=10),
        usage=Usage(input_tokens=10, output_tokens=5, cost_usd=0.002),
    )
    hanging = Span(
        id="hanging",
        trace_id=trace_id,
        kind=SpanKind.CUSTOM,
        name="hanging",
        start_time=T0 + timedelta(milliseconds=40),
    )
    trace = Trace(
        id=trace_id,
        name="agent",
        start_time=T0,
        end_time=None,  # unfinished: the total comes from the spans
        spans=[hanging, tool, step, llm],
        metadata={"case_id": "c"},
    )
    result = RunResult(case_id="c", repetition=2, trace=trace)
    payload = html_module._trace_payload(result, "candidate")
    assert payload["total_ms"] == pytest.approx(30.0)
    assert payload["latency_ms"] is None
    assert payload["cost_usd"] == pytest.approx(0.002)
    assert payload["tokens"] == 15
    assert payload["repetition"] == 2
    roots = payload["spans"]
    assert isinstance(roots, list)
    assert [node["id"] for node in roots] == ["step", "hanging"]  # type: ignore[index]
    step_node = roots[0]
    assert isinstance(step_node, dict)
    children = step_node["children"]
    assert isinstance(children, list)
    assert [c["id"] for c in children] == ["llm", "tool"]  # type: ignore[index]
    assert step_node["offset_ms"] == pytest.approx(1.0)
    assert step_node["duration_ms"] == pytest.approx(29.0)
    assert step_node["usage"] is None
    assert children[0]["usage"] == {  # type: ignore[index]
        "input_tokens": 10,
        "output_tokens": 5,
        "cost_usd": 0.002,
    }
    assert children[1]["error"] == "KeyError: 'x'"  # type: ignore[index]
    assert roots[1]["duration_ms"] is None  # type: ignore[index]
    assert roots[1]["children"] == []  # type: ignore[index]

    empty = Trace(id="t-empty", name="x", start_time=T0)
    assert html_module._trace_payload(RunResult("c", 0, empty), None)["total_ms"] == 0.0


def test_run_view_previews_output() -> None:
    long = run("c")
    long.trace.output = "word " * 100
    view = html_module._run_view(long)
    assert len(view.output) == html_module._PREVIEW_CHARS
    assert view.output.endswith("…")
    structured = run("c")
    structured.trace.output = {"answer": 42}
    assert html_module._run_view(structured).output == '{"answer": 42}'
    assert html_module._run_view(run("c")).output == ""


# -- formatting helpers --------------------------------------------------------------


@pytest.mark.parametrize(
    ("direction", "delta", "trend"),
    [
        (Direction.HIGHER_IS_BETTER, None, ""),
        (Direction.HIGHER_IS_BETTER, 0.0, ""),
        (Direction.HIGHER_IS_BETTER, math.nan, ""),
        (Direction.HIGHER_IS_BETTER, -0.1, "worse"),
        (Direction.HIGHER_IS_BETTER, 0.1, "better"),
        (Direction.LOWER_IS_BETTER, 0.1, "worse"),
        (Direction.LOWER_IS_BETTER, -0.1, "better"),
    ],
)
def test_trend(direction: Direction, delta: float | None, trend: str) -> None:
    assert html_module._trend(direction, delta) == trend


def test_ci_bar() -> None:
    def metric(low: float | None, high: float | None, delta: float | None) -> MetricDelta:
        return MetricDelta(
            name="steps",
            direction=Direction.LOWER_IS_BETTER,
            n=1,
            delta=delta,
            ci_low=low,
            ci_high=high,
            status=Status.UNCHANGED,
        )

    assert html_module._ci_bar(metric(None, None, None)) is None
    assert html_module._ci_bar(metric(0.0, 0.0, 0.0)) == {"left": 50.0, "width": 0.0, "point": 50.0}
    assert html_module._ci_bar(metric(-1.0, 2.0, 0.5)) == {
        "left": 25.0,
        "width": 75.0,
        "point": 62.5,
    }


def test_formatters() -> None:
    assert html_module._fmt_ms(None) == "-"
    assert html_module._fmt_ms(12.345) == "12.3 ms"
    assert html_module._fmt_ms(1500.0) == "1.5 s"
    assert html_module._fmt_cost(None) == "-"
    assert html_module._fmt_cost(0.01234) == "$0.01234"
    assert html_module._fmt_metric("pass_rate", 0.5) == "50%"
    assert html_module._fmt_metric("steps", 2.0) == "2"
    assert html_module._fmt_metric("steps", None) == "-"
