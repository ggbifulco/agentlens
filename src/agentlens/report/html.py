"""The HTML report (FR-C2): one self-contained file with a trace viewer.

:class:`HtmlRenderer` is registered as the ``html`` format, so
``render(report, "html")`` works on any report. A report model carries
aggregates only, so to browse the traces behind it pass the runs::

    from agentlens.report import ComparisonReport, render_html

    html = render_html(
        ComparisonReport(comparison=comparison, gates=gates),
        runs=candidate_runs,              # list[RunResult] (traces + eval results)
        baseline_runs=baseline_runs,
    )
    Path("report.html").write_text(html, encoding="utf-8")

The page has a verdict header, the metrics table with confidence intervals,
the gates, and one row per case; a case opens to show its runs side by side
(baseline vs candidate: pass/fail, every evaluator's score and reason, cost,
latency, steps) with a button that opens the run's **span tree**: collapsible
nested spans with kind badges, a timing waterfall, pretty-printed
input/output/attributes, usage and cost, errors highlighted.

**Self-contained**: CSS and JavaScript are inlined and nothing is loaded from
the network, so the file can be attached to a CI run or mailed. The page
follows the system's light/dark preference.

**Escaping**: the page is rendered by Jinja2 with autoescaping on, so text
from users (names, case ids, reasons) is HTML-escaped; the traces travel as
JSON in a ``<script type="application/json">`` block with ``<``, ``>`` and
``&`` written as ``\\u003c``/``\\u003e``/``\\u0026`` (so a ``</script>`` in a
trace cannot end the block), and the script builds the viewer with
``textContent``, never ``innerHTML``.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime

from jinja2 import Environment, PackageLoader, StrictUndefined
from markupsafe import Markup
from pydantic import JsonValue

from agentlens import __version__
from agentlens.compare.comparator import metric_direction, run_metrics
from agentlens.compare.gates import GateReport, GateResult
from agentlens.compare.model import CaseComparison, Comparison, Direction, MetricDelta, Status
from agentlens.report.model import ComparisonReport, ExperimentReport
from agentlens.report.plain import MISSING, fmt_value
from agentlens.report.terminal import _bound, _metric_label
from agentlens.schema import Experiment, Span
from agentlens.suite.runner import RunResult

__all__ = ["HtmlRenderer", "render_html"]

_PREVIEW_CHARS = 160

_env = Environment(
    loader=PackageLoader("agentlens.report", "templates"),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
    keep_trailing_newline=False,
)

_STATUS_ORDER = {
    Status.REGRESSION.value: 0,
    Status.IMPROVEMENT.value: 1,
    Status.UNCHANGED.value: 2,
    Status.NO_DATA.value: 3,
    "added": 4,
    "removed": 5,
}
_STATUS_LABEL = {
    Status.REGRESSION.value: "regression",
    Status.IMPROVEMENT.value: "improvement",
    Status.UNCHANGED.value: "unchanged",
    Status.NO_DATA.value: "no data",
    "added": "added",
    "removed": "removed",
}


def _asset(name: str) -> Markup:
    """A packaged CSS/JS file, inlined verbatim (not a template: no Jinja syntax inside)."""
    loader = _env.loader
    assert loader is not None
    source, _, _ = loader.get_source(_env, name)
    return Markup(source)  # our own packaged file


# -- JSON payload --------------------------------------------------------------------


def script_json(data: object) -> Markup:
    """``data`` as JSON safe to put inside a ``<script>`` element.

    ``<``, ``>`` and ``&`` become ``\\u003c``/``\\u003e``/``\\u0026`` (the same
    JSON value once parsed), so no string in the data can close the element
    (``</script>``) or open a comment; U+2028/U+2029 are escaped too.
    """
    try:
        text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), allow_nan=False)
    except ValueError:  # NaN/inf in user data: not JSON, so they become null
        text = json.dumps(_finite(data), ensure_ascii=False, separators=(",", ":"))
    for char, escape in (
        ("<", "\\u003c"),
        (">", "\\u003e"),
        ("&", "\\u0026"),
        ("\u2028", "\\u2028"),
        ("\u2029", "\\u2029"),
    ):
        text = text.replace(char, escape)
    return Markup(text)  # escaped above: safe inside <script>


def _finite(value: object) -> object:
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _finite(item) for key, item in value.items()}
    if isinstance(value, list | tuple):
        return [_finite(item) for item in value]
    return value


def _ms(start: datetime, end: datetime) -> float:
    return (end - start).total_seconds() * 1000


def _span_node(
    span: Span, children: dict[str | None, list[Span]], origin: datetime
) -> dict[str, JsonValue]:
    usage: JsonValue = None
    if span.usage is not None:
        usage = {
            "input_tokens": span.usage.input_tokens,
            "output_tokens": span.usage.output_tokens,
            "cost_usd": span.usage.cost_usd,
        }
    return {
        "id": span.id,
        "kind": span.kind.value,
        "name": span.name,
        "start": span.start_time.isoformat(),
        "offset_ms": _ms(origin, span.start_time),
        "duration_ms": span.latency_ms,
        "input": span.input,
        "output": span.output,
        "error": span.error,
        "usage": usage,
        "attributes": span.attributes,
        "children": [_span_node(child, children, origin) for child in children.get(span.id, [])],
    }


def _trace_payload(run: RunResult, side: str | None) -> dict[str, JsonValue]:
    trace = run.trace
    origin = min([trace.start_time, *(s.start_time for s in trace.spans)])
    ends = [trace.end_time, *(s.end_time for s in trace.spans)]
    known_ends = [end for end in ends if end is not None]
    total = max([_ms(origin, end) for end in known_ends], default=0.0)
    children: dict[str | None, list[Span]] = {}
    for span in sorted(trace.spans, key=lambda span: span.start_time):
        children.setdefault(span.parent_id, []).append(span)
    return {
        "id": trace.id,
        "name": trace.name,
        "case_id": run.case_id,
        "repetition": run.repetition,
        "side": side,
        "passed": run.passed,
        "start": trace.start_time.isoformat(),
        "total_ms": total,
        "latency_ms": trace.latency_ms,
        "cost_usd": trace.total_cost,
        "tokens": trace.total_tokens,
        "steps": trace.n_steps,
        "input": trace.input,
        "output": trace.output,
        "error": trace.error,
        "metadata": trace.metadata,
        "evals": [
            {
                "evaluator": result.evaluator,
                "score": result.score,
                "passed": result.passed,
                "reason": result.reason,
            }
            for result in run.eval_results
        ],
        "spans": [_span_node(root, children, origin) for root in children.get(None, [])],
    }


# -- view models ---------------------------------------------------------------------


@dataclass(frozen=True)
class _EvalView:
    evaluator: str
    score: str
    passed: bool
    reason: str


@dataclass(frozen=True)
class _RunView:
    trace_id: str
    repetition: int
    passed: bool
    error: str | None
    latency: str
    cost: str
    steps: int
    tokens: int
    output: str
    evals: list[_EvalView]


@dataclass(frozen=True)
class _Pair:
    """A metric of one case: baseline and candidate values (formatted) and the verdict."""

    label: str
    baseline: str
    candidate: str
    trend: str
    """``worse``, ``better`` or ``""`` (no change or unknown)."""


@dataclass
class _CaseView:
    case_id: str
    status: str
    status_label: str
    pairs: list[_Pair] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    baseline_runs: list[_RunView] = field(default_factory=list)
    candidate_runs: list[_RunView] = field(default_factory=list)


@dataclass(frozen=True)
class _MetricRow:
    label: str
    better: str
    n: int
    baseline: str
    candidate: str
    delta: str
    delta_pct: str
    ci: str
    status: str
    status_label: str
    ci_bar: dict[str, float] | None
    """Interval and point estimate as percentages of the row's scale (``None``: no CI)."""


@dataclass(frozen=True)
class _GateRow:
    verdict: str
    name: str
    rule: str
    actual: str
    reason: str


@dataclass(frozen=True)
class _GatesView:
    passed: bool
    counts: str
    rows: list[_GateRow]
    notes: list[str]


@dataclass(frozen=True)
class _Card:
    label: str
    value: str
    detail: str = ""
    trend: str = ""


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return MISSING
    if value >= 1000:
        return f"{value / 1000:.3g} s"
    return f"{value:.3g} ms"


def _fmt_cost(value: float | None) -> str:
    return MISSING if value is None else f"${value:.4g}"


def _fmt_metric(name: str, value: float | None) -> str:
    if value is None:
        return MISSING
    if name == "pass_rate":
        return f"{value:.0%}"
    if name == "cost_usd":
        return _fmt_cost(value)
    if name == "latency_ms":
        return _fmt_ms(value)
    return fmt_value(value)


def _preview(value: JsonValue) -> str:
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 1] + "…"


def _run_view(run: RunResult) -> _RunView:
    trace = run.trace
    return _RunView(
        trace_id=trace.id,
        repetition=run.repetition,
        passed=run.passed,
        error=trace.error,
        latency=_fmt_ms(trace.latency_ms),
        cost=_fmt_cost(trace.total_cost),
        steps=trace.n_steps,
        tokens=trace.total_tokens,
        output="" if trace.output is None else _preview(trace.output),
        evals=[
            _EvalView(r.evaluator, f"{r.score:.2f}", r.passed, r.reason or "")
            for r in run.eval_results
        ],
    )


def _trend(direction: Direction, delta: float | None) -> str:
    if delta is None or delta == 0 or math.isnan(delta):
        return ""
    worse = delta < 0 if direction is Direction.HIGHER_IS_BETTER else delta > 0
    return "worse" if worse else "better"


def _ci_bar(metric: MetricDelta) -> dict[str, float] | None:
    low, high, delta = metric.ci_low, metric.ci_high, metric.delta
    if low is None or high is None or delta is None:
        return None
    scale = max(abs(low), abs(high), abs(delta))
    if scale == 0:
        return {"left": 50.0, "width": 0.0, "point": 50.0}

    def pos(value: float) -> float:
        return round(50 + 50 * value / scale, 2)

    return {"left": pos(low), "width": round(pos(high) - pos(low), 2), "point": pos(delta)}


def _metric_row(metric: MetricDelta) -> _MetricRow:
    name = metric.name
    if metric.ci_low is None or metric.ci_high is None:
        ci = MISSING
    else:
        ci = f"[{fmt_value(metric.ci_low, signed=True)}, {fmt_value(metric.ci_high, signed=True)}]"
    return _MetricRow(
        label=_metric_label(name),
        better="↑" if metric.direction is Direction.HIGHER_IS_BETTER else "↓",
        n=metric.n,
        baseline=_fmt_metric(name, metric.baseline),
        candidate=_fmt_metric(name, metric.candidate),
        delta=fmt_value(metric.delta, signed=True),
        delta_pct=MISSING if metric.delta_pct is None else f"{metric.delta_pct:+.1f}%",
        ci=ci,
        status=metric.status.value,
        status_label=_STATUS_LABEL[metric.status.value],
        ci_bar=_ci_bar(metric),
    )


def _gate_row(result: GateResult) -> _GateRow:
    verdict = "skip" if result.skipped else "pass" if result.passed else "fail"
    return _GateRow(
        verdict=verdict,
        name=result.name,
        rule=f"{result.rule} {_bound(result.rule, result.threshold)}",
        actual=MISSING if result.actual is None else _bound(result.rule, result.actual),
        reason=result.reason,
    )


def _gates_view(gates: GateReport | None) -> _GatesView | None:
    if gates is None:
        return None
    n_skipped, n_failed = len(gates.skipped), len(gates.failures)
    n_passed = len(gates.results) - n_skipped - n_failed
    return _GatesView(
        passed=gates.passed,
        counts=f"{n_passed} passed, {n_failed} failed, {n_skipped} skipped",
        rows=[_gate_row(result) for result in gates.results],
        notes=list(gates.notes),
    )


def _plural(n: int, word: str) -> str:
    return f"{n} {word}{'' if n == 1 else 's'}"


def _experiment_meta(experiment: Experiment | None) -> dict[str, str] | None:
    if experiment is None:
        return None
    sha = experiment.config.get("git_sha")
    git = ""
    if isinstance(sha, str):
        git = sha[:12] + (" (dirty)" if experiment.config.get("git_dirty") else "")
    model = experiment.config.get("model")
    return {
        "name": experiment.name,
        "id": experiment.id,
        "suite": experiment.suite,
        "created": experiment.created_at.strftime("%Y-%m-%d %H:%M UTC"),
        "git": git,
        "model": model if isinstance(model, str) else "",
        "baseline": "baseline" if experiment.is_baseline else "",
    }


def _group(runs: Iterable[RunResult]) -> dict[str, list[RunResult]]:
    grouped: dict[str, list[RunResult]] = {}
    for run in runs:
        grouped.setdefault(run.case_id, []).append(run)
    for case_runs in grouped.values():
        case_runs.sort(key=lambda run: run.repetition)
    return grouped


def _case_pairs(case: CaseComparison) -> list[_Pair]:
    pairs = []
    for name, metric in case.metrics.items():
        pairs.append(
            _Pair(
                label=_metric_label(name),
                baseline=_fmt_metric(name, metric.baseline),
                candidate=_fmt_metric(name, metric.candidate),
                # Per-case latency moves with machine noise: shown, not coloured.
                trend="" if name == "latency_ms" else _trend(metric_direction(name), metric.delta),
            )
        )
    return pairs


def _comparison_cases(
    cmp: Comparison, baseline: dict[str, list[RunResult]], candidate: dict[str, list[RunResult]]
) -> list[_CaseView]:
    views: list[_CaseView] = []
    for case in cmp.cases:
        view = _CaseView(
            case_id=case.case_id,
            status=case.status.value,
            status_label=_STATUS_LABEL[case.status.value],
            pairs=_case_pairs(case),
        )
        if case.regressed_evaluators:
            view.notes.append("regressed evaluators: " + ", ".join(case.regressed_evaluators))
        views.append(view)
    for status, ids in (("added", cmp.added_cases), ("removed", cmp.removed_cases)):
        for case_id in ids:
            note = "only in the candidate" if status == "added" else "only in the baseline"
            views.append(
                _CaseView(case_id, status, _STATUS_LABEL[status], notes=[note]),
            )
    for view in views:
        view.baseline_runs = [_run_view(run) for run in baseline.get(view.case_id, [])]
        view.candidate_runs = [_run_view(run) for run in candidate.get(view.case_id, [])]
    return sorted(views, key=lambda view: _STATUS_ORDER[view.status])


def _comparison_verdict(report: ComparisonReport) -> tuple[bool, str]:
    cmp, gates = report.comparison, report.gates
    findings: list[str] = []
    if cmp.regressions:
        findings.append(f"{_plural(len(cmp.regressions), 'metric')} regressed")
    if cmp.improvements:
        findings.append(f"{len(cmp.improvements)} improved")
    if cmp.case_regressions:
        findings.append(f"{_plural(len(cmp.case_regressions), 'case')} regressed")
    if not findings:
        findings.append("no metric changed" if cmp.metrics else "nothing to compare")
    if gates is not None:
        findings.insert(0, "gates passed" if gates.passed else "gates failed")
        return gates.passed, " · ".join(findings)
    return not cmp.regressions, " · ".join(findings)


def _comparison_cards(cmp: Comparison) -> list[_Card]:
    cards = []
    for name in ("pass_rate", "cost_usd", "latency_ms", "steps"):
        try:
            metric = cmp.metric(name)
        except KeyError:
            continue
        cards.append(
            _Card(
                label=_metric_label(name),
                value=_fmt_metric(name, metric.candidate),
                detail=f"baseline {_fmt_metric(name, metric.baseline)}",
                trend="worse"
                if metric.is_regression
                else "better"
                if metric.is_improvement
                else "",
            )
        )
    return cards


def _experiment_cases(
    report: ExperimentReport, runs: dict[str, list[RunResult]]
) -> list[_CaseView]:
    views = []
    for case in report.cases:
        case_runs = runs.get(case.case_id, [])
        failed = case.passed < case.runs
        view = _CaseView(
            case_id=case.case_id,
            status="fail" if failed else "pass",
            status_label=f"{case.passed}/{case.runs} passed",
        )
        view.pairs.append(_Pair("pass rate", "", f"{case.pass_rate:.0%}", ""))
        for name, score in case.scores.items():
            view.pairs.append(_Pair(f"score.{name}", "", fmt_value(score), ""))
        for name, (value, _) in run_metrics(case_runs).items():
            if name in ("cost_usd", "latency_ms", "steps"):
                view.pairs.append(_Pair(_metric_label(name), "", _fmt_metric(name, value), ""))
        if case.failed_evaluators:
            view.notes.append("failed evaluators: " + ", ".join(case.failed_evaluators))
        if case.errors:
            view.notes.append(_plural(case.errors, "run") + " raised or timed out")
        view.candidate_runs = [_run_view(run) for run in case_runs]
        views.append(view)
    return sorted(views, key=lambda view: view.status != "fail")


# -- renderer ------------------------------------------------------------------------


class HtmlRenderer:
    """Self-contained HTML reports with a span-tree trace viewer.

    Args:
        runs: The runs of the report's experiment (for a comparison: of the
            candidate), e.g. :func:`~agentlens.compare.experiment_runs` or a
            :func:`~agentlens.suite.run_suite` result's ``runs``. Their traces
            and evaluator reasons are embedded; without runs the page shows the
            aggregates only.
        baseline_runs: The baseline's runs (comparisons only).
        title: Page title (default: built from the experiment names).
    """

    def __init__(
        self,
        *,
        runs: Iterable[RunResult] | None = None,
        baseline_runs: Iterable[RunResult] | None = None,
        title: str | None = None,
    ) -> None:
        self.runs: list[RunResult] = list(runs or [])
        self.baseline_runs: list[RunResult] = list(baseline_runs or [])
        self.title = title

    def _chrome(self) -> dict[str, object]:
        """Page chrome around the report: navigation links (none) and the footer line.

        Overridden by the local web UI (:mod:`agentlens.ui`) to add its navigation.
        """
        return {
            "nav": None,
            "footer": f"Generated by AgentLens {__version__} · self-contained: "
            "no network access needed.",
        }

    def _page(self, context: dict[str, object], traces: Sequence[dict[str, JsonValue]]) -> str:
        template = _env.get_template("report.html.j2")
        payload = {"traces": {trace["id"]: trace for trace in traces}}
        return template.render(
            **self._chrome(),
            **context,
            data=script_json(payload),
            css=_asset("report.css"),
            js=_asset("report.js"),
            has_traces=bool(traces),
            version=__version__,
        )

    def comparison(self, report: ComparisonReport) -> str:
        """A candidate-vs-baseline comparison, cases side by side."""
        cmp = report.comparison
        ok, verdict = _comparison_verdict(report)
        base, cand = _experiment_meta(cmp.baseline), _experiment_meta(cmp.candidate)
        title = self.title or (
            f"AgentLens: {cand['name'] if cand else 'candidate'} vs "
            f"{base['name'] if base else 'baseline'}"
        )
        cases = _comparison_cases(cmp, _group(self.baseline_runs), _group(self.runs))
        counts = {status: 0 for status in _STATUS_ORDER}
        for case in cases:
            counts[case.status] += 1
        traces = [_trace_payload(run, "baseline") for run in self.baseline_runs]
        traces += [_trace_payload(run, "candidate") for run in self.runs]
        context: dict[str, object] = {
            "kind": "comparison",
            "title": title,
            "ok": ok,
            "verdict": verdict,
            "baseline": base,
            "candidate": cand,
            "context_line": (
                f"{_plural(len(cmp.cases), 'case')} paired · {len(cmp.added_cases)} added · "
                f"{len(cmp.removed_cases)} removed · paired bootstrap, {cmp.resamples} "
                f"resamples, {cmp.confidence:.0%} CI"
                + (f", seed {cmp.seed}" if cmp.seed is not None else "")
            ),
            "confidence": f"{cmp.confidence:.0%}",
            "cards": _comparison_cards(cmp),
            "metrics": [_metric_row(metric) for metric in cmp.metrics],
            "gates": _gates_view(report.gates),
            "cases": cases,
            "counts": {k: v for k, v in counts.items() if v},
            "status_labels": _STATUS_LABEL,
        }
        return self._page(context, traces)

    def experiment(self, report: ExperimentReport) -> str:
        """One experiment's results, every case with its runs."""
        exp = _experiment_meta(report.experiment)
        gates = report.gates
        ok = gates.passed if gates is not None else report.passed == report.runs
        verdict = f"{report.passed}/{report.runs} runs passed"
        if report.errors:
            verdict += f" · {_plural(report.errors, 'error')}"
        if gates is not None:
            verdict = ("gates passed" if gates.passed else "gates failed") + " · " + verdict
        cases = _experiment_cases(report, _group(self.runs))
        failing = sum(case.status == "fail" for case in cases)
        cards = [
            _Card("runs passed", f"{report.passed}/{report.runs}", f"{report.pass_rate:.0%}"),
            *(
                _Card(_metric_label(name), _fmt_metric(name, value))
                for name, value in report.metrics.items()
                if not name.startswith("score.")
            ),
        ]
        context: dict[str, object] = {
            "kind": "experiment",
            "title": self.title or f"AgentLens: {report.experiment.name}",
            "ok": ok,
            "verdict": verdict,
            "experiment": exp,
            "context_line": (
                f"{_plural(len(report.cases), 'case')} · {_plural(report.runs, 'run')} · "
                f"{failing} failing"
            ),
            "cards": cards,
            "experiment_metrics": [
                {"label": _metric_label(name), "value": _fmt_metric(name, value)}
                for name, value in report.metrics.items()
            ],
            "gates": _gates_view(gates),
            "cases": cases,
            "counts": {"fail": failing, "pass": len(cases) - failing},
            "status_labels": {"fail": "failing", "pass": "passing"},
        }
        return self._page(context, [_trace_payload(run, None) for run in self.runs])


def render_html(
    report: ExperimentReport | ComparisonReport,
    *,
    runs: Iterable[RunResult] | None = None,
    baseline_runs: Iterable[RunResult] | None = None,
    title: str | None = None,
) -> str:
    """Render ``report`` as a self-contained HTML page (see :class:`HtmlRenderer`)."""
    renderer = HtmlRenderer(runs=runs, baseline_runs=baseline_runs, title=title)
    if isinstance(report, ComparisonReport):
        return renderer.comparison(report)
    return renderer.experiment(report)
