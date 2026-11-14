"""The local web UI's application: routing, query parsing, pages and the JSON API.

:class:`UIApp` maps a request (method + target) to a :class:`Response`. It has
no socket code, so it is tested directly and served by
:func:`agentlens.ui.make_server` (``http.server``). Every route is read-only:

=====================================  ==============================================
``GET /traces``                        traces, filtered and paginated
``GET /traces/{id}``                   one trace: its span tree (the report's viewer)
``GET /experiments``                   experiments (baseline marked), pick two to compare
``GET /experiments/{id-or-name}``      an experiment's report (the HTML report page)
``GET /compare?baseline=&candidate=``  a comparison (the HTML report page)
``GET /api/...``                       the same data as JSON
=====================================  ==============================================

Trace filters (pages and ``/api/traces``): ``name``, ``experiment`` (id or
name), ``case_id``, ``meta=key=value`` (repeatable; the value is read as JSON
when it is a JSON scalar, so ``repetition=0`` matches the number), ``errors``
(``1``: only traces where the run or a span failed; ``0``: only the others),
``limit`` (1-500, default 50) and ``offset``.
"""

from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import TypeVar
from urllib.parse import parse_qs, quote, unquote, urlencode, urlsplit

from jinja2 import ChoiceLoader, Environment, PackageLoader, StrictUndefined
from markupsafe import Markup
from pydantic import JsonValue

from agentlens import __version__
from agentlens.compare import (
    DEFAULT_CONFIDENCE,
    DEFAULT_RESAMPLES,
    Gates,
    compare_runs,
    evaluate_gates,
    experiment_runs,
)
from agentlens.report import ComparisonReport, ExperimentReport, HtmlRenderer, ReportFormat, render
from agentlens.report.html import _finite, _trace_payload, script_json
from agentlens.schema import EvalResult, Experiment, Trace
from agentlens.store import MetadataScalar, SQLiteStore, StoreError
from agentlens.suite import ExperimentNotFoundError, RunResult, find_experiment, get_baseline

__all__ = ["DEFAULT_LIMIT", "MAX_LIMIT", "HTTPError", "Response", "UIApp"]

DEFAULT_LIMIT = 50
"""Traces per page unless ``limit`` says otherwise."""

MAX_LIMIT = 500
"""Largest accepted ``limit``."""

_MAX_RESAMPLES = 100_000
_PAGE_SIZES = (25, 50, 100, 200)
_PREVIEW_CHARS = 120
_BASELINE_KEYWORD = "baseline"

_HTML = "text/html; charset=utf-8"
_JSON = "application/json; charset=utf-8"

_env = Environment(
    # UI pages first, then the HTML report's templates and assets (macros, CSS, JS).
    loader=ChoiceLoader(
        [PackageLoader("agentlens.ui", "templates"), PackageLoader("agentlens.report", "templates")]
    ),
    autoescape=True,
    undefined=StrictUndefined,
    trim_blocks=True,
    lstrip_blocks=True,
)


def _asset(name: str) -> Markup:
    loader = _env.loader
    assert loader is not None
    source, _, _ = loader.get_source(_env, name)
    return Markup(source)  # our own packaged file


@dataclass(frozen=True)
class Response:
    """An HTTP response: status, body and headers (``Content-Type`` included)."""

    status: int
    body: bytes
    content_type: str = _HTML
    headers: dict[str, str] = field(default_factory=dict)

    @property
    def text(self) -> str:
        """The body decoded as UTF-8."""
        return self.body.decode("utf-8")


class HTTPError(Exception):
    """A request that cannot be served: 400 (bad parameter), 404, 405, 503, ..."""

    def __init__(self, status: int, message: str) -> None:
        super().__init__(message)
        self.status = status
        self.message = message


# -- query parameters ----------------------------------------------------------------

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off"}

_T = TypeVar("_T", int, float)


class _Query:
    """The query string; empty values count as absent (blank form fields)."""

    def __init__(self, query: str) -> None:
        self.raw = query
        self.values = parse_qs(query, keep_blank_values=False)

    def text(self, name: str) -> str | None:
        values = self.values.get(name)
        return (values[0].strip() or None) if values else None

    def all(self, name: str) -> list[str]:
        return [value for value in self.values.get(name, []) if value.strip()]

    def number(
        self, name: str, kind: Callable[[str], _T], default: _T, low: _T, high: _T | None = None
    ) -> _T:
        text = self.text(name)
        if text is None:
            return default
        try:
            value = kind(text)
        except ValueError:
            raise HTTPError(400, f"{name} must be a number, got {text!r}") from None
        if value < low or (high is not None and value > high):
            bounds = f"between {low} and {high}" if high is not None else f">= {low}"
            raise HTTPError(400, f"{name} must be {bounds}, got {text!r}")
        return value

    def flag(self, name: str) -> bool | None:
        text = self.text(name)
        if text is None:
            return None
        if text.lower() in _TRUE:
            return True
        if text.lower() in _FALSE:
            return False
        raise HTTPError(400, f"{name} must be 1 or 0 (true/false), got {text!r}")


def _metadata_value(text: str) -> MetadataScalar:
    try:
        value = json.loads(text)
    except ValueError:
        return text
    return value if isinstance(value, str | int | float | bool) or value is None else text


def _metadata_filter(items: list[str]) -> dict[str, MetadataScalar]:
    found: dict[str, MetadataScalar] = {}
    for item in items:
        key, sep, value = item.partition("=")
        if not sep or not key.strip():
            raise HTTPError(400, f"meta must be key=value, got {item!r}")
        found[key.strip()] = _metadata_value(value.strip())
    return found


def _url(path: str, params: Mapping[str, str | int | list[str] | None] | None = None) -> str:
    """``path`` with ``params`` as a query string (``None`` and empty values dropped)."""
    pairs: list[tuple[str, str]] = []
    for key, value in (params or {}).items():
        values = value if isinstance(value, list) else [value]
        pairs += [(key, str(v)) for v in values if v is not None and v != ""]
    return path + ("?" + urlencode(pairs) if pairs else "")


def _path(*segments: str) -> str:
    return "/" + "/".join(quote(segment, safe="") for segment in segments)


# -- view models ---------------------------------------------------------------------


def _fmt_time(value: datetime) -> str:
    return value.strftime("%Y-%m-%d %H:%M:%S")


def _fmt_ms(value: float | None) -> str:
    if value is None:
        return "-"
    return f"{value / 1000:.3g} s" if value >= 1000 else f"{value:.3g} ms"


def _fmt_cost(value: float | None) -> str:
    return "-" if value is None else f"${value:.4g}"


def _preview(value: JsonValue) -> str:
    if value is None:
        return ""
    text = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
    text = " ".join(text.split())
    return text if len(text) <= _PREVIEW_CHARS else text[: _PREVIEW_CHARS - 1] + "…"


def _first_error(trace: Trace) -> str | None:
    if trace.error:
        return trace.error
    return next((f"{span.name}: {span.error}" for span in trace.spans if span.error), None)


def _str_meta(trace: Trace, key: str) -> str | None:
    value = trace.metadata.get(key)
    return value if isinstance(value, str) else None


def _repetition(trace: Trace) -> int:
    value = trace.metadata.get("repetition")
    return value if isinstance(value, int) and not isinstance(value, bool) else 0


def _verdict(trace: Trace, results: list[EvalResult]) -> str:
    """``error`` (the run or a span failed), ``fail`` (an evaluator failed), ``pass``
    (evaluated, all passed) or ``ok`` (no error, not evaluated)."""
    if _first_error(trace) is not None:
        return "error"
    if not results:
        return "ok"
    return "pass" if all(result.passed for result in results) else "fail"


def _trace_json(trace: Trace, results: list[EvalResult]) -> dict[str, JsonValue]:
    return {
        "id": trace.id,
        "name": trace.name,
        "start_time": trace.start_time.isoformat(),
        "latency_ms": trace.latency_ms,
        "cost_usd": trace.total_cost,
        "tokens": trace.total_tokens,
        "steps": trace.n_steps,
        "error": _first_error(trace),
        "status": _verdict(trace, results),
        "failed_evaluators": [r.evaluator for r in results if not r.passed],
        "case_id": _str_meta(trace, "case_id"),
        "experiment_id": _str_meta(trace, "experiment"),
        "metadata": trace.metadata,
    }


@dataclass(frozen=True)
class _TraceRow:
    id: str
    href: str
    name: str
    started: str
    case_id: str | None
    repetition: int
    experiment: str | None
    experiment_href: str | None
    status: str
    error: str | None
    failed: str
    """Failed evaluators, comma-separated."""
    latency: str
    cost: str
    tokens: int
    steps: int
    output: str


@dataclass(frozen=True)
class _ExperimentRow:
    id: str
    href: str
    name: str
    suite: str
    created: str
    model: str
    git: str
    is_baseline: bool
    vs_baseline: str | None
    traces_href: str


@dataclass
class _TraceFilters:
    name: str | None
    case_id: str | None
    meta: list[str]
    errors: bool | None
    limit: int
    offset: int
    experiment: str | None = None
    """The experiment's id (the filter may name it)."""
    has_more: bool = False

    @property
    def active(self) -> bool:
        """Whether any filter (not counting the page) is set."""
        return bool(self.name or self.case_id or self.meta or self.experiment) or (
            self.errors is not None
        )

    def params(self) -> dict[str, str | int | list[str] | None]:
        """The filters as query parameters (defaults left out), without the offset."""
        return {
            "name": self.name,
            "experiment": self.experiment,
            "case_id": self.case_id,
            "meta": self.meta,
            "errors": None if self.errors is None else int(self.errors),
            "limit": None if self.limit == DEFAULT_LIMIT else self.limit,
        }


def _experiment_label(exp: Experiment) -> str:
    return f"{exp.name} · {exp.suite} · {_fmt_time(exp.created_at)} · {exp.id[:8]}"


def _config_str(exp: Experiment, key: str) -> str:
    value = exp.config.get(key)
    return value if isinstance(value, str) else ""


class _UIRenderer(HtmlRenderer):
    """The HTML report with the UI's navigation and footer."""

    def __init__(
        self,
        chrome: dict[str, object],
        *,
        runs: list[RunResult],
        baseline_runs: list[RunResult] | None = None,
    ) -> None:
        super().__init__(runs=runs, baseline_runs=baseline_runs)
        self._ui_chrome = chrome

    def _chrome(self) -> dict[str, object]:
        return self._ui_chrome


# -- the application -----------------------------------------------------------------

_Handler = Callable[["UIApp", _Query, str], Response]


class UIApp:
    """The read-only web UI over one AgentLens database.

    Args:
        db: Path of the SQLite database. It is opened on the first request that
            needs it, and never created: while it does not exist, pages say so
            (status 503) instead of creating an empty database.
        gates: Gates checked on every comparison page (optional).
        resamples: Bootstrap resamples of the comparisons (the same default as
            ``agentlens compare``); ``/compare`` takes it as a query parameter too.
        seed: Bootstrap seed, as ``resamples``.
        confidence: Confidence level of the intervals, as ``resamples``.
    """

    def __init__(
        self,
        db: str | os.PathLike[str],
        *,
        gates: Gates | None = None,
        resamples: int = DEFAULT_RESAMPLES,
        seed: int = 0,
        confidence: float = DEFAULT_CONFIDENCE,
    ) -> None:
        self.db = Path(db)
        self.gates = gates
        self.resamples = resamples
        self.seed = seed
        self.confidence = confidence
        self._store: SQLiteStore | None = None
        self._lock = threading.Lock()

    # -- lifecycle -----------------------------------------------------------

    def close(self) -> None:
        """Close the database connection (if it was opened)."""
        with self._lock:
            if self._store is not None:
                self._store.close()
                self._store = None

    def store(self) -> SQLiteStore:
        """The opened store; :class:`HTTPError` 503 while the database does not exist."""
        with self._lock:
            if self._store is None:
                if not self.db.is_file():
                    raise HTTPError(
                        503,
                        f"no AgentLens database at {self.db}: run a suite first "
                        "(`agentlens run SUITE`) or start the UI with --db PATH",
                    )
                try:
                    self._store = SQLiteStore(self.db)
                except (StoreError, sqlite3.DatabaseError) as exc:
                    raise HTTPError(500, f"cannot open {self.db}: {exc}") from None
            return self._store

    # -- dispatch ------------------------------------------------------------

    def handle(self, method: str, target: str) -> Response:
        """Serve one request; ``target`` is the request path with its query string."""
        parts = urlsplit(target)
        path = parts.path.rstrip("/") or "/"
        is_api = path == "/api" or path.startswith("/api/")
        try:
            if method not in ("GET", "HEAD"):
                raise HTTPError(405, f"method {method} not allowed: the UI is read-only")
            if path == "/":
                return Response(302, b"", headers={"Location": "/traces"})
            for pattern, handler in _ROUTES:
                match = pattern.fullmatch(path)
                if match:
                    arg = unquote(match.group(1)) if match.groups() else ""
                    return handler(self, _Query(parts.query), arg)
            raise HTTPError(404, f"no page at {path}")
        except HTTPError as exc:
            return self._error(exc, is_api=is_api)
        except (StoreError, sqlite3.DatabaseError) as exc:
            return self._error(HTTPError(500, f"database error: {exc}"), is_api=is_api)

    def _error(self, exc: HTTPError, *, is_api: bool) -> Response:
        headers = {"Allow": "GET, HEAD"} if exc.status == 405 else {}
        if is_api:
            return _json({"error": exc.message}, status=exc.status, headers=headers)
        reason = {400: "Bad request", 404: "Not found", 405: "Method not allowed"}.get(
            exc.status, "Unavailable" if exc.status == 503 else "Error"
        )
        html = self._render(
            "error.html.j2", page="error", title=reason, status=exc.status, message=exc.message
        )
        return Response(exc.status, html, headers=headers)

    # -- rendering helpers ---------------------------------------------------

    def _chrome(self, page: str) -> dict[str, object]:
        nav = [
            {"label": "Traces", "href": "/traces", "active": page in ("traces", "trace")},
            {"label": "Experiments", "href": "/experiments", "active": page == "experiments"},
        ]
        return {
            "nav": nav,
            "footer": f"AgentLens {__version__} · local UI, read-only · database {self.db}",
        }

    def _render(self, template: str, *, page: str, **context: object) -> bytes:
        html = _env.get_template(template).render(
            **self._chrome(page),
            **context,
            page=page,
            css=_asset("report.css"),
            ui_css=_asset("ui.css"),
        )
        return html.encode("utf-8")

    def _experiment(self, key: str) -> Experiment:
        try:
            return find_experiment(self.store(), key)
        except ExperimentNotFoundError:
            raise HTTPError(404, f"no experiment with id or name {key!r}") from None

    # -- traces --------------------------------------------------------------

    def _list_traces(self, query: _Query) -> tuple[list[Trace], _TraceFilters]:
        """The page of traces asked for, and the filters as given (for links and forms)."""
        meta = query.all("meta")
        metadata = _metadata_filter(meta)
        f = _TraceFilters(
            name=query.text("name"),
            case_id=query.text("case_id"),
            meta=meta,
            errors=query.flag("errors"),
            limit=query.number("limit", int, DEFAULT_LIMIT, 1, MAX_LIMIT),
            offset=query.number("offset", int, 0, 0),
        )
        if f.case_id is not None:
            metadata["case_id"] = f.case_id
        experiment = query.text("experiment")
        if experiment is not None:
            f.experiment = self._experiment(experiment).id
        traces = self.store().list_traces(
            name=f.name,
            experiment_id=f.experiment,
            metadata=metadata or None,
            has_error=f.errors,
            limit=f.limit + 1,  # one more: is there a next page?
            offset=f.offset,
        )
        f.has_more = len(traces) > f.limit
        return traces[: f.limit], f

    def traces_page(self, query: _Query, _arg: str) -> Response:
        traces, f = self._list_traces(query)
        experiments = self.store().list_experiments()
        names = {exp.id: exp.name for exp in experiments}
        rows = []
        for trace in traces:
            exp_id = _str_meta(trace, "experiment")
            known = exp_id is not None and exp_id in names
            results = self.store().get_eval_results(trace.id)
            rows.append(
                _TraceRow(
                    id=trace.id,
                    href=_path("traces", trace.id),
                    name=trace.name,
                    started=_fmt_time(trace.start_time),
                    case_id=_str_meta(trace, "case_id"),
                    repetition=_repetition(trace),
                    experiment=names[exp_id] if known and exp_id else None,
                    experiment_href=_url("/traces", {"experiment": exp_id}) if known else None,
                    status=_verdict(trace, results),
                    error=_first_error(trace),
                    failed=", ".join(r.evaluator for r in results if not r.passed),
                    latency=_fmt_ms(trace.latency_ms),
                    cost=_fmt_cost(trace.total_cost),
                    tokens=trace.total_tokens,
                    steps=trace.n_steps,
                    output=_preview(trace.output),
                )
            )
        params = f.params()
        prev_href = None
        if f.offset:
            prev_href = _url("/traces", {**params, "offset": max(0, f.offset - f.limit) or None})
        next_href = None
        if f.has_more:
            next_href = _url("/traces", {**params, "offset": f.offset + f.limit})
        html = self._render(
            "traces.html.j2",
            page="traces",
            title="Traces",
            f=f,
            rows=rows,
            experiments=[{"id": e.id, "label": _experiment_label(e)} for e in experiments],
            sizes=sorted({*_PAGE_SIZES, f.limit}),
            range_label=f"{f.offset + 1}-{f.offset + len(rows)}" if rows else "",
            prev_href=prev_href,
            next_href=next_href,
            api_href=_url("/api/traces", {**params, "offset": f.offset or None}),
        )
        return Response(200, html)

    def traces_api(self, query: _Query, _arg: str) -> Response:
        traces, f = self._list_traces(query)
        return _json(
            {
                "traces": [
                    _trace_json(trace, self.store().get_eval_results(trace.id)) for trace in traces
                ],
                "limit": f.limit,
                "offset": f.offset,
                "next_offset": f.offset + f.limit if f.has_more else None,
            }
        )

    def _trace(self, trace_id: str) -> Trace:
        trace = self.store().get_trace(trace_id)
        if trace is None:
            raise HTTPError(404, f"no trace with id {trace_id!r}")
        return trace

    def trace_page(self, _query: _Query, trace_id: str) -> Response:
        trace = self._trace(trace_id)
        store = self.store()
        results = store.get_eval_results(trace.id)
        case_id = _str_meta(trace, "case_id")
        run = RunResult(
            case_id=case_id or "",
            repetition=_repetition(trace),
            trace=trace,
            eval_results=results,
        )
        payload = _trace_payload(run, None)
        if case_id is None:
            payload["case_id"] = None
        exp_id = _str_meta(trace, "experiment")
        experiment = store.get_experiment(exp_id) if exp_id else None
        html = self._render(
            "trace.html.j2",
            page="trace",
            title=trace.name,
            trace=trace,
            run=run,
            case_id=case_id,
            error=_first_error(trace),
            started=_fmt_time(trace.start_time),
            latency=_fmt_ms(trace.latency_ms),
            cost=_fmt_cost(trace.total_cost),
            experiment=experiment,
            experiment_href=_path("experiments", experiment.id) if experiment else None,
            siblings_href=_url("/traces", {"experiment": experiment.id, "case_id": case_id})
            if experiment
            else None,
            api_href=_path("api", "traces", trace.id),
            data=script_json({"traces": {trace.id: payload}}),
            js=_asset("report.js"),
        )
        return Response(200, html)

    def trace_api(self, _query: _Query, trace_id: str) -> Response:
        trace = self._trace(trace_id)
        results = self.store().get_eval_results(trace.id)
        return _json(
            {
                "trace": trace.model_dump(mode="json"),
                "eval_results": [result.model_dump(mode="json") for result in results],
            }
        )

    # -- experiments ---------------------------------------------------------

    def _list_experiments(self, query: _Query) -> list[Experiment]:
        return self.store().list_experiments(
            name=query.text("name"), suite=query.text("suite"), is_baseline=query.flag("baseline")
        )

    def experiments_page(self, query: _Query, _arg: str) -> Response:
        experiments = self._list_experiments(query)
        baselines = {
            exp.suite: exp for exp in reversed(self.store().list_experiments(is_baseline=True))
        }
        rows = []
        for exp in experiments:
            base = baselines.get(exp.suite)
            sha = _config_str(exp, "git_sha")
            rows.append(
                _ExperimentRow(
                    id=exp.id,
                    href=_path("experiments", exp.id),
                    name=exp.name,
                    suite=exp.suite,
                    created=_fmt_time(exp.created_at),
                    model=_config_str(exp, "model"),
                    git=sha[:12],
                    is_baseline=exp.is_baseline,
                    vs_baseline=_url("/compare", {"baseline": base.id, "candidate": exp.id})
                    if base is not None and base.id != exp.id
                    else None,
                    traces_href=_url("/traces", {"experiment": exp.id}),
                )
            )
        candidate = rows[0].id if rows else None
        default_base = baselines.get(experiments[0].suite) if experiments else None
        baseline = (
            default_base.id
            if default_base is not None and default_base.id != candidate
            else rows[1].id
            if len(rows) > 1
            else None
        )
        suites = sorted({exp.suite for exp in self.store().list_experiments()})
        html = self._render(
            "experiments.html.j2",
            page="experiments",
            title="Experiments",
            rows=rows,
            suites=suites,
            f={"name": query.text("name"), "suite": query.text("suite")},
            checked={"baseline": baseline, "candidate": candidate},
            has_gates=self.gates is not None,
        )
        return Response(200, html)

    def experiments_api(self, query: _Query, _arg: str) -> Response:
        experiments = self._list_experiments(query)
        return _json({"experiments": [exp.model_dump(mode="json") for exp in experiments]})

    def _experiment_report(self, key: str) -> tuple[ExperimentReport, list[RunResult]]:
        experiment = self._experiment(key)
        runs = experiment_runs(self.store(), experiment)
        return ExperimentReport.from_runs(experiment, runs), runs

    def experiment_page(self, _query: _Query, key: str) -> Response:
        report, runs = self._experiment_report(key)
        renderer = _UIRenderer(self._chrome("experiments"), runs=runs)
        return Response(200, renderer.experiment(report).encode("utf-8"))

    def experiment_api(self, _query: _Query, key: str) -> Response:
        report, _ = self._experiment_report(key)
        return Response(200, render(report, ReportFormat.JSON).encode("utf-8"), _JSON)

    # -- comparison ----------------------------------------------------------

    def _comparison(
        self, query: _Query
    ) -> tuple[ComparisonReport, list[RunResult], list[RunResult]]:
        candidate_key = query.text("candidate")
        if candidate_key is None:
            raise HTTPError(400, "candidate is required (an experiment id or name)")
        baseline_key = query.text("baseline") or _BASELINE_KEYWORD
        resamples = query.number("resamples", int, self.resamples, 1, _MAX_RESAMPLES)
        seed = query.number("seed", int, self.seed, 0)
        confidence = query.number("confidence", float, self.confidence, 0.0, 1.0)
        if not 0.0 < confidence < 1.0:
            raise HTTPError(400, f"confidence must be between 0 and 1, got {confidence:g}")
        candidate = self._experiment(candidate_key)
        if baseline_key == _BASELINE_KEYWORD:
            found = get_baseline(self.store(), candidate.suite)
            if found is None:
                raise HTTPError(
                    404,
                    f"suite {candidate.suite!r} has no baseline: pass baseline=ID or NAME "
                    "(or mark one with `agentlens baseline set EXPERIMENT`)",
                )
            baseline = found
        else:
            baseline = self._experiment(baseline_key)
        base_runs = experiment_runs(self.store(), baseline)
        cand_runs = experiment_runs(self.store(), candidate)
        comparison = compare_runs(
            base_runs,
            cand_runs,
            resamples=resamples,
            seed=seed,
            confidence=confidence,
            baseline_experiment=baseline,
            candidate_experiment=candidate,
        )
        gates = (
            evaluate_gates(self.gates, comparison=comparison, run=cand_runs) if self.gates else None
        )
        return ComparisonReport(comparison=comparison, gates=gates), base_runs, cand_runs

    def compare_page(self, query: _Query, _arg: str) -> Response:
        if not query.values:  # nothing picked yet: pick two experiments first
            return Response(302, b"", headers={"Location": "/experiments"})
        report, base_runs, cand_runs = self._comparison(query)
        renderer = _UIRenderer(self._chrome("experiments"), runs=cand_runs, baseline_runs=base_runs)
        return Response(200, renderer.comparison(report).encode("utf-8"))

    def compare_api(self, query: _Query, _arg: str) -> Response:
        report, _, _ = self._comparison(query)
        return Response(200, render(report, ReportFormat.JSON).encode("utf-8"), _JSON)


def _json(data: object, *, status: int = 200, headers: dict[str, str] | None = None) -> Response:
    try:
        text = json.dumps(data, ensure_ascii=False, allow_nan=False, indent=1)
    except ValueError:  # NaN/inf in user data: not JSON, so they become null
        text = json.dumps(_finite(data), ensure_ascii=False, indent=1)
    body = text.encode("utf-8")
    return Response(status, body, _JSON, headers or {})


_ROUTES: list[tuple[re.Pattern[str], _Handler]] = [
    (re.compile(r"/traces"), UIApp.traces_page),
    (re.compile(r"/traces/([^/]+)"), UIApp.trace_page),
    (re.compile(r"/experiments"), UIApp.experiments_page),
    (re.compile(r"/experiments/([^/]+)"), UIApp.experiment_page),
    (re.compile(r"/compare"), UIApp.compare_page),
    (re.compile(r"/api/traces"), UIApp.traces_api),
    (re.compile(r"/api/traces/([^/]+)"), UIApp.trace_api),
    (re.compile(r"/api/experiments"), UIApp.experiments_api),
    (re.compile(r"/api/experiments/([^/]+)"), UIApp.experiment_api),
    (re.compile(r"/api/compare"), UIApp.compare_api),
]
