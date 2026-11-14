"""The UI application without sockets: routes, filters, pagination, errors and escaping."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from agentlens.compare import load_gates
from agentlens.schema import Experiment, Trace
from agentlens.store import SQLiteStore
from agentlens.ui import UIApp
from agentlens.ui.app import _fmt_ms, _json
from tests.ui.conftest import EVIL, Data


def _ok(app: UIApp, target: str) -> str:
    response = app.handle("GET", target)
    assert response.status == 200, response.text
    return response.text


def _trace_ids(html: str) -> list[str]:
    return re.findall(r'<td class="num"><a href="/traces/([^"]+)">', html)


def _api(app: UIApp, target: str, status: int = 200) -> dict[str, object]:
    response = app.handle("GET", target)
    assert response.status == status, response.text
    assert response.content_type == "application/json; charset=utf-8"
    loaded = json.loads(response.body)
    assert isinstance(loaded, dict)
    return loaded


# -- routing ---------------------------------------------------------------------------


def test_root_redirects_to_traces(app: UIApp) -> None:
    response = app.handle("GET", "/")
    assert response.status == 302
    assert response.headers["Location"] == "/traces"


@pytest.mark.parametrize("target", ["/nope", "/traces/a/b", "/api", "/api/nope", "/static/x.js"])
def test_unknown_paths_are_404(app: UIApp, target: str) -> None:
    response = app.handle("GET", target)
    assert response.status == 404
    if target.startswith("/api"):
        assert json.loads(response.body) == {"error": f"no page at {target}"}
    else:
        assert "Not found" in response.text
        assert "HTTP 404" in response.text


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_only_get_and_head_are_allowed(app: UIApp, method: str) -> None:
    page = app.handle(method, "/traces")
    assert page.status == 405
    assert page.headers["Allow"] == "GET, HEAD"
    assert "read-only" in page.text
    api = app.handle(method, "/api/traces")
    assert api.status == 405
    assert "read-only" in json.loads(api.body)["error"]


def test_trailing_slash_is_ignored(app: UIApp) -> None:
    assert app.handle("GET", "/traces/").status == 200
    assert app.handle("HEAD", "/experiments/").status == 200


# -- traces ----------------------------------------------------------------------------


def test_traces_page_lists_every_trace_most_recent_first(app: UIApp, data: Data) -> None:
    html = _ok(app, "/traces")
    ids = _trace_ids(html)
    assert len(ids) == 14
    assert ids[:2] == [data.evil.id, data.standalone.id]
    assert ids[-1] == data.base_runs[0].trace.id
    assert "1-14" in html
    assert '<a href="/traces" class="active" aria-current="page">Traces</a>' in html
    # experiment names link to the experiment's traces, errors are flagged
    assert f'<a href="/traces?experiment={data.cand.id}">pr-1</a>' in html
    assert "ValueError: boom" in html
    assert "lookup: KeyError: &#39;order&#39;" in html
    assert '<span class="pill st-fail" title="failed: check">fail</span>' in html
    assert html.count('<span class="pill st-pass">pass</span>') == 8  # 6 baseline + q3, q4
    assert '<span class="pill">ok</span>' in html  # the evil trace: no evaluators
    assert html.count('<tr class="has-error">') == 5  # 3 failed + 2 errors
    assert 'rel="next"' not in html
    assert 'rel="prev"' not in html


def test_blank_filters_are_ignored(app: UIApp) -> None:
    html = _ok(app, "/traces?name=&experiment=&case_id=&meta=&errors=&limit=")
    assert len(_trace_ids(html)) == 14
    assert '<a href="/traces">Reset</a>' not in html


def test_filter_by_name(app: UIApp, data: Data) -> None:
    html = _ok(app, "/traces?name=adhoc")
    assert _trace_ids(html) == [data.standalone.id]
    assert '<a href="/traces">Reset</a>' in html
    assert 'value="adhoc"' in html


@pytest.mark.parametrize("key", ["id", "name"])
def test_filter_by_experiment_id_or_name(app: UIApp, data: Data, key: str) -> None:
    html = _ok(app, f"/traces?experiment={getattr(data.cand, key)}")
    assert _trace_ids(html) == [r.trace.id for r in reversed(data.cand_runs)]
    assert f'<option value="{data.cand.id}" selected>' in html


def test_unknown_experiment_filter_is_404(app: UIApp) -> None:
    response = app.handle("GET", "/traces?experiment=ghost")
    assert response.status == 404
    assert "no experiment with id or name &#39;ghost&#39;" in response.text


def test_filter_by_case_and_metadata(app: UIApp, data: Data) -> None:
    by_case = _trace_ids(_ok(app, "/traces?case_id=q1"))
    assert by_case == [data.cand_runs[1].trace.id, data.base_runs[1].trace.id]
    # a JSON scalar value is matched as such: repetition=0 is the number 0
    both = _ok(app, f"/traces?meta=repetition%3D0&meta=experiment%3D{data.base.id}")
    assert _trace_ids(both) == [r.trace.id for r in reversed(data.base_runs)]
    assert 'value="repetition=0"' in both
    assert _trace_ids(_ok(app, "/traces?meta=repetition%3D%220%22")) == []
    note = _ok(app, "/traces?meta=note%3D" + EVIL.replace("<", "%3C").replace(">", "%3E"))
    assert _trace_ids(note) == [data.evil.id]


def test_filter_errors_only_counts_span_errors(app: UIApp, data: Data) -> None:
    errors = _trace_ids(_ok(app, "/traces?errors=1"))
    assert errors == [data.standalone.id, data.cand_runs[-1].trace.id]
    ok = _trace_ids(_ok(app, "/traces?errors=false"))
    assert len(ok) == 12
    assert data.standalone.id not in ok


@pytest.mark.parametrize(
    ("query", "message"),
    [
        ("limit=abc", "limit must be a number, got &#39;abc&#39;"),
        ("limit=0", "limit must be between 1 and 500"),
        ("limit=501", "limit must be between 1 and 500"),
        ("offset=-1", "offset must be &gt;= 0"),
        ("offset=1.5", "offset must be a number"),
        ("errors=maybe", "errors must be 1 or 0"),
        ("meta=novalue", "meta must be key=value, got &#39;novalue&#39;"),
        ("meta=%3Dx", "meta must be key=value"),
    ],
)
def test_bad_trace_parameters_are_400(app: UIApp, query: str, message: str) -> None:
    response = app.handle("GET", f"/traces?{query}")
    assert response.status == 400
    assert message in response.text
    assert "Bad request" in response.text
    api = app.handle("GET", f"/api/traces?{query}")
    assert api.status == 400
    assert "error" in json.loads(api.body)


def test_pagination_links_keep_the_filters(app: UIApp, data: Data) -> None:
    first = _ok(app, "/traces?limit=5&errors=0")
    assert len(_trace_ids(first)) == 5
    assert 'href="/traces?errors=0&amp;limit=5&amp;offset=5" rel="next"' in first
    assert 'rel="prev"' not in first
    assert "1-5" in first
    assert '<option value="5" selected>5</option>' in first  # a non-standard page size

    middle = _ok(app, "/traces?limit=5&errors=0&offset=5")
    assert 'href="/traces?errors=0&amp;limit=5" rel="prev"' in middle
    assert 'href="/traces?errors=0&amp;limit=5&amp;offset=10" rel="next"' in middle

    last = _ok(app, "/traces?limit=5&errors=0&offset=10")
    assert len(_trace_ids(last)) == 2
    assert 'rel="next"' not in last
    assert "11-12" in last

    beyond = _ok(app, "/traces?offset=100")
    assert "No traces on this page." in beyond
    assert 'href="/traces?offset=50" rel="prev"' in beyond


def test_empty_states(app: UIApp, tmp_path: Path) -> None:
    assert "No trace matches the filters." in _ok(app, "/traces?name=nothing")
    empty = UIApp(tmp_path / "empty.db")
    SQLiteStore(tmp_path / "empty.db").close()
    try:
        assert "No traces yet" in _ok(empty, "/traces")
        assert "No experiments yet" in _ok(empty, "/experiments")
    finally:
        empty.close()


def test_traces_api(app: UIApp, data: Data) -> None:
    body = _api(app, "/api/traces?limit=2&offset=2")
    assert body["limit"] == 2
    assert body["offset"] == 2
    assert body["next_offset"] == 4
    traces = body["traces"]
    assert isinstance(traces, list)
    first = traces[0]
    boom = data.cand_runs[-1].trace
    assert first == {
        "id": boom.id,
        "name": "toy",
        "start_time": boom.start_time.isoformat(),
        "latency_ms": 100.0,
        "cost_usd": None,
        "tokens": 0,
        "steps": 1,
        "error": "ValueError: boom",
        "status": "error",
        "failed_evaluators": [],
        "case_id": "boom",
        "experiment_id": data.cand.id,
        "metadata": {"case_id": "boom", "repetition": 0, "experiment": data.cand.id},
    }
    statuses = {t["case_id"]: (t["status"], t["failed_evaluators"]) for t in traces}  # type: ignore[index, union-attr]
    assert statuses["q4"] == ("pass", [])
    all_traces = _api(app, "/api/traces")["traces"]
    by_id = {t["id"]: t for t in all_traces}  # type: ignore[union-attr, index]
    assert by_id[data.cand_runs[0].trace.id]["status"] == "fail"
    assert by_id[data.cand_runs[0].trace.id]["failed_evaluators"] == ["check"]
    assert by_id[data.standalone.id]["status"] == "error"
    assert by_id[data.evil.id]["status"] == "ok"
    end = _api(app, "/api/traces?offset=13")
    assert end["next_offset"] is None
    assert len(end["traces"]) == 1  # type: ignore[arg-type]


# -- one trace -------------------------------------------------------------------------


def _payload(html: str) -> dict[str, object]:
    found = re.search(r'<script type="application/json" id="agentlens-data">(.*?)</script>', html)
    assert found is not None
    loaded = json.loads(found.group(1))
    assert isinstance(loaded, dict)
    return loaded


def test_trace_page_shows_the_span_tree_inline(app: UIApp, data: Data) -> None:
    result = data.cand_runs[0]
    html = _ok(app, f"/traces/{result.trace.id}")
    assert f'data-inline="{result.trace.id}" open' in html
    assert "failed" in html  # q0 fails its evaluator in the candidate
    assert f'<a href="/experiments/{data.cand.id}">pr-1</a>' in html
    assert f'href="/traces?experiment={data.cand.id}&amp;case_id=q0"' in html
    assert f'href="/api/traces/{result.trace.id}"' in html
    trace = _payload(html)["traces"][result.trace.id]  # type: ignore[index]
    assert trace["case_id"] == "q0"
    assert trace["passed"] is False
    assert [e["evaluator"] for e in trace["evals"]] == ["check"]
    assert len(trace["spans"]) == 1
    assert "dialog.dataset.inline" in html  # the report's viewer script, inline mode


def test_standalone_trace_has_no_case(app: UIApp, data: Data) -> None:
    html = _ok(app, f"/traces/{data.standalone.id}")
    trace = _payload(html)["traces"][data.standalone.id]  # type: ignore[index]
    assert trace["case_id"] is None
    assert trace["spans"][0]["error"] == "KeyError: 'order'"
    assert "lookup: KeyError: &#39;order&#39;" in html
    assert "all runs of this case" not in html
    assert "Experiment</dt>" not in html


def test_unknown_trace_is_404(app: UIApp) -> None:
    response = app.handle("GET", "/traces/ghost")
    assert response.status == 404
    assert "no trace with id &#39;ghost&#39;" in response.text
    assert _api(app, "/api/traces/ghost", 404) == {"error": "no trace with id 'ghost'"}


def test_trace_api(app: UIApp, data: Data) -> None:
    result = data.cand_runs[0]
    body = _api(app, f"/api/traces/{result.trace.id}")
    assert Trace.model_validate(body["trace"]) == result.trace
    assert body["eval_results"] == [r.model_dump(mode="json") for r in result.eval_results]


# -- experiments -----------------------------------------------------------------------


def test_experiments_page(app: UIApp, data: Data) -> None:
    html = _ok(app, "/experiments")
    assert html.index("pr-1") < html.index(">main<")
    assert '<span class="tag baseline-tag">baseline</span>' in html
    assert '<tr class="is-baseline">' in html
    # defaults: the newest as candidate, its suite's baseline as baseline
    assert f'name="candidate" value="{data.cand.id}" aria-label="candidate: pr-1" checked' in html
    assert f'name="baseline" value="{data.base.id}" aria-label="baseline: main" checked' in html
    assert f'href="/compare?baseline={data.base.id}&amp;candidate={data.cand.id}"' in html
    assert html.count("vs baseline") == 1
    assert '<form method="get" action="/compare"' in html
    assert "gates checked" not in html


def test_experiments_page_filters(app: UIApp, data: Data) -> None:
    html = _ok(app, "/experiments?name=main")
    assert f'value="{data.base.id}"' in html
    assert f'value="{data.cand.id}"' not in html
    assert "No experiment matches the filters." in _ok(app, "/experiments?suite=other")
    assert app.handle("GET", "/experiments?baseline=x").status == 400


def test_experiments_page_without_baseline_picks_the_two_newest(app: UIApp, data: Data) -> None:
    with SQLiteStore(data.db) as store:
        store.save_experiment(data.base.model_copy(update={"is_baseline": False}))
    html = _ok(app, "/experiments")
    assert f'value="{data.base.id}" aria-label="baseline: main" checked' in html
    assert "vs baseline" not in html


def test_experiments_api(app: UIApp, data: Data) -> None:
    body = _api(app, "/api/experiments")
    experiments = [Experiment.model_validate(e) for e in body["experiments"]]  # type: ignore[attr-defined]
    assert experiments == [data.cand, data.base]
    only = _api(app, "/api/experiments?baseline=1")
    assert [e["name"] for e in only["experiments"]] == ["main"]  # type: ignore[attr-defined]


def test_experiment_page_is_the_html_report_with_navigation(app: UIApp, data: Data) -> None:
    html = _ok(app, "/experiments/pr-1")
    assert "<title>AgentLens: pr-1</title>" in html
    assert '<a href="/experiments" class="active" aria-current="page">Experiments</a>' in html
    assert "local UI, read-only" in html
    assert "self-contained" not in html
    assert set(_payload(html)["traces"]) == {r.trace.id for r in data.cand_runs}  # type: ignore[arg-type]
    assert app.handle("GET", "/experiments/ghost").status == 404


def test_experiment_api(app: UIApp, data: Data) -> None:
    body = _api(app, f"/api/experiments/{data.cand.id}")
    assert body["experiment"]["name"] == "pr-1"  # type: ignore[index]
    assert body["runs"] == 6
    assert _api(app, "/api/experiments/ghost", 404)["error"] == (
        "no experiment with id or name 'ghost'"
    )


# -- compare ---------------------------------------------------------------------------


def test_compare_without_parameters_redirects_to_the_picker(app: UIApp) -> None:
    response = app.handle("GET", "/compare")
    assert response.status == 302
    assert response.headers["Location"] == "/experiments"


def test_compare_page(app: UIApp, data: Data) -> None:
    html = _ok(app, f"/compare?baseline={data.base.id}&candidate={data.cand.id}")
    assert "<title>AgentLens: pr-1 vs main</title>" in html
    assert "200 resamples" in html  # the app's default
    assert "regression" in html
    assert '<details class="case st-regression"' in html
    traces = _payload(html)["traces"]
    assert {t["side"] for t in traces.values()} == {"baseline", "candidate"}  # type: ignore[attr-defined]
    assert 'id="gates"' not in html


def test_compare_defaults_to_the_suite_baseline(app: UIApp) -> None:
    html = _ok(app, "/compare?candidate=pr-1&baseline=")
    assert "pr-1 vs main" in html


def test_compare_parameters(app: UIApp) -> None:
    html = _ok(app, "/compare?candidate=pr-1&resamples=50&seed=3&confidence=0.9")
    assert "50 resamples, 90% CI, seed 3" in html


@pytest.mark.parametrize(
    ("query", "status", "message"),
    [
        ("baseline=main", 400, "candidate is required"),
        ("candidate=ghost", 404, "no experiment with id or name"),
        ("candidate=pr-1&baseline=ghost", 404, "no experiment with id or name"),
        ("candidate=pr-1&resamples=0", 400, "resamples must be between 1 and 100000"),
        ("candidate=pr-1&confidence=1", 400, "confidence must be between 0 and 1"),
        ("candidate=pr-1&confidence=x", 400, "confidence must be a number"),
        ("candidate=pr-1&seed=-1", 400, "seed must be &gt;= 0"),
    ],
)
def test_bad_compare_parameters(app: UIApp, query: str, status: int, message: str) -> None:
    response = app.handle("GET", f"/compare?{query}")
    assert response.status == status
    assert message in response.text


def test_compare_without_a_baseline_in_the_suite(app: UIApp, data: Data) -> None:
    with SQLiteStore(data.db) as store:
        store.save_experiment(data.base.model_copy(update={"is_baseline": False}))
    response = app.handle("GET", "/compare?candidate=pr-1")
    assert response.status == 404
    assert "suite &#39;toy&#39; has no baseline" in response.text


def test_compare_with_gates(data: Data, tmp_path: Path) -> None:
    (tmp_path / "gates.yaml").write_text("success_rate: {min: 0.9}\n", encoding="utf-8")
    gates = load_gates(tmp_path / "gates.yaml")
    app = UIApp(data.db, gates=gates, resamples=100)
    try:
        html = _ok(app, "/compare?candidate=pr-1")
        assert 'id="gates"' in html
        assert "gates failed" in html
        assert "gates checked" in _ok(app, "/experiments")
        body = _api(app, "/api/compare?candidate=pr-1")
        assert body["gates"]["passed"] is False  # type: ignore[index]
    finally:
        app.close()


def test_compare_api(app: UIApp, data: Data) -> None:
    body = _api(app, f"/api/compare?baseline=main&candidate={data.cand.id}")
    comparison = body["comparison"]
    assert isinstance(comparison, dict)
    assert comparison["resamples"] == 200
    assert comparison["candidate"]["id"] == data.cand.id
    regressed = [c["case_id"] for c in comparison["cases"] if c["status"] == "regression"]
    assert sorted(regressed) == ["q0", "q1", "q2"]
    assert body["gates"] is None
    assert _api(app, "/api/compare", 400)["error"].startswith("candidate is required")  # type: ignore[union-attr]


# -- the database ----------------------------------------------------------------------


def test_missing_database_is_reported_then_picked_up(data: Data, tmp_path: Path) -> None:
    path = tmp_path / "later" / "agentlens.db"
    app = UIApp(path)
    try:
        page = app.handle("GET", "/traces")
        assert page.status == 503
        assert "no AgentLens database at" in page.text
        assert "Unavailable" in page.text
        api = app.handle("GET", "/api/experiments")
        assert api.status == 503
        assert not path.parent.exists()  # never created by the UI
        path.parent.mkdir()
        shutil.copy(data.db, path)
        assert app.handle("GET", "/traces").status == 200
    finally:
        app.close()
        app.close()  # idempotent


def test_unreadable_database_is_500(tmp_path: Path) -> None:
    path = tmp_path / "broken.db"
    path.write_bytes(b"this is not a sqlite database at all" * 100)
    app = UIApp(path)
    response = app.handle("GET", "/api/traces")
    assert response.status == 500
    assert "cannot open" in json.loads(response.body)["error"]
    page = app.handle("GET", "/experiments")
    assert page.status == 500
    assert "Error" in page.text


def test_database_errors_while_serving_are_500(app: UIApp, monkeypatch: pytest.MonkeyPatch) -> None:
    import sqlite3

    def broken(*args: object, **kwargs: object) -> None:
        raise sqlite3.OperationalError("disk I/O error")

    monkeypatch.setattr(app.store(), "list_traces", broken)
    response = app.handle("GET", "/api/traces")
    assert response.status == 500
    assert json.loads(response.body) == {"error": "database error: disk I/O error"}


# -- escaping --------------------------------------------------------------------------


def test_hostile_strings_are_escaped_everywhere(app: UIApp, data: Data) -> None:
    pages = [
        _ok(app, "/traces"),
        _ok(app, f"/traces/{data.evil.id}"),
        app.handle("GET", "/traces?experiment=" + EVIL.replace("<", "%3C")).text,
        app.handle("GET", "/traces/" + EVIL.replace("<", "%3C").replace("/", "%2F")).text,
    ]
    for html in pages:
        assert EVIL not in html
        assert "<script>alert" not in html
    assert "&lt;/script&gt;&lt;script&gt;alert(1)&lt;/script&gt;" in pages[0]
    trace_page = pages[1]
    assert trace_page.count("<script") == 2  # the data block and the viewer, nothing injected
    trace = _payload(trace_page)["traces"][data.evil.id]  # type: ignore[index]
    assert trace["name"] == EVIL
    assert trace["case_id"] == EVIL
    assert "\\u003c/script\\u003e" in trace_page


def test_hostile_experiment_names_are_escaped(app: UIApp, data: Data) -> None:
    with SQLiteStore(data.db) as store:
        store.save_experiment(data.cand.model_copy(update={"name": EVIL, "suite": EVIL}))
    for target in ("/experiments", "/traces", f"/compare?candidate={data.cand.id}&baseline=main"):
        html = _ok(app, target)
        assert EVIL not in html
        assert "&lt;/script&gt;" in html


def test_formatting_helpers() -> None:
    assert _fmt_ms(None) == "-"
    assert _fmt_ms(1500.0) == "1.5 s"
    assert _fmt_ms(12.34) == "12.3 ms"


def test_json_api_turns_non_finite_numbers_into_null() -> None:
    response = _json({"x": float("nan"), "y": [float("inf"), 1.5]})
    assert json.loads(response.body) == {"x": None, "y": [None, 1.5]}
