"""The UI over real HTTP: a server on an ephemeral port in a thread, requests with httpx."""

from __future__ import annotations

import logging
import socket

import httpx
import pytest

from agentlens.ui import DEFAULT_HOST, DEFAULT_PORT, UIApp, make_server, serve
from agentlens.ui.server import _is_loopback
from tests.ui.conftest import EVIL, Data, Live


def test_defaults() -> None:
    assert DEFAULT_HOST == "127.0.0.1"
    assert DEFAULT_PORT == 8765


def test_binds_loopback_on_an_ephemeral_port(live: Live) -> None:
    host, port = live.server.server_address[:2]
    assert host == "127.0.0.1"
    assert port != 0
    assert live.server.url == f"http://127.0.0.1:{port}/"


def test_pages_and_api(live: Live, data: Data) -> None:
    client = live.client
    root = client.get("/")
    assert root.status_code == 302
    assert root.headers["location"] == "/traces"
    traces = client.get("/traces", follow_redirects=True)
    assert traces.status_code == 200
    assert traces.headers["content-type"] == "text/html; charset=utf-8"
    assert "pr-1" in traces.text
    trace = client.get(f"/traces/{data.cand_runs[0].trace.id}")
    assert trace.status_code == 200
    assert 'id="trace-viewer"' in trace.text
    assert client.get("/experiments").status_code == 200
    compare = client.get("/compare", params={"candidate": "pr-1", "baseline": "main"})
    assert compare.status_code == 200
    assert "pr-1 vs main" in compare.text
    api = client.get("/api/traces", params={"experiment": "main", "limit": 2})
    assert api.status_code == 200
    assert api.headers["content-type"] == "application/json; charset=utf-8"
    assert [t["case_id"] for t in api.json()["traces"]] == ["q5", "q4"]
    assert api.json()["next_offset"] == 2
    assert client.get("/api/experiments").json()["experiments"][0]["name"] == "pr-1"
    cmp = client.get("/api/compare", params={"candidate": "pr-1"}).json()
    assert cmp["comparison"]["baseline"]["name"] == "main"


@pytest.mark.parametrize(
    ("path", "status"),
    [
        ("/nope", 404),
        ("/traces/ghost", 404),
        ("/api/traces/ghost", 404),
        ("/experiments/ghost", 404),
        ("/traces?limit=abc", 400),
        ("/api/traces?errors=maybe", 400),
        ("/api/compare?baseline=main", 400),
        ("/compare?candidate=pr-1&confidence=2", 400),
    ],
)
def test_error_statuses(live: Live, path: str, status: int) -> None:
    response = live.client.get(path)
    assert response.status_code == status
    if path.startswith("/api/"):
        assert "error" in response.json()


def test_security_headers(live: Live) -> None:
    response = live.client.get("/traces")
    csp = response.headers["content-security-policy"]
    assert "default-src 'none'" in csp
    assert "frame-ancestors 'none'" in csp
    assert "connect-src" not in csp  # no fetch/XHR anywhere
    assert response.headers["x-content-type-options"] == "nosniff"
    assert response.headers["referrer-policy"] == "no-referrer"
    assert response.headers["cache-control"] == "no-store"
    assert response.headers["server"].startswith("AgentLens-UI")


def test_head_has_headers_but_no_body(live: Live) -> None:
    response = live.client.head("/traces")
    assert response.status_code == 200
    assert int(response.headers["content-length"]) > 1000
    assert response.content == b""


@pytest.mark.parametrize("method", ["POST", "PUT", "PATCH", "DELETE", "OPTIONS"])
def test_writes_are_refused(live: Live, method: str) -> None:
    response = live.client.request(method, "/api/traces", content=b'{"x": 1}')
    assert response.status_code == 405
    assert response.headers["allow"] == "GET, HEAD"
    # the connection is closed (the body was not read); the next request still works
    assert live.client.get("/api/traces").status_code == 200


def test_keep_alive_serves_several_requests(live: Live) -> None:
    for _ in range(3):
        assert live.client.get("/api/experiments").status_code == 200


@pytest.mark.parametrize("host", ["evil.example", "evil.example:8765", "127.0.0.2"])
def test_foreign_host_headers_are_refused(live: Live, host: str) -> None:
    response = live.client.get("/api/traces", headers={"Host": host})
    assert response.status_code == 400
    assert "unexpected Host header" in response.text


def test_loopback_host_names_are_accepted(live: Live) -> None:
    port = live.server.port
    for host in (f"localhost:{port}", f"127.0.0.1:{port}", "LOCALHOST", f"[::1]:{port}"):
        assert live.client.get("/api/traces", headers={"Host": host}).status_code == 200


def test_host_check_is_off_when_not_bound_to_loopback(app: UIApp) -> None:
    server = make_server(app, host="0.0.0.0", port=0)
    try:
        assert server.host_allowed("my-laptop.local:8765")
        assert server.host_allowed(None)
    finally:
        server.server_close()


def test_missing_host_header_is_refused_on_loopback(app: UIApp) -> None:
    server = make_server(app, port=0)
    try:
        assert not server.host_allowed(None)
    finally:
        server.server_close()


def test_ipv6_loopback(app: UIApp) -> None:
    if not socket.has_ipv6:  # pragma: no cover - depends on the machine
        pytest.skip("no IPv6")
    try:
        server = make_server(app, host="::1", port=0)
    except OSError:  # pragma: no cover - IPv6 disabled
        pytest.skip("cannot bind ::1")
    try:
        assert server.url == f"http://[::1]:{server.port}/"
        assert server.host_allowed(f"[::1]:{server.port}")
    finally:
        server.server_close()


def test_port_in_use_raises(live: Live, app: UIApp) -> None:
    with pytest.raises(OSError, match="in use"):
        make_server(app, port=live.server.port)


def test_loopback_detection_and_ipv6_urls(app: UIApp) -> None:
    assert _is_loopback("localhost")
    assert _is_loopback("127.0.0.1")
    assert _is_loopback("[::1]")
    assert not _is_loopback("0.0.0.0")
    assert not _is_loopback("my-laptop.local")
    server = make_server(app, host="localhost", port=0)
    try:
        assert server.host_allowed(f"localhost:{server.port}")
        server.server_address = ("::1", 9999)
        assert server.url == "http://[::1]:9999/"
    finally:
        server.server_close()


def test_escaping_over_http(live: Live, data: Data) -> None:
    for path in ("/traces", f"/traces/{data.evil.id}"):
        text = live.client.get(path).text
        assert EVIL not in text
        assert "<script>alert" not in text


def test_requests_are_logged(live: Live, caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.INFO, logger="agentlens.ui"):
        live.client.get("/api/experiments")
    assert any("GET /api/experiments" in r.getMessage() for r in caplog.records)


def test_serve_opens_the_browser_and_stops_on_ctrl_c(
    app: UIApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    server = make_server(app, port=0)

    def interrupted(poll_interval: float = 0.5) -> None:
        raise KeyboardInterrupt

    monkeypatch.setattr(server, "serve_forever", interrupted)
    serve(server, open_browser=True)
    assert opened == [server.url]
    with pytest.raises(httpx.ConnectError):
        httpx.get(server.url, timeout=2)  # closed


def test_serve_without_browser(app: UIApp, monkeypatch: pytest.MonkeyPatch) -> None:
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    server = make_server(app, port=0)
    monkeypatch.setattr(server, "serve_forever", lambda poll_interval=0.5: None)
    serve(server)
    assert opened == []
