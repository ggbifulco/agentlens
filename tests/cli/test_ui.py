"""``agentlens ui``: wiring of options to the server, without blocking.

``serve_forever`` is replaced by a version that answers exactly one request
made from a background thread, then returns, so the command really binds,
serves the app over HTTP and shuts down.
"""

from __future__ import annotations

import socket
import threading
from dataclasses import dataclass, field
from pathlib import Path

import httpx
import pytest

from agentlens.cli import app
from agentlens.ui import UIServer
from tests.cli.conftest import Toy, plain


@dataclass
class Served:
    """What the patched server saw."""

    urls: list[str] = field(default_factory=list)
    responses: list[httpx.Response] = field(default_factory=list)
    opened: list[str] = field(default_factory=list)
    paths: list[str] = field(default_factory=lambda: ["/api/experiments"])


@pytest.fixture
def served(monkeypatch: pytest.MonkeyPatch) -> Served:
    seen = Served()

    def one_request(self: UIServer, poll_interval: float = 0.5) -> None:
        seen.urls.append(self.url)
        for path in seen.paths:
            client = threading.Thread(
                target=lambda p=path: seen.responses.append(
                    httpx.get(self.url.rstrip("/") + p, timeout=10)
                )
            )
            client.start()
            self.handle_request()
            client.join(10)

    monkeypatch.setattr(UIServer, "serve_forever", one_request)
    monkeypatch.setattr("webbrowser.open", seen.opened.append)
    return seen


def test_ui_serves_the_database(ran: Toy, served: Served) -> None:
    served.paths = ["/api/experiments", "/traces?experiment=pr-1"]
    result = ran("ui", "--port", "0", "--no-open")
    assert result.exit_code == 0, result.output
    (url,) = served.urls
    assert url.startswith("http://127.0.0.1:")
    assert not url.endswith(":0/")
    assert f"AgentLens UI on {url}" in result.output
    assert "press Ctrl+C to stop" in result.output
    experiments, traces = served.responses
    assert [e["name"] for e in experiments.json()["experiments"]] == ["pr-1", "main"]
    assert traces.status_code == 200
    assert "boom" in traces.text
    assert served.opened == []


def test_ui_opens_the_browser_by_default(ran: Toy, served: Served) -> None:
    result = ran("ui", "--port", "0")
    assert result.exit_code == 0, result.output
    assert served.opened == served.urls


def test_ui_db_option_overrides_the_global_one(ran: Toy, served: Served, tmp_path: Path) -> None:
    result = ran.runner.invoke(
        app,
        ["--db", str(tmp_path / "other.db"), "ui", "--db", str(ran.db), "--port", "0", "--no-open"],
    )
    assert result.exit_code == 0, result.output
    assert len(served.responses[0].json()["experiments"]) == 2
    assert str(ran.db) in result.output


def test_ui_reads_the_db_from_the_environment(
    ran: Toy, served: Served, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setenv("AGENTLENS_DB", str(ran.db))
    result = ran("ui", "--port", "0", "--no-open", db=False)
    assert result.exit_code == 0, result.output
    assert len(served.responses[0].json()["experiments"]) == 2


def test_ui_with_a_missing_database_starts_and_says_so(toy: Toy, served: Served) -> None:
    result = toy("ui", "--port", "0", "--no-open")
    assert result.exit_code == 0, result.output
    assert f"note: no AgentLens database at {toy.db}" in plain(result.output)
    (response,) = served.responses
    assert response.status_code == 503
    assert not toy.db.exists()


def test_ui_with_gates(ran: Toy, served: Served) -> None:
    gates = ran.write("gates.yaml", "success_rate: {min: 0.99}\n")
    served.paths = ["/compare?candidate=pr-1"]
    result = ran("ui", "--port", "0", "--no-open", "--gates", str(gates))
    assert result.exit_code == 0, result.output
    assert "gates failed" in served.responses[0].text


def test_ui_with_an_invalid_gates_file_is_a_usage_error(ran: Toy, served: Served) -> None:
    gates = ran.write("gates.yaml", "nonsense: {min: 1}\n")
    result = ran("ui", "--port", "0", "--no-open", "--gates", str(gates))
    assert result.exit_code == 2
    assert "error:" in plain(result.output)
    assert served.urls == []


def test_ui_port_in_use_is_a_usage_error(ran: Toy, served: Served) -> None:
    with socket.socket() as busy:
        busy.bind(("127.0.0.1", 0))
        busy.listen()
        port = busy.getsockname()[1]
        result = ran("ui", "--port", str(port), "--no-open")
    assert result.exit_code == 2
    assert f"error: cannot listen on 127.0.0.1:{port}" in plain(result.output)
    assert served.urls == []


def test_ui_rejects_bad_ports(ran: Toy) -> None:
    result = ran("ui", "--port", "70000")
    assert result.exit_code == 2


def test_ui_is_listed_in_help(toy: Toy) -> None:
    result = toy("ui", "--help")
    assert result.exit_code == 0
    text = plain(result.output)
    for option in ("--host", "--port", "--db", "--gates", "--open", "--no-open"):
        assert option in text
