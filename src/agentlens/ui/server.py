"""Serving :class:`~agentlens.ui.UIApp` over HTTP with the standard library.

``http.server.ThreadingHTTPServer`` is enough for a local, single-user,
read-only UI (one thread per request, no dependency). Two safeguards:

- the server binds ``127.0.0.1`` by default, so only this machine can reach it;
- when bound to a loopback address it only answers requests whose ``Host``
  header names that address (or ``localhost``), so a web page you visit cannot
  read your traces through DNS rebinding (its requests carry its own host name).

Every response has a strict ``Content-Security-Policy`` (no network access, no
framing; only the page's own inline style and script) and ``Cache-Control:
no-store``.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from agentlens.ui.app import Response, UIApp

__all__ = ["DEFAULT_HOST", "DEFAULT_PORT", "UIServer", "make_server", "serve"]

DEFAULT_HOST = "127.0.0.1"
"""Loopback: the UI is reachable from this machine only."""

DEFAULT_PORT = 8765

logger = logging.getLogger("agentlens.ui")

_SECURITY_HEADERS = {
    "Content-Security-Policy": (
        "default-src 'none'; style-src 'unsafe-inline'; script-src 'unsafe-inline'; "
        "img-src data:; form-action 'self'; base-uri 'none'; frame-ancestors 'none'"
    ),
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "no-referrer",
    "Cache-Control": "no-store",
}


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


class _Handler(BaseHTTPRequestHandler):
    server: UIServer
    server_version = "AgentLens-UI"
    protocol_version = "HTTP/1.1"

    def _serve(self, method: str, *, close: bool = False) -> None:
        if not self.server.host_allowed(self.headers.get("Host")):
            body = b"unexpected Host header: open the UI at its 127.0.0.1/localhost address\n"
            response = Response(400, body, "text/plain; charset=utf-8")
        else:
            response = self.server.app.handle(method, self.path)
        self.send_response(response.status)
        self.send_header("Content-Type", response.content_type)
        self.send_header("Content-Length", str(len(response.body)))
        for name, value in {**_SECURITY_HEADERS, **response.headers}.items():
            self.send_header(name, value)
        if close:
            self.send_header("Connection", "close")  # also sets self.close_connection
        self.end_headers()
        if method != "HEAD":
            self.wfile.write(response.body)

    def do_GET(self) -> None:
        self._serve("GET")

    def do_HEAD(self) -> None:
        self._serve("HEAD")

    def _refuse(self) -> None:
        # The app answers 405 for every other method. The request body (if any) is
        # not read, so the connection cannot be reused: say so and close it.
        self._serve(self.command, close=True)

    do_POST = do_PUT = do_PATCH = do_DELETE = do_OPTIONS = _refuse

    def log_message(self, format: str, *args: object) -> None:
        logger.info("%s %s", self.address_string(), format % args)


class UIServer(ThreadingHTTPServer):
    """A threading HTTP server for one :class:`UIApp` (see :func:`make_server`)."""

    daemon_threads = True

    def __init__(self, app: UIApp, host: str, port: int) -> None:
        self.app = app
        if ":" in host.strip("[]"):
            self.address_family = socket.AF_INET6
        super().__init__((host.strip("[]"), port), _Handler)
        self._allowed_hosts: set[str] | None = None
        if _is_loopback(host):
            port = self.port
            names = {"localhost", "127.0.0.1", "[::1]", host.strip("[]")}
            self._allowed_hosts = names | {f"{name}:{port}" for name in names}

    @property
    def port(self) -> int:
        """The bound port (the real one when ``0`` asked for an ephemeral port)."""
        return int(self.server_address[1])

    @property
    def url(self) -> str:
        """The UI's address, e.g. ``http://127.0.0.1:8765/``."""
        host = str(self.server_address[0])
        if ":" in host:
            host = f"[{host}]"
        return f"http://{host}:{self.port}/"

    def host_allowed(self, host: str | None) -> bool:
        """Whether a request's ``Host`` header is acceptable (always, off loopback)."""
        if self._allowed_hosts is None:
            return True
        return host is not None and host.lower() in self._allowed_hosts

    def server_close(self) -> None:
        super().server_close()
        self.app.close()


def make_server(app: UIApp, *, host: str = DEFAULT_HOST, port: int = DEFAULT_PORT) -> UIServer:
    """Bind a server for ``app`` (``port=0`` picks a free port); call ``serve_forever()``.

    Raises:
        OSError: if the address cannot be bound (e.g. the port is in use).
    """
    return UIServer(app, host, port)


def serve(server: UIServer, *, open_browser: bool = False) -> None:
    """Serve until interrupted (Ctrl+C), optionally opening the UI in a browser first.

    The socket is already listening, so the browser's first request waits in
    the queue until the loop starts. The server is closed on return.
    """
    if open_browser:
        webbrowser.open(server.url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
