"""``agentlens ui``: serve the local web UI (:mod:`agentlens.ui`)."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from agentlens.cli._common import _state, fail, user_errors
from agentlens.compare import Gates, load_gates
from agentlens.ui import DEFAULT_HOST, DEFAULT_PORT, UIApp, make_server, serve

__all__ = ["ui"]


def ui(
    ctx: typer.Context,
    host: Annotated[
        str,
        typer.Option(
            "--host",
            help="Address to listen on. The default is reachable from this machine only; "
            "anything else exposes your traces to the network.",
        ),
    ] = DEFAULT_HOST,
    port: Annotated[
        int, typer.Option("--port", "-p", min=0, max=65535, help="Port (0: any free port).")
    ] = DEFAULT_PORT,
    db: Annotated[
        Path | None,
        typer.Option(
            "--db",
            help="SQLite database to browse [default: the global --db / $AGENTLENS_DB].",
            show_default=False,
        ),
    ] = None,
    gates: Annotated[
        Path | None,
        typer.Option("--gates", "-g", help="Gates YAML file checked on every comparison page."),
    ] = None,
    open_browser: Annotated[
        bool, typer.Option("--open/--no-open", help="Open the UI in a web browser.")
    ] = True,
) -> None:
    """Browse traces and experiments in a local, read-only web UI.

    Lists and filters traces (name, experiment, case, metadata, errors only),
    shows a trace's span tree, lists experiments with the baseline marked, and
    compares two of them (metrics with confidence intervals, gates, every case
    side by side). The same data is served as JSON under /api/. Stop with Ctrl+C.
    """
    path = db if db is not None else _state(ctx).db
    with user_errors():
        loaded: Gates | None = load_gates(gates) if gates is not None else None
    app = UIApp(path, gates=loaded)
    try:
        server = make_server(app, host=host, port=port)
    except OSError as exc:
        fail(f"cannot listen on {host}:{port}: {exc.strerror or exc}")
    if not path.is_file():
        typer.echo(
            f"note: no AgentLens database at {path} yet; the UI will show it once "
            "`agentlens run` creates it",
            err=True,
        )
    typer.echo(f"AgentLens UI on {server.url} (database {path}); press Ctrl+C to stop", err=True)
    serve(server, open_browser=open_browser)
