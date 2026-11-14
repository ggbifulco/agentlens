"""Local web UI to browse traces and experiments (FR-C4): ``agentlens ui``.

::

    agentlens ui                       # http://127.0.0.1:8765/, opens a browser
    agentlens --db runs.db ui --port 9000 --no-open

From Python::

    from agentlens.ui import UIApp, make_server, serve

    server = make_server(UIApp("runs.db"), port=0)   # 0: any free port
    print(server.url)
    serve(server)                                    # until Ctrl+C

Pages: traces (filter by name, experiment, case, metadata, errors; paginated),
one trace's span tree, experiments (baseline marked), an experiment's report and
the comparison of two experiments (metrics with confidence intervals, optional
gates, every case side by side). The pages reuse the HTML report's templates,
CSS and trace viewer; ``/api/...`` serves the same data as JSON. Read-only,
standard library server, no JavaScript framework, nothing loaded from the
network.
"""

from agentlens.ui.app import DEFAULT_LIMIT, MAX_LIMIT, HTTPError, Response, UIApp
from agentlens.ui.server import DEFAULT_HOST, DEFAULT_PORT, UIServer, make_server, serve

__all__ = [
    "DEFAULT_HOST",
    "DEFAULT_LIMIT",
    "DEFAULT_PORT",
    "MAX_LIMIT",
    "HTTPError",
    "Response",
    "UIApp",
    "UIServer",
    "make_server",
    "serve",
]
