"""The ``agentlens`` command-line interface (``docs/design.md`` §8).

::

    agentlens run suites/support.yaml --name pr-123 --gates gates.yaml
    agentlens compare --baseline baseline --candidate pr-123 --gates gates.yaml
    agentlens report pr-123
    agentlens baseline set main
    agentlens traces ls --experiment pr-123

Exit codes: :data:`EXIT_OK` (0), :data:`EXIT_GATE_FAILED` (1, a gate failed)
and :data:`EXIT_USAGE` (2, bad arguments or configuration: invalid suite or
gates file, unknown experiment or trace, unreadable file). ``--db`` (or
``$AGENTLENS_DB``) picks the database, default ``.agentlens/agentlens.db``.
"""

from agentlens.cli._common import DB_ENV, EXIT_GATE_FAILED, EXIT_OK, EXIT_USAGE
from agentlens.cli.app import BASELINE_KEYWORD, app, main

__all__ = [
    "BASELINE_KEYWORD",
    "DB_ENV",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "EXIT_USAGE",
    "app",
    "main",
]
