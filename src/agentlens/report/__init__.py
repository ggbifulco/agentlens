"""Reports: an experiment's results or a comparison, rendered for humans or tools.

Build a report model, then render it in a format::

    from agentlens.report import ExperimentReport, render
    from agentlens.suite import run_suite

    result = run_suite("suites/support.yaml")
    print(render(ExperimentReport.from_runs(result.experiment, result)))           # text
    print(render(ExperimentReport.from_runs(result.experiment, result), "json"))   # JSON

:class:`ComparisonReport` wraps a :class:`~agentlens.compare.Comparison` and
its gate verdict; :func:`load_report` reads either kind back from JSON.

Formats available today are plain ``text`` and ``json``. The rich terminal
report (#26), ``markdown`` (#27) and ``html`` (#28) register their renderers
with :func:`register_renderer`.
"""

from agentlens.report.model import (
    CaseSummary,
    ComparisonReport,
    ExperimentReport,
    ReportError,
    load_report,
)
from agentlens.report.plain import JSONRenderer, PlainTextRenderer, fmt_value, table
from agentlens.report.render import (
    Renderer,
    ReportFormat,
    ReportFormatUnavailableError,
    available_formats,
    get_renderer,
    register_renderer,
    render,
)

__all__ = [
    "CaseSummary",
    "ComparisonReport",
    "ExperimentReport",
    "JSONRenderer",
    "PlainTextRenderer",
    "Renderer",
    "ReportError",
    "ReportFormat",
    "ReportFormatUnavailableError",
    "available_formats",
    "fmt_value",
    "get_renderer",
    "load_report",
    "register_renderer",
    "render",
    "table",
]
