"""Reports: an experiment's results or a comparison, rendered for humans or tools.

Build a report model, then render it in a format::

    from agentlens.report import ExperimentReport, render
    from agentlens.suite import run_suite

    result = run_suite("suites/support.yaml")
    print(render(ExperimentReport.from_runs(result.experiment, result)))           # text
    print(render(ExperimentReport.from_runs(result.experiment, result), "json"))   # JSON

:class:`ComparisonReport` wraps a :class:`~agentlens.compare.Comparison` and
its gate verdict; :func:`load_report` reads either kind back from JSON.

Formats available today are plain ``text``, ``rich`` (coloured tables for a
terminal: ``render(report, "rich")``, or print :func:`comparison_renderable` /
:func:`experiment_renderable` to a ``rich`` console), ``json`` and
``markdown`` (GitHub-flavoured, for PR comments: :class:`MarkdownRenderer`,
starting with the hidden :func:`marker_comment`) and ``html`` (one
self-contained page with a span-tree trace viewer: :func:`render_html` with
the runs, so the traces are embedded).
"""

from agentlens.report.html import HtmlRenderer, render_html
from agentlens.report.markdown import (
    DEFAULT_MARKER,
    MarkdownRenderer,
    marker_comment,
)
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
from agentlens.report.terminal import (
    RichRenderer,
    comparison_renderable,
    experiment_renderable,
)

__all__ = [
    "DEFAULT_MARKER",
    "CaseSummary",
    "ComparisonReport",
    "ExperimentReport",
    "HtmlRenderer",
    "JSONRenderer",
    "MarkdownRenderer",
    "PlainTextRenderer",
    "Renderer",
    "ReportError",
    "ReportFormat",
    "ReportFormatUnavailableError",
    "RichRenderer",
    "available_formats",
    "comparison_renderable",
    "experiment_renderable",
    "fmt_value",
    "get_renderer",
    "load_report",
    "marker_comment",
    "register_renderer",
    "render",
    "render_html",
    "table",
]
