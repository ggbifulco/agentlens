"""Report formats and the renderer registry.

A renderer turns a report model into text. ``text`` and ``json`` are built in;
``markdown`` and ``html`` are reserved for the reporters of #27 and #28, which
plug in with :func:`register_renderer` (the rich terminal report of #26 will
do the same), so the CLI's ``agentlens report --format`` needs no change.
"""

from __future__ import annotations

from enum import StrEnum
from typing import Protocol

from agentlens.report.model import ComparisonReport, ExperimentReport
from agentlens.report.plain import JSONRenderer, PlainTextRenderer

__all__ = [
    "Renderer",
    "ReportFormat",
    "ReportFormatUnavailableError",
    "available_formats",
    "get_renderer",
    "register_renderer",
    "render",
]


class ReportFormat(StrEnum):
    """Output formats of ``agentlens report``."""

    TEXT = "text"
    JSON = "json"
    MARKDOWN = "markdown"
    HTML = "html"


class Renderer(Protocol):
    """Renders both kinds of report in one format."""

    def experiment(self, report: ExperimentReport) -> str:
        """One experiment's results."""
        ...

    def comparison(self, report: ComparisonReport) -> str:
        """A candidate-vs-baseline comparison."""
        ...


class ReportFormatUnavailableError(LookupError):
    """No renderer is registered for the requested format (yet)."""


_PLANNED = {
    ReportFormat.MARKDOWN: "the Markdown report (#27)",
    ReportFormat.HTML: "the HTML report (#28)",
}

_RENDERERS: dict[ReportFormat, Renderer] = {
    ReportFormat.TEXT: PlainTextRenderer(),
    ReportFormat.JSON: JSONRenderer(),
}


def register_renderer(fmt: ReportFormat | str, renderer: Renderer) -> None:
    """Make ``renderer`` the one used for ``fmt`` (replacing any previous one)."""
    _RENDERERS[ReportFormat(fmt)] = renderer


def available_formats() -> list[ReportFormat]:
    """Formats that have a renderer, in declaration order."""
    return [fmt for fmt in ReportFormat if fmt in _RENDERERS]


def get_renderer(fmt: ReportFormat | str) -> Renderer:
    """The renderer for ``fmt``.

    Raises:
        ReportFormatUnavailableError: if no renderer is registered for it.
        ValueError: if ``fmt`` is not a :class:`ReportFormat`.
    """
    key = ReportFormat(fmt)
    try:
        return _RENDERERS[key]
    except KeyError:
        planned = _PLANNED.get(key)
        hint = f"; it arrives with {planned}" if planned else ""
        available = ", ".join(f.value for f in available_formats())
        raise ReportFormatUnavailableError(
            f"report format {key.value!r} is not available yet{hint} (available: {available})"
        ) from None


def render(
    report: ExperimentReport | ComparisonReport, fmt: ReportFormat | str = ReportFormat.TEXT
) -> str:
    """Render an experiment or comparison report in ``fmt`` (default plain text)."""
    renderer = get_renderer(fmt)
    if isinstance(report, ComparisonReport):
        return renderer.comparison(report)
    return renderer.experiment(report)
