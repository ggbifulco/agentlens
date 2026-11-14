"""Errors raised while loading or validating a suite.

Every problem found in a suite file is collected as a :class:`SuiteIssue`
(``file:line:column: path: message``) and raised together in one
:class:`SuiteError`, so a broken suite is fixed in one pass rather than one
error at a time.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

__all__ = ["AgentImportError", "Loc", "SuiteError", "SuiteIssue", "format_loc"]

Loc = tuple[str | int, ...]
"""A location inside the suite document: ``("cases", 2, "evaluators", 0)``."""


def format_loc(loc: Loc) -> str:
    """Render a location as ``cases[2].evaluators[0]`` (empty for the document root)."""
    out = ""
    for part in loc:
        if isinstance(part, int):
            out += f"[{part}]"
        else:
            out += f".{part}" if out else part
    return out


@dataclass(frozen=True)
class SuiteIssue:
    """One problem in a suite, with its position in the source file when known."""

    message: str
    loc: Loc = ()
    """Where in the suite document (``cases[2].id``); empty for the whole suite."""
    file: str | None = None
    """The file the problem is in (the suite or an external dataset), if any."""
    line: int | None = None
    """1-based line in :attr:`file`."""
    column: int | None = None
    """1-based column in :attr:`file`."""

    @property
    def path(self) -> str:
        """:attr:`loc` rendered as ``cases[2].evaluators[0]``."""
        return format_loc(self.loc)

    def __str__(self) -> str:
        parts: list[str] = []
        if self.file is not None:
            where = self.file
            if self.line is not None:
                where += f":{self.line}"
                if self.column is not None:
                    where += f":{self.column}"
            parts.append(where)
        if self.loc:
            parts.append(self.path)
        parts.append(self.message)
        return ": ".join(parts)


class SuiteError(Exception):
    """A suite could not be loaded: all problems found are in :attr:`issues`.

    ``str(error)`` lists every issue on its own line, e.g.::

        invalid suite suites/support.yaml (2 problems):
          suites/support.yaml:12:9: cases[2].evaluators[0]: unknown evaluator 'foo'
          suites/support.yaml:20:5: cases[3].id: duplicate case id 'refund' (first: cases[0])
    """

    def __init__(self, message: str, issues: Sequence[SuiteIssue] = ()) -> None:
        self.issues: list[SuiteIssue] = list(issues)
        self.summary = message
        if self.issues:
            lines = "\n".join(f"  {issue}" for issue in self.issues)
            n = len(self.issues)
            message = f"{message} ({n} problem{'s' if n != 1 else ''}):\n{lines}"
        super().__init__(message)


class AgentImportError(SuiteError):
    """The suite's ``agent`` (or an ``imports`` module) could not be imported or is not callable."""

    def __init__(self, message: str) -> None:
        super().__init__(message)
