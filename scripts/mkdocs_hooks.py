"""MkDocs hook: tidy the reStructuredText left in docstrings for the API reference.

The docstrings are written to read well in ``help()`` and in an editor, so they use Sphinx roles
(``:class:`Trace```) and ``::`` before code blocks. mkdocstrings renders them as Markdown, which
leaves ``:class:`` and ``::`` visible; this hook removes them from the rendered HTML
(``~pkg.mod.Name`` becomes ``Name``, as Sphinx would show it). Wired in ``mkdocs.yml``.
"""

from __future__ import annotations

import re
from typing import Any

_ROLE = re.compile(r":(?:func|class|meth|mod|attr|data|exc|obj|const):(?=<code>)")
_TILDE = re.compile(r"<code>~(?:\w+\.)*(\w+)</code>")
_DOUBLE_COLON = re.compile(r"::(</p>)")


def tidy(html: str) -> str:
    """Drop Sphinx roles and ``::`` markers from rendered API-reference HTML."""
    html = _ROLE.sub("", html)
    html = _TILDE.sub(r"<code>\1</code>", html)
    return _DOUBLE_COLON.sub(r":\1", html)


def on_page_content(html: str, page: Any, **_: Any) -> str:
    """Only API reference pages are rewritten; prose pages are left alone."""
    if page.file.src_uri.startswith("api/"):
        return tidy(html)
    return html
