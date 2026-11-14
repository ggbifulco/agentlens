"""Judge prompt templates: packaged text files under ``agentlens/evals/prompts/``.

Templates use :class:`string.Template` placeholders (``$criteria``, ``${output}``):
no template engine dependency, and the ``{``/``}`` of the JSON answer format need
no escaping. Values are substituted once, so a ``$`` inside an agent's output is
left alone. Packaged templates:

* ``judge_system`` — the system prompt shared by every judge;
* ``rubric`` — ``$criteria``, ``$input``, ``$output``, ``$reference``;
* ``pass_fail`` — the same placeholders;
* ``reference`` — ``$expected``: the block substituted for ``$reference`` when the
  case has an expected output (empty otherwise).
"""

from __future__ import annotations

import string
from collections.abc import Iterable, Mapping
from functools import cache
from importlib import resources

from agentlens.evals.errors import EvaluatorConfigError

__all__ = ["check_template", "load_template", "render_template"]

_PACKAGE = "agentlens.evals"
_DIRECTORY = "prompts"


@cache
def load_template(name: str) -> str:
    """The text of the packaged template ``prompts/<name>.txt`` (cached).

    Raises:
        FileNotFoundError: if there is no such template.
    """
    if not name or "/" in name or "\\" in name or name.startswith("."):
        raise FileNotFoundError(f"invalid judge prompt template name {name!r}")
    path = resources.files(_PACKAGE).joinpath(_DIRECTORY).joinpath(f"{name}.txt")
    if not path.is_file():
        raise FileNotFoundError(f"no judge prompt template {name!r} in {_PACKAGE}/{_DIRECTORY}/")
    return path.read_text(encoding="utf-8")


def check_template(template: object, placeholders: Iterable[str]) -> str:
    """Validate a custom template: a non-empty string using only known ``placeholders``.

    Raises:
        EvaluatorConfigError: for a non-string, an empty template, invalid ``$``
            syntax or an unknown placeholder.
    """
    if not isinstance(template, str) or not template.strip():
        raise EvaluatorConfigError(f"template must be a non-empty string, got {template!r}")
    parsed = string.Template(template)
    if not parsed.is_valid():
        raise EvaluatorConfigError(
            "template has an invalid '$' placeholder (write '$$' for a literal '$')"
        )
    allowed = set(placeholders)
    unknown = sorted(set(parsed.get_identifiers()) - allowed)
    if unknown:
        raise EvaluatorConfigError(
            f"template uses unknown placeholder(s) {unknown}; available: {sorted(allowed)}"
        )
    return template


def render_template(template: str, values: Mapping[str, str]) -> str:
    """Substitute ``values`` into ``template`` (:meth:`string.Template.substitute`).

    Raises:
        KeyError: if the template uses a placeholder missing from ``values``.
        ValueError: for invalid ``$`` syntax.
    """
    return string.Template(template).substitute(values)
