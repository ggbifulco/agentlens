"""Helpers shared by the built-in evaluators: which part of the output to check, as what text."""

from __future__ import annotations

import json
import re
import unicodedata
from collections.abc import Sequence
from typing import TypeAlias

from pydantic import JsonValue

from agentlens.evals.errors import EvaluatorConfigError
from agentlens.schema import Trace

__all__ = [
    "OutputPath",
    "OutputPathError",
    "check_bool",
    "check_path",
    "clip",
    "normalise_text",
    "output_text",
    "output_value",
    "preview",
    "to_text",
]

OutputPath: TypeAlias = str | Sequence[str | int]
"""Where to look inside a structured output: ``"answer"``, ``"items.0.text"`` or
``["weird.key", 0]`` (the list form for keys that contain dots)."""

PREVIEW_CHARS = 120
"""Longest output/expected snippet quoted in a ``reason`` (details keep more)."""

DETAIL_CHARS = 2000
"""Longest output/expected text stored in ``details``."""

_WHITESPACE = re.compile(r"\s+")


class OutputPathError(LookupError):
    """The requested ``path`` does not exist in the output (a failed check, not a config error)."""


def check_path(path: object) -> tuple[str | int, ...] | None:
    """Validate a ``path`` parameter and split it into segments (``None`` = whole output).

    Raises:
        EvaluatorConfigError: for an empty path or segments that are not ``str``/``int``.
    """
    if path is None:
        return None
    if isinstance(path, str):
        segments: list[str | int] = list(path.split("."))
    elif isinstance(path, Sequence) and all(
        isinstance(s, str | int) and not isinstance(s, bool) for s in path
    ):
        segments = list(path)
    else:
        raise EvaluatorConfigError(
            f"path must be a dotted string or a list of keys/indexes, got {path!r}"
        )
    if not segments or any(s == "" for s in segments):
        raise EvaluatorConfigError(f"path must not be empty or contain empty keys, got {path!r}")
    return tuple(segments)


def _segment_text(segments: Sequence[str | int]) -> str:
    return ".".join(str(s) for s in segments)


def output_value(trace: Trace, *, path: OutputPath | None = None) -> JsonValue:
    """The value an output evaluator checks: ``trace.output`` or the part ``path`` points to.

    With a ``path``, a string output is first parsed as JSON (agents often return
    JSON text). Dict keys are looked up as strings; on lists a segment must be an
    index (``"items.0"``; negative indexes count from the end).

    Raises:
        OutputPathError: if the path does not exist in the output.
        EvaluatorConfigError: if ``path`` itself is malformed.
    """
    segments = check_path(path)
    value: JsonValue = trace.output
    if segments is None:
        return value
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError:
            raise OutputPathError(
                f"output is a non-JSON string, it has no {_segment_text(segments)!r}"
            ) from None
    for depth, segment in enumerate(segments):
        where = _segment_text(segments[: depth + 1])
        if isinstance(value, dict):
            key = str(segment)
            if key not in value:
                raise OutputPathError(f"output has no {where!r}")
            value = value[key]
        elif isinstance(value, list):
            size = len(value)
            try:
                value = value[int(segment)]
            except (ValueError, IndexError):
                raise OutputPathError(f"output has no {where!r} (a list of {size} items)") from None
        else:
            raise OutputPathError(f"output has no {where!r} ({type(value).__name__} reached)")
    return value


def to_text(value: JsonValue) -> str | None:
    """Text form of an output value: a string as is, ``None`` as ``None``, anything
    else (dict, list, number, bool) as canonical JSON (sorted keys, ``", "``/``": "``
    separators, non-ASCII kept), so ``{"b": 1, "a": "é"}`` reads ``{"a": "é", "b": 1}``.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    return json.dumps(value, sort_keys=True, ensure_ascii=False)


def output_text(trace: Trace, *, path: OutputPath | None = None) -> str | None:
    """The text that ``contains``/``regex`` search: :func:`to_text` of :func:`output_value`.

    ``None`` when the (selected) output is ``None``.

    Raises:
        OutputPathError: if ``path`` does not exist in the output.
    """
    return to_text(output_value(trace, path=path))


def normalise_text(
    text: str,
    *,
    case_sensitive: bool = True,
    strip: bool = False,
    normalise_whitespace: bool = False,
) -> str:
    """Unicode NFC normalisation plus the optional text-comparison relaxations.

    NFC is always applied, so a composed ``"é"`` equals ``"e"`` + combining accent.
    ``normalise_whitespace`` collapses runs of whitespace into one space;
    ``case_sensitive=False`` uses :meth:`str.casefold` (``"Straße"`` matches ``"STRASSE"``).
    """
    text = unicodedata.normalize("NFC", text)
    if normalise_whitespace:
        text = _WHITESPACE.sub(" ", text)
    if strip:
        text = text.strip()
    if not case_sensitive:
        text = text.casefold()
    return text


def preview(text: str, limit: int = PREVIEW_CHARS) -> str:
    """``repr``-style quote of ``text``, shortened to about ``limit`` characters."""
    if len(text) <= limit:
        return repr(text)
    return repr(text[:limit]) + f"... ({len(text)} chars)"


def clip(text: str, limit: int = DETAIL_CHARS) -> str:
    """``text`` cut to ``limit`` characters (with a marker) for storage in ``details``."""
    return text if len(text) <= limit else text[:limit] + "…"


def check_bool(name: str, value: object) -> bool:
    """Reject non-booleans for flag parameters (YAML ``"no"`` strings, numbers...)."""
    if not isinstance(value, bool):
        raise EvaluatorConfigError(f"{name} must be true or false, got {value!r}")
    return value
