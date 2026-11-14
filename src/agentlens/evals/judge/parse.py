"""Robust parsing of a judge's JSON reply.

Models asked for JSON still wrap it in Markdown fences, prefix it with "Sure, here
is my evaluation:", or write ``"score": "4"``. :func:`extract_json_object` finds
the first JSON object in the reply; :func:`parse_score` / :func:`parse_verdict` /
:func:`parse_reasoning` read and validate individual fields. Every failure raises
:class:`JudgeParseError`, which judges turn into a retry and then a failed result.
"""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping, Sequence

from pydantic import JsonValue

__all__ = [
    "JudgeParseError",
    "extract_json_object",
    "parse_reasoning",
    "parse_score",
    "parse_verdict",
]

_FENCE = re.compile(r"```[ \t]*(?:json|JSON)?[ \t]*\n?(.*?)```", re.DOTALL)
_PASS_WORDS = frozenset({"pass", "passed", "yes", "true", "y", "ok", "correct", "success"})
_FAIL_WORDS = frozenset({"fail", "failed", "no", "false", "n", "incorrect", "failure"})
_REASONING_KEYS = ("reasoning", "reason", "explanation", "rationale")


class JudgeParseError(ValueError):
    """The judge's reply is not the JSON object it was asked for."""


def _candidates(text: str) -> list[str]:
    stripped = text.strip()
    fenced = [m.group(1).strip() for m in _FENCE.finditer(stripped)]
    return [*fenced, stripped]


def extract_json_object(text: str | None) -> dict[str, JsonValue]:
    """The first JSON object in ``text``.

    Tries, in order: the content of each Markdown code fence, the whole reply, and
    finally every ``{`` in the reply as the start of an object
    (:meth:`json.JSONDecoder.raw_decode`), so prose before or after the object
    is ignored.

    Raises:
        JudgeParseError: if the reply is empty or holds no JSON object.
    """
    if text is None or not text.strip():
        raise JudgeParseError("the judge returned an empty reply")
    decoder = json.JSONDecoder()
    for candidate in _candidates(text):
        try:
            value = json.loads(candidate)
        except ValueError:
            pass
        else:
            if isinstance(value, dict):
                return value
        start = candidate.find("{")
        while start != -1:
            try:
                obj, _ = decoder.raw_decode(candidate, start)
            except ValueError:
                pass
            else:
                return dict(obj)  # decoding from a "{" always yields an object
            start = candidate.find("{", start + 1)
    raise JudgeParseError("no JSON object found in the judge's reply")


def _field(data: Mapping[str, JsonValue], keys: Sequence[str]) -> tuple[str, JsonValue] | None:
    lowered = {k.lower(): k for k in data}
    for key in keys:
        if key in lowered:
            original = lowered[key]
            return original, data[original]
    return None


def parse_score(
    data: Mapping[str, JsonValue],
    *,
    low: int = 1,
    high: int = 5,
    keys: Sequence[str] = ("score", "rating"),
) -> float:
    """Read a ``low``..``high`` score from ``data`` (key ``score`` or ``rating``).

    Accepts numbers and numeric strings (``"4"``, ``" 4.5 "``, ``"4/5"``); keys are
    matched case-insensitively. Fractional scores inside the range are kept.

    Raises:
        JudgeParseError: if the key is missing, the value is not a finite number
            (booleans included) or it is outside ``[low, high]``.
    """
    found = _field(data, keys)
    if found is None:
        raise JudgeParseError(f"the judge's JSON has no {keys[0]!r} field")
    key, raw = found
    value: float
    if isinstance(raw, bool):
        raise JudgeParseError(f"{key!r} must be a number, got {raw!r}")
    if isinstance(raw, int | float):
        value = float(raw)
    elif isinstance(raw, str):
        text = raw.strip()
        if "/" in text:  # "4/5"
            text = text.split("/", 1)[0].strip()
        try:
            value = float(text)
        except ValueError:
            raise JudgeParseError(f"{key!r} must be a number, got {raw!r}") from None
    else:
        raise JudgeParseError(f"{key!r} must be a number, got {raw!r}")
    if not math.isfinite(value) or not low <= value <= high:
        raise JudgeParseError(f"{key!r} must be between {low} and {high}, got {raw!r}")
    return value


def parse_verdict(
    data: Mapping[str, JsonValue],
    *,
    keys: Sequence[str] = ("verdict", "passed", "pass", "result"),
) -> bool:
    """Read a pass/fail verdict: ``"pass"``/``"fail"`` (also yes/no, true/false) or a boolean.

    Raises:
        JudgeParseError: if no verdict key is present or its value is not recognised.
    """
    found = _field(data, keys)
    if found is None:
        raise JudgeParseError(f"the judge's JSON has no {keys[0]!r} field")
    key, raw = found
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        word = raw.strip().strip(".!").lower()
        if word in _PASS_WORDS:
            return True
        if word in _FAIL_WORDS:
            return False
    raise JudgeParseError(f"{key!r} must be 'pass' or 'fail', got {raw!r}")


def parse_reasoning(data: Mapping[str, JsonValue]) -> str | None:
    """The judge's explanation (``reasoning``, ``reason``, ``explanation`` or ``rationale``).

    ``None`` when absent or empty; a non-string value is returned as JSON text.
    """
    found = _field(data, _REASONING_KEYS)
    if found is None or found[1] is None:
        return None
    raw = found[1]
    text = raw if isinstance(raw, str) else json.dumps(raw, ensure_ascii=False)
    return text.strip() or None
