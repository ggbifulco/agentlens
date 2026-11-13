"""Deterministic checks on the agent's final output: ``exact_match``, ``contains``,
``regex`` and ``json_schema`` (FR-E1, ``docs/design.md`` §5).

All four look at ``trace.output`` — or the part of it named by a ``path``
parameter (``"answer"``, ``"items.0.text"``; see
:func:`~agentlens.evals.output_value`). Text checks see a non-string output as
canonical JSON (:func:`~agentlens.evals.to_text`). Each returns a score in
``[0, 1]``, a human-readable ``reason`` and ``details`` for reports.

Bad parameters (an invalid regex or JSON schema, an unknown mode) raise
:class:`~agentlens.evals.EvaluatorConfigError` when the evaluator is built, so a
broken suite fails at load time. A missing expectation that is only known at
evaluation time (no ``value`` parameter and no ``case.expected``) raises
:class:`~agentlens.evals.EvaluatorConfigError` from ``evaluate``, which
:func:`~agentlens.evals.run_evaluator` reports as an evaluator error rather than
an agent failure.
"""

from __future__ import annotations

import contextlib
import json
import re
from collections.abc import Mapping
from enum import Enum
from typing import Any, Final, Literal

import jsonschema
import jsonschema.exceptions
import jsonschema.protocols
import jsonschema.validators
from pydantic import JsonValue

from agentlens.evals.builtin._common import (
    OutputPath,
    OutputPathError,
    check_bool,
    check_path,
    clip,
    normalise_text,
    output_value,
    preview,
    to_text,
)
from agentlens.evals.errors import EvaluatorConfigError
from agentlens.schema import Case, EvalResult, Trace

__all__ = ["UNSET", "Contains", "ExactMatch", "JsonSchema", "Regex"]


class _Unset(Enum):
    UNSET = "UNSET"

    def __repr__(self) -> str:
        return "UNSET"


UNSET: Final = _Unset.UNSET
"""Default of :class:`ExactMatch`'s ``value``: "not given, use ``case.expected``"
(distinct from ``None``, which expects a ``null`` output)."""


def _fail(name: str, reason: str, **details: JsonValue) -> EvalResult:
    return EvalResult(evaluator=name, score=0.0, passed=False, reason=reason, details=details)


def _choice(param: str, value: object, allowed: tuple[str, ...]) -> str:
    if not isinstance(value, str) or value not in allowed:
        options = ", ".join(repr(a) for a in allowed)
        raise EvaluatorConfigError(f"{param} must be one of {options}, got {value!r}")
    return value


def _params_repr(cls_name: str, params: Mapping[str, Any]) -> str:
    shown = ", ".join(f"{k}={v!r}" for k, v in params.items())
    return f"{cls_name}({shown})"


def _json_text(value: JsonValue) -> str:
    text = to_text(value)
    return "null" if text is None else text


# -- exact_match ---------------------------------------------------------------


class ExactMatch:
    """``exact_match``: the output equals an expected value.

    The expected value is the ``value`` parameter when given, else
    ``case.expected`` (``{exact_match: "Paris"}`` overrides the case). With
    neither, ``evaluate`` raises :class:`~agentlens.evals.EvaluatorConfigError`.

    * **Text comparison** when the expected value is a string: a non-string
      output is compared as canonical JSON text. Both sides are NFC-normalised,
      then ``strip`` (default on), ``normalise_whitespace`` and
      ``case_sensitive`` apply.
    * **Structural comparison** when the expected value is anything else (dict,
      list, number, bool, ``null``): a string output is parsed as JSON first, so
      ``'{"a": 1}'`` matches ``{a: 1}``. Key order does not matter, ``1 == 1.0``,
      but ``true != 1``; the text options apply to every string inside.

    Score ``1`` on a match, else ``0``.

    Args:
        value: Expected value; defaults to ``case.expected``. Pass ``None``
            explicitly to expect a ``null`` output.
        case_sensitive: Compare strings case-sensitively (``False`` uses casefold).
        strip: Ignore leading/trailing whitespace.
        normalise_whitespace: Treat any run of whitespace as one space.
        path: Compare only this part of the output (see :func:`~agentlens.evals.output_value`).
    """

    name: str = "exact_match"

    def __init__(
        self,
        value: JsonValue | _Unset = UNSET,
        *,
        case_sensitive: bool = True,
        strip: bool = True,
        normalise_whitespace: bool = False,
        path: OutputPath | None = None,
    ) -> None:
        self.value = value
        self.case_sensitive = check_bool("case_sensitive", case_sensitive)
        self.strip = check_bool("strip", strip)
        self.normalise_whitespace = check_bool("normalise_whitespace", normalise_whitespace)
        check_path(path)
        self.path = path

    def _text(self, text: str) -> str:
        return normalise_text(
            text,
            case_sensitive=self.case_sensitive,
            strip=self.strip,
            normalise_whitespace=self.normalise_whitespace,
        )

    def _canon(self, value: JsonValue) -> object:
        # Tagged so that JSON types stay distinct (Python says True == 1).
        if isinstance(value, bool):
            return ("bool", value)
        if isinstance(value, int | float):
            return ("number", value)
        if isinstance(value, str):
            return ("string", self._text(value))
        if isinstance(value, list):
            return ("array", [self._canon(v) for v in value])
        if isinstance(value, dict):
            return ("object", {k: self._canon(v) for k, v in value.items()})
        return ("null", None)

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Compare the output with the expected value (see the class docstring)."""
        if self.value is not UNSET:
            expected: JsonValue = self.value
            source = "value"
        elif case.expected is not None:
            expected = case.expected
            source = "case.expected"
        else:
            raise EvaluatorConfigError(
                f"exact_match has nothing to compare with in case {case.id!r}: "
                "set the 'value' parameter or the case's 'expected'"
            )
        try:
            actual = output_value(trace, path=self.path)
        except OutputPathError as exc:
            return _fail(self.name, str(exc), source=source)

        if isinstance(expected, str):
            mode = "text"
            actual_text = to_text(actual)
            matched = actual_text is not None and self._text(actual_text) == self._text(expected)
        else:
            mode = "json"
            if isinstance(actual, str):
                # A plain (non-JSON) string never equals a non-string expected value.
                with contextlib.suppress(json.JSONDecodeError):
                    actual = json.loads(actual)
            matched = self._canon(actual) == self._canon(expected)

        expected_text = _json_text(expected)
        actual_text = _json_text(actual)
        details: dict[str, JsonValue] = {
            "mode": mode,
            "source": source,
            "expected": clip(expected_text),
            "actual": clip(actual_text),
        }
        if matched:
            reason = f"output matches the expected value ({source})"
        elif actual is None:
            reason = f"no output (None); expected {preview(expected_text)}"
        else:
            reason = f"output {preview(actual_text)} != expected {preview(expected_text)}"
        return EvalResult(
            evaluator=self.name,
            score=1.0 if matched else 0.0,
            passed=matched,
            reason=reason,
            details=details,
        )

    def __repr__(self) -> str:
        return _params_repr(
            "ExactMatch",
            {
                "value": self.value,
                "case_sensitive": self.case_sensitive,
                "strip": self.strip,
                "normalise_whitespace": self.normalise_whitespace,
                "path": self.path,
            },
        )


# -- contains ------------------------------------------------------------------


def _check_needles(param: str, value: object) -> list[str]:
    needles = [value] if isinstance(value, str) else value
    if (
        not isinstance(needles, list)
        or not needles
        or not all(isinstance(n, str) and n for n in needles)
    ):
        raise EvaluatorConfigError(
            f"{param} must be a non-empty string or a non-empty list of non-empty strings, "
            f"got {value!r}"
        )
    return list(needles)


class Contains:
    """``contains``: the output text contains one or more substrings.

    ``{contains: "refund"}`` or ``{contains: {text: [refund, "order 1234"], mode: any}}``.
    Without ``text`` the substrings come from ``case.expected`` (a string or a list
    of strings). The output is searched as text (non-strings as canonical JSON).

    The score is the fraction of substrings found. ``mode="all"`` (default) passes
    only when every substring is found; ``mode="any"`` passes when at least one is.
    A ``None`` output contains nothing.

    Args:
        text: Substring or list of substrings; defaults to ``case.expected``.
        mode: ``"all"`` or ``"any"``.
        case_sensitive: ``False`` compares casefolded text.
        normalise_whitespace: Treat any run of whitespace (in both) as one space.
        path: Search only this part of the output.
    """

    name: str = "contains"

    def __init__(
        self,
        text: str | list[str] | None = None,
        *,
        mode: Literal["all", "any"] = "all",
        case_sensitive: bool = True,
        normalise_whitespace: bool = False,
        path: OutputPath | None = None,
    ) -> None:
        self.text = None if text is None else _check_needles("text", text)
        self.mode = _choice("mode", mode, ("all", "any"))
        self.case_sensitive = check_bool("case_sensitive", case_sensitive)
        self.normalise_whitespace = check_bool("normalise_whitespace", normalise_whitespace)
        check_path(path)
        self.path = path

    def _norm(self, text: str) -> str:
        return normalise_text(
            text,
            case_sensitive=self.case_sensitive,
            normalise_whitespace=self.normalise_whitespace,
        )

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Look for each substring in the output text."""
        if self.text is not None:
            needles = self.text
        elif case.expected is not None:
            needles = _check_needles(f"case {case.id!r} expected (used by contains)", case.expected)
        else:
            raise EvaluatorConfigError(
                f"contains has no substrings for case {case.id!r}: "
                "set the 'text' parameter or the case's 'expected'"
            )
        try:
            haystack = to_text(output_value(trace, path=self.path))
        except OutputPathError as exc:
            return _fail(self.name, str(exc), mode=self.mode, matched=[], missing=list(needles))

        normalised = self._norm(haystack) if haystack is not None else ""
        matched = [n for n in needles if haystack is not None and self._norm(n) in normalised]
        missing = [n for n in needles if n not in matched]
        score = len(matched) / len(needles)
        passed = not missing if self.mode == "all" else bool(matched)
        found = f"{len(matched)}/{len(needles)} found"
        if haystack is None:
            reason = f"no output (None) to search; {found}"
        elif passed and self.mode == "all":
            reason = f"output contains {_quote_all(matched)}"
        elif passed:
            reason = f"output contains {_quote_all(matched)} ({found}, mode any)"
        elif self.mode == "all":
            reason = f"output is missing {_quote_all(missing)} ({found})"
        else:
            reason = f"output contains none of {_quote_all(missing)}"
        details: dict[str, JsonValue] = {
            "mode": self.mode,
            "matched": list(matched),
            "missing": list(missing),
        }
        return EvalResult(
            evaluator=self.name, score=score, passed=passed, reason=reason, details=details
        )

    def __repr__(self) -> str:
        return _params_repr(
            "Contains",
            {
                "text": self.text,
                "mode": self.mode,
                "case_sensitive": self.case_sensitive,
                "normalise_whitespace": self.normalise_whitespace,
                "path": self.path,
            },
        )


def _quote_all(items: list[str], limit: int = 5) -> str:
    shown = ", ".join(preview(i, 60) for i in items[:limit])
    return shown + (f" and {len(items) - limit} more" if len(items) > limit else "")


_MODE_VERBS = {
    "search": "contain a match for",
    "match": "start with a match for",
    "fullmatch": "fully match",
}

# -- regex ---------------------------------------------------------------------


class Regex:
    """``regex``: the output text matches a regular expression (Python :mod:`re` syntax).

    ``{regex: "order \\d+"}`` or ``{regex: {pattern: "^Refund", ignore_case: true}}``.
    The pattern is compiled when the evaluator is built, so an invalid one is an
    :class:`~agentlens.evals.EvaluatorConfigError` at suite load. A ``None`` output
    never matches.

    Args:
        pattern: The regular expression.
        mode: ``"search"`` (anywhere, default), ``"match"`` (at the start) or
            ``"fullmatch"`` (the whole text).
        ignore_case: :data:`re.IGNORECASE`.
        multiline: :data:`re.MULTILINE` (``^``/``$`` match at every line).
        dotall: :data:`re.DOTALL` (``.`` matches newlines).
        path: Match only this part of the output.
    """

    name: str = "regex"

    def __init__(
        self,
        pattern: str,
        *,
        mode: Literal["search", "match", "fullmatch"] = "search",
        ignore_case: bool = False,
        multiline: bool = False,
        dotall: bool = False,
        path: OutputPath | None = None,
    ) -> None:
        if not isinstance(pattern, str) or not pattern:
            raise EvaluatorConfigError(f"pattern must be a non-empty string, got {pattern!r}")
        self.pattern = pattern
        self.mode = _choice("mode", mode, ("search", "match", "fullmatch"))
        self.ignore_case = check_bool("ignore_case", ignore_case)
        self.multiline = check_bool("multiline", multiline)
        self.dotall = check_bool("dotall", dotall)
        check_path(path)
        self.path = path
        flags = re.NOFLAG
        if self.ignore_case:
            flags |= re.IGNORECASE
        if self.multiline:
            flags |= re.MULTILINE
        if self.dotall:
            flags |= re.DOTALL
        try:
            self.compiled = re.compile(pattern, flags)
        except re.error as exc:
            raise EvaluatorConfigError(f"invalid regex {pattern!r}: {exc}") from exc

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Apply the pattern to the output text."""
        base: dict[str, JsonValue] = {"pattern": self.pattern, "mode": self.mode}
        try:
            text = to_text(output_value(trace, path=self.path))
        except OutputPathError as exc:
            return _fail(self.name, str(exc), **base)
        if text is None:
            return _fail(self.name, f"no output (None) to match against /{self.pattern}/", **base)
        found = getattr(self.compiled, self.mode)(text)
        if found is None:
            return _fail(
                self.name,
                f"output {preview(text)} does not {_MODE_VERBS[self.mode]} /{self.pattern}/",
                **base,
            )
        assert isinstance(found, re.Match)
        groups: list[JsonValue] = list(found.groups())
        named: dict[str, JsonValue] = dict(found.groupdict())
        details: dict[str, JsonValue] = {
            **base,
            "match": clip(found.group(0)),
            "span": [found.start(), found.end()],
            "groups": groups,
            "named_groups": named,
        }
        return EvalResult(
            evaluator=self.name,
            score=1.0,
            passed=True,
            reason=f"output matches /{self.pattern}/ at {found.start()}: "
            f"{preview(found.group(0), 60)}",
            details=details,
        )

    def __repr__(self) -> str:
        return _params_repr(
            "Regex",
            {
                "pattern": self.pattern,
                "mode": self.mode,
                "ignore_case": self.ignore_case,
                "multiline": self.multiline,
                "dotall": self.dotall,
                "path": self.path,
            },
        )


# -- json_schema ---------------------------------------------------------------

_FENCE = re.compile(r"^\s*```[A-Za-z0-9_-]*[ \t]*\n(.*?)\n?```\s*$", re.DOTALL)


def _validator(schema: object, where: str) -> jsonschema.protocols.Validator:
    if not isinstance(schema, dict):
        raise EvaluatorConfigError(
            f"{where} must be a JSON schema (a mapping), got {type(schema).__name__}"
        )
    cls = jsonschema.validators.validator_for(schema)
    try:
        cls.check_schema(schema)
    except jsonschema.exceptions.SchemaError as exc:
        raise EvaluatorConfigError(f"{where} is not a valid JSON schema: {exc.message}") from exc
    return cls(schema)


def _parse_json_text(text: str) -> JsonValue:
    """``json.loads`` that also accepts one surrounding Markdown code fence (```json ... ```)."""
    fenced = _FENCE.match(text)
    loaded: JsonValue = json.loads(fenced.group(1) if fenced else text)
    return loaded


class JsonSchema:
    """``json_schema``: the output is valid against a JSON Schema.

    The schema is the ``schema`` parameter, else ``case.expected["schema"]``
    (``{json_schema: {schema: {type: object, required: [id]}}}``). The draft is
    taken from ``$schema`` (default 2020-12); the schema itself is checked when
    the evaluator is built (invalid → :class:`~agentlens.evals.EvaluatorConfigError`).

    A string output is parsed as JSON first (one surrounding Markdown code fence
    is tolerated); unparsable text fails with the parse error. Set
    ``parse_json=False`` to validate a string output as a JSON string instead.
    A ``None`` output is validated as ``null``. Score ``1`` when valid, else ``0``;
    the reason lists the first ``max_errors`` errors (sorted by location) and
    ``details["errors"]`` holds them as ``{path, message, validator}``.

    Args:
        schema: The JSON Schema; defaults to ``case.expected["schema"]``.
        parse_json: Parse a string output as JSON before validating.
        max_errors: How many validation errors to report (``>= 1``).
        path: Validate only this part of the output.
    """

    name: str = "json_schema"

    def __init__(
        self,
        schema: dict[str, JsonValue] | None = None,
        *,
        parse_json: bool = True,
        max_errors: int = 5,
        path: OutputPath | None = None,
    ) -> None:
        self.schema = schema
        self._validator = None if schema is None else _validator(schema, "schema")
        self.parse_json = check_bool("parse_json", parse_json)
        if isinstance(max_errors, bool) or not isinstance(max_errors, int) or max_errors < 1:
            raise EvaluatorConfigError(f"max_errors must be an integer >= 1, got {max_errors!r}")
        self.max_errors = max_errors
        check_path(path)
        self.path = path

    def _case_validator(self, case: Case) -> jsonschema.protocols.Validator:
        expected = case.expected
        if not isinstance(expected, dict) or "schema" not in expected:
            raise EvaluatorConfigError(
                f"json_schema has no schema for case {case.id!r}: set the 'schema' "
                "parameter or the case's expected.schema"
            )
        return _validator(expected["schema"], f"case {case.id!r} expected.schema")

    def evaluate(self, trace: Trace, case: Case) -> EvalResult:
        """Validate the (parsed) output against the schema."""
        if self._validator is not None:
            validator, source = self._validator, "schema"
        else:
            validator, source = self._case_validator(case), "case.expected.schema"
        try:
            instance = output_value(trace, path=self.path)
        except OutputPathError as exc:
            return _fail(self.name, str(exc), source=source)
        if self.parse_json and isinstance(instance, str):
            raw = instance
            try:
                instance = _parse_json_text(raw)
            except json.JSONDecodeError as exc:
                return _fail(
                    self.name,
                    f"output is not valid JSON ({exc}): {preview(raw, 60)}",
                    source=source,
                )

        errors = sorted(validator.iter_errors(instance), key=lambda e: (e.json_path, e.message))
        shown: list[JsonValue] = [
            {"path": e.json_path, "message": clip(e.message, 300), "validator": str(e.validator)}
            for e in errors[: self.max_errors]
        ]
        details: dict[str, JsonValue] = {
            "source": source,
            "error_count": len(errors),
            "errors": shown,
        }
        if not errors:
            return EvalResult(
                evaluator=self.name,
                score=1.0,
                passed=True,
                reason="output matches the JSON schema",
                details=details,
            )
        listed = "; ".join(
            f"{e.json_path}: {clip(e.message, 160)}" for e in errors[: self.max_errors]
        )
        more = f" (+{len(errors) - self.max_errors} more)" if len(errors) > self.max_errors else ""
        noun = "error" if len(errors) == 1 else "errors"
        return _fail(
            self.name,
            f"output does not match the JSON schema, {len(errors)} {noun}: {listed}{more}",
            **details,
        )

    def __repr__(self) -> str:
        return _params_repr(
            "JsonSchema",
            {
                "schema": self.schema,
                "parse_json": self.parse_json,
                "max_errors": self.max_errors,
                "path": self.path,
            },
        )
