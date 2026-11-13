"""Tests for output selection and text helpers shared by the built-ins."""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from agentlens.evals import (
    EvaluatorConfigError,
    OutputPathError,
    output_text,
    output_value,
    to_text,
)
from agentlens.evals.builtin._common import check_bool, check_path, clip, normalise_text, preview
from tests.evals.builtin.conftest import make_trace


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, None),
        ("plain", "plain"),
        ("", ""),
        (42, "42"),
        (1.5, "1.5"),
        (True, "true"),
        ([1, "a"], '[1, "a"]'),
        ({"b": 1, "a": "é"}, '{"a": "é", "b": 1}'),
    ],
)
def test_to_text(value: JsonValue, expected: str | None) -> None:
    assert to_text(value) == expected


def test_output_text_uses_the_whole_output_or_a_path() -> None:
    trace = make_trace({"answer": "Paris", "meta": {"n": 2}})
    assert output_text(trace) == '{"answer": "Paris", "meta": {"n": 2}}'
    assert output_text(trace, path="answer") == "Paris"
    assert output_text(trace, path="meta") == '{"n": 2}'
    assert output_text(make_trace(None)) is None


@pytest.mark.parametrize(
    ("output", "path", "expected"),
    [
        ({"a": {"b": [10, {"c": "x"}]}}, "a.b.1.c", "x"),
        ({"a": {"b": [10, 20]}}, "a.b.-1", 20),
        ({"a.b": 1}, ["a.b"], 1),
        ([{"t": "first"}], [0, "t"], "first"),
        ('{"answer": "from json text"}', "answer", "from json text"),
        ({"n": None}, "n", None),
    ],
)
def test_output_value_paths(
    output: JsonValue, path: str | list[str | int], expected: JsonValue
) -> None:
    assert output_value(make_trace(output), path=path) == expected


@pytest.mark.parametrize(
    ("output", "path", "message"),
    [
        ({"a": 1}, "b", "output has no 'b'"),
        ({"a": {"b": 1}}, "a.c", "output has no 'a.c'"),
        ({"a": [1]}, "a.3", "a list of 1 items"),
        ({"a": [1]}, "a.x", "output has no 'a.x'"),
        ({"a": 1}, "a.b", "int reached"),
        (None, "a", "NoneType reached"),
        ("not json", "a", "non-JSON string"),
    ],
)
def test_output_value_missing_path(output: JsonValue, path: str, message: str) -> None:
    with pytest.raises(OutputPathError, match=message):
        output_value(make_trace(output), path=path)


@pytest.mark.parametrize("path", ["", "a..b", [], ["a", ""], [1.5], [True], 3, {"a": 1}])
def test_check_path_rejects_malformed_paths(path: object) -> None:
    with pytest.raises(EvaluatorConfigError, match="path"):
        check_path(path)


def test_check_path_splits_dotted_strings() -> None:
    assert check_path(None) is None
    assert check_path("a.0.b") == ("a", "0", "b")
    assert check_path(["a.b", 0]) == ("a.b", 0)


@pytest.mark.parametrize(
    ("text", "options", "expected"),
    [
        ("Café", {}, "Café"),
        ("  a  b \n", {}, "  a  b \n"),
        ("  a  b \n", {"strip": True}, "a  b"),
        ("  a \t\n b ", {"normalise_whitespace": True}, " a b "),
        ("  a \t\n b ", {"normalise_whitespace": True, "strip": True}, "a b"),
        ("Straße", {"case_sensitive": False}, "strasse"),
    ],
)
def test_normalise_text(text: str, options: dict[str, bool], expected: str) -> None:
    assert normalise_text(text, **options) == expected


def test_preview_and_clip_shorten_long_text() -> None:
    assert preview("short") == "'short'"
    assert preview("x" * 200, 10) == "'xxxxxxxxxx'... (200 chars)"
    assert clip("abc", 5) == "abc"
    assert clip("abcdef", 3) == "abc…"


def test_check_bool() -> None:
    assert check_bool("flag", True) is True
    with pytest.raises(EvaluatorConfigError, match="flag must be true or false"):
        check_bool("flag", "no")
