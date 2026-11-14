"""Parsing judge replies: JSON extraction and field validation."""

from __future__ import annotations

import pytest

from agentlens.evals.judge import (
    JudgeParseError,
    extract_json_object,
    parse_reasoning,
    parse_score,
    parse_verdict,
)


@pytest.mark.parametrize(
    "text",
    [
        '{"score": 4}',
        '  {"score": 4}\n',
        '```json\n{"score": 4}\n```',
        '```\n{"score": 4}\n```',
        'Here is my evaluation:\n```json\n{"score": 4}\n```\nHope it helps.',
        'Sure! {"score": 4} That is all.',
        'Considering {braces} first, then {"score": 4}',
        '[1, 2] then {"score": 4}',
    ],
)
def test_extract_json_object_finds_the_object(text: str) -> None:
    assert extract_json_object(text) == {"score": 4}


def test_extract_prefers_the_fenced_object() -> None:
    text = 'Example: {"score": 1}\n```json\n{"score": 5}\n```'
    assert extract_json_object(text) == {"score": 5}


def test_extract_nested_object() -> None:
    assert extract_json_object('x {"a": {"b": [1, {"c": 2}]}} y') == {"a": {"b": [1, {"c": 2}]}}


@pytest.mark.parametrize(
    "text",
    [None, "", "   ", "no json here", "[1, 2, 3]", '"just a string"', "{not: json}", '{"a": 1'],
)
def test_extract_rejects_replies_without_an_object(text: str | None) -> None:
    with pytest.raises(JudgeParseError):
        extract_json_object(text)


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"score": 4}, 4.0),
        ({"score": 1}, 1.0),
        ({"score": 5.0}, 5.0),
        ({"score": 3.5}, 3.5),
        ({"score": "4"}, 4.0),
        ({"score": " 2 "}, 2.0),
        ({"score": "4/5"}, 4.0),
        ({"Score": 3}, 3.0),
        ({"rating": 2}, 2.0),
    ],
)
def test_parse_score_accepts(data: dict[str, object], expected: float) -> None:
    assert parse_score(data) == expected  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({}, "no 'score'"),
        ({"score": None}, "must be a number"),
        ({"score": True}, "must be a number"),
        ({"score": "four"}, "must be a number"),
        ({"score": [4]}, "must be a number"),
        ({"score": 0}, "between 1 and 5"),
        ({"score": 6}, "between 1 and 5"),
        ({"score": "10"}, "between 1 and 5"),
        ({"score": "nan"}, "between 1 and 5"),
    ],
)
def test_parse_score_rejects(data: dict[str, object], message: str) -> None:
    with pytest.raises(JudgeParseError, match=message):
        parse_score(data)  # type: ignore[arg-type]


def test_parse_score_custom_range() -> None:
    assert parse_score({"score": 0}, low=0, high=10) == 0.0


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("pass", True),
        ("PASS", True),
        (" Pass. ", True),
        ("yes", True),
        ("true", True),
        (True, True),
        ("fail", False),
        ("Failed", False),
        ("no", False),
        (False, False),
    ],
)
def test_parse_verdict(value: object, expected: bool) -> None:
    assert parse_verdict({"verdict": value}) is expected  # type: ignore[dict-item]


def test_parse_verdict_alternative_keys() -> None:
    assert parse_verdict({"passed": True}) is True
    assert parse_verdict({"Result": "fail"}) is False


@pytest.mark.parametrize("data", [{}, {"verdict": "maybe"}, {"verdict": 1}, {"verdict": None}])
def test_parse_verdict_rejects(data: dict[str, object]) -> None:
    with pytest.raises(JudgeParseError):
        parse_verdict(data)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"reasoning": " Good answer. "}, "Good answer."),
        ({"reason": "r"}, "r"),
        ({"Explanation": "e"}, "e"),
        ({"rationale": ["a", "b"]}, '["a", "b"]'),
        ({"reasoning": ""}, None),
        ({"reasoning": None}, None),
        ({}, None),
    ],
)
def test_parse_reasoning(data: dict[str, object], expected: str | None) -> None:
    assert parse_reasoning(data) == expected  # type: ignore[arg-type]
