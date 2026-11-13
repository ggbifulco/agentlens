"""Tests for the exact_match, contains, regex and json_schema evaluators."""

from __future__ import annotations

from typing import Any

import pytest
from pydantic import JsonValue

from agentlens.evals import (
    UNSET,
    Contains,
    EvaluatorConfigError,
    ExactMatch,
    JsonSchema,
    Regex,
    run_evaluator,
)
from tests.evals.builtin.conftest import make_case, make_trace

# -- exact_match ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "expected", "options", "passed"),
    [
        ("Paris", "Paris", {}, True),
        ("  Paris\n", "Paris", {}, True),
        ("  Paris\n", "Paris", {"strip": False}, False),
        ("paris", "Paris", {}, False),
        ("PARIS", "Paris", {"case_sensitive": False}, True),
        ("Straße", "STRASSE", {"case_sensitive": False}, True),
        ("New   York\ncity", "New York city", {}, False),
        ("New   York\ncity", "New York city", {"normalise_whitespace": True}, True),
        ("Café", "Café", {}, True),
        ("日本語 ✓", "日本語 ✓", {}, True),
        ("Paris!", "Paris", {}, False),
        ("", "", {}, True),
        # non-string output vs text expectation: compared as canonical JSON text
        (42, "42", {}, True),
        ({"b": 2, "a": 1}, '{"a": 1, "b": 2}', {}, True),
        (True, "true", {}, True),
        # structural expectations
        ({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1}, {}, True),
        ({"a": 1, "b": [2, 1]}, {"a": 1, "b": [1, 2]}, {}, False),
        ('{"a": 1}', {"a": 1}, {}, True),
        ("not json", {"a": 1}, {}, False),
        (1, 1.0, {}, True),
        (True, 1, {}, False),
        ("42", 42, {}, True),
        ({"name": " ADA "}, {"name": "ada"}, {"case_sensitive": False}, True),
        ({"name": "Ada"}, {"name": "Ada", "x": None}, {}, False),
        (None, {"a": 1}, {}, False),
        ([1, "x", None, False], [1, "x", None, False], {}, True),
    ],
)
def test_exact_match(
    output: JsonValue, expected: JsonValue, options: dict[str, Any], passed: bool
) -> None:
    result = ExactMatch(expected, **options).evaluate(make_trace(output), make_case())
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert result.evaluator == "exact_match"
    assert result.reason


def test_exact_match_value_param_overrides_case_expected() -> None:
    trace = make_trace("Rome")
    assert not ExactMatch().evaluate(trace, make_case("Paris")).passed
    result = ExactMatch("Rome").evaluate(trace, make_case("Paris"))
    assert result.passed
    assert result.details["source"] == "value"
    assert ExactMatch().evaluate(trace, make_case("Rome")).details["source"] == "case.expected"


def test_exact_match_explicit_none_expects_null_output() -> None:
    assert ExactMatch(None).evaluate(make_trace(None), make_case("ignored")).passed
    assert ExactMatch(None).evaluate(make_trace("null"), make_case()).passed
    assert not ExactMatch(None).evaluate(make_trace("x"), make_case()).passed


def test_exact_match_without_expectation_is_a_config_error() -> None:
    evaluator = ExactMatch()
    assert evaluator.value is UNSET
    with pytest.raises(EvaluatorConfigError, match="nothing to compare with in case 'c1'"):
        evaluator.evaluate(make_trace("x"), make_case())
    result = run_evaluator(evaluator, make_trace("x"), make_case())
    assert not result.passed
    error = result.details["error"]
    assert isinstance(error, dict)
    assert error["type"] == "EvaluatorConfigError"


def test_exact_match_reasons_and_details() -> None:
    fail = ExactMatch("Paris").evaluate(make_trace("Rome"), make_case())
    assert fail.reason == "output 'Rome' != expected 'Paris'"
    assert fail.details == {
        "mode": "text",
        "source": "value",
        "expected": "Paris",
        "actual": "Rome",
    }
    none = ExactMatch("Paris").evaluate(make_trace(None), make_case())
    assert none.reason == "no output (None); expected 'Paris'"
    assert none.details["actual"] == "null"
    ok = ExactMatch({"a": 1}).evaluate(make_trace({"a": 1}), make_case())
    assert ok.reason == "output matches the expected value (value)"
    assert ok.details["mode"] == "json"
    long = ExactMatch("x").evaluate(make_trace("y" * 5000), make_case())
    assert long.reason is not None
    assert "(5000 chars)" in long.reason
    assert len(str(long.details["actual"])) == 2001


def test_exact_match_path() -> None:
    trace = make_trace({"answer": "Paris", "sources": ["wiki"]})
    assert ExactMatch("Paris", path="answer").evaluate(trace, make_case()).passed
    assert ExactMatch(["wiki"], path="sources").evaluate(trace, make_case()).passed
    missing = ExactMatch("Paris", path="city").evaluate(trace, make_case())
    assert not missing.passed
    assert missing.reason == "output has no 'city'"


@pytest.mark.parametrize(
    "kwargs",
    [{"case_sensitive": "no"}, {"strip": 1}, {"normalise_whitespace": None}, {"path": ""}],
)
def test_exact_match_rejects_bad_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(EvaluatorConfigError):
        ExactMatch("x", **kwargs)


# -- contains ------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "text", "options", "score", "passed"),
    [
        ("Refund issued", "Refund", {}, 1.0, True),
        ("Refund issued", "refund", {}, 0.0, False),
        ("Refund issued", "refund", {"case_sensitive": False}, 1.0, True),
        ("Refund issued", ["Refund", "issued", "order"], {}, 2 / 3, False),
        ("Refund issued", ["Refund", "order"], {"mode": "any"}, 0.5, True),
        ("Refund issued", ["order", "cancel"], {"mode": "any"}, 0.0, False),
        ("Refund issued", ["Refund", "issued"], {}, 1.0, True),
        ("refund\n  issued", "refund issued", {}, 0.0, False),
        ("refund\n  issued", "refund issued", {"normalise_whitespace": True}, 1.0, True),
        ("Café ouvert", "Café", {}, 1.0, True),
        ("ÜBER alles", "über", {"case_sensitive": False}, 1.0, True),
        (None, "anything", {}, 0.0, False),
        (None, ["a", "b"], {"mode": "any"}, 0.0, False),
        ({"status": "ok", "id": 7}, '"status": "ok"', {}, 1.0, True),
        (12345, "234", {}, 1.0, True),
        (["refund"], "refund", {}, 1.0, True),
    ],
)
def test_contains(
    output: JsonValue,
    text: str | list[str],
    options: dict[str, Any],
    score: float,
    passed: bool,
) -> None:
    result = Contains(text, **options).evaluate(make_trace(output), make_case())
    assert result.score == pytest.approx(score)
    assert result.passed is passed
    assert result.evaluator == "contains"


def test_contains_reasons_and_details() -> None:
    trace = make_trace("Refund issued for order 1234")
    ok = Contains("order 1234").evaluate(trace, make_case())
    assert ok.reason == "output contains 'order 1234'"
    assert ok.details == {"mode": "all", "matched": ["order 1234"], "missing": []}
    partial = Contains(["Refund", "cancel"]).evaluate(trace, make_case())
    assert partial.reason == "output is missing 'cancel' (1/2 found)"
    assert partial.details["missing"] == ["cancel"]
    anyok = Contains(["cancel", "Refund"], mode="any").evaluate(trace, make_case())
    assert anyok.reason == "output contains 'Refund' (1/2 found, mode any)"
    none = Contains(["x", "y"], mode="any").evaluate(trace, make_case())
    assert none.reason == "output contains none of 'x', 'y'"
    empty = Contains("x").evaluate(make_trace(None), make_case())
    assert empty.reason == "no output (None) to search; 0/1 found"
    many = Contains([f"w{i}" for i in range(8)]).evaluate(trace, make_case())
    assert many.reason is not None
    assert many.reason.endswith("and 3 more (0/8 found)")


def test_contains_falls_back_to_case_expected() -> None:
    trace = make_trace("Refund issued for order 1234")
    assert Contains().evaluate(trace, make_case("order 1234")).passed
    assert Contains().evaluate(trace, make_case(["Refund", "1234"])).passed
    with pytest.raises(EvaluatorConfigError, match="no substrings for case 'c1'"):
        Contains().evaluate(trace, make_case())
    with pytest.raises(EvaluatorConfigError, match="expected \\(used by contains\\)"):
        Contains().evaluate(trace, make_case({"tools": ["a"]}))


def test_contains_path() -> None:
    trace = make_trace({"answer": "Paris is the capital", "note": "Rome"})
    assert Contains("Paris", path="answer").evaluate(trace, make_case()).passed
    assert not Contains("Rome", path="answer").evaluate(trace, make_case()).passed
    missing = Contains(["a", "b"], path="nope").evaluate(trace, make_case())
    assert missing.details == {"mode": "all", "matched": [], "missing": ["a", "b"]}
    assert missing.reason == "output has no 'nope'"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"text": ""},
        {"text": []},
        {"text": ["ok", ""]},
        {"text": ["ok", 3]},
        {"text": 5},
        {"text": "x", "mode": "some"},
        {"text": "x", "case_sensitive": "yes"},
        {"text": "x", "path": [None]},
    ],
)
def test_contains_rejects_bad_options(kwargs: dict[str, Any]) -> None:
    with pytest.raises(EvaluatorConfigError):
        Contains(**kwargs)


# -- regex ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("output", "pattern", "options", "passed"),
    [
        ("Refund for order 1234", r"order \d+", {}, True),
        ("Refund for order 1234", r"^order", {}, False),
        ("Refund for order 1234", r"refund", {}, False),
        ("Refund for order 1234", r"refund", {"ignore_case": True}, True),
        ("line one\nline two", r"^line two$", {}, False),
        ("line one\nline two", r"^line two$", {"multiline": True}, True),
        ("a\nb", r"a.b", {}, False),
        ("a\nb", r"a.b", {"dotall": True}, True),
        ("order 1234", r"order", {"mode": "match"}, True),
        ("my order 1234", r"order", {"mode": "match"}, False),
        ("order 1234", r"order", {"mode": "fullmatch"}, False),
        ("order 1234", r"order \d{4}", {"mode": "fullmatch"}, True),
        ("prix: 12 €", r"\d+ €", {}, True),
        ("ПРИВЕТ мир", r"привет", {"ignore_case": True}, True),
        ({"id": 7}, r'"id": \d', {}, True),
        (3.5, r"^3\.5$", {}, True),
        (None, r".*", {}, False),
        ("", r"^$", {}, True),
    ],
)
def test_regex(output: JsonValue, pattern: str, options: dict[str, Any], passed: bool) -> None:
    result = Regex(pattern, **options).evaluate(make_trace(output), make_case())
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert result.evaluator == "regex"


def test_regex_details_and_reasons() -> None:
    trace = make_trace("Refund for order 1234")
    ok = Regex(r"order (?P<id>\d+)").evaluate(trace, make_case())
    assert ok.reason == "output matches /order (?P<id>\\d+)/ at 11: 'order 1234'"
    assert ok.details == {
        "pattern": r"order (?P<id>\d+)",
        "mode": "search",
        "match": "order 1234",
        "span": [11, 21],
        "groups": ["1234"],
        "named_groups": {"id": "1234"},
    }
    optional = Regex(r"order (x)?").evaluate(trace, make_case())
    assert optional.details["groups"] == [None]
    fail = Regex(r"cancel", mode="fullmatch").evaluate(trace, make_case())
    assert fail.reason == "output 'Refund for order 1234' does not fully match /cancel/"
    assert fail.details == {"pattern": "cancel", "mode": "fullmatch"}
    assert "does not contain a match for" in str(Regex("x").evaluate(trace, make_case()).reason)
    start = Regex("x", mode="match").evaluate(trace, make_case())
    assert "does not start with a match for /x/" in str(start.reason)
    none = Regex("x").evaluate(make_trace(None), make_case())
    assert none.reason == "no output (None) to match against /x/"


def test_regex_path() -> None:
    trace = make_trace({"answer": "order 1234", "note": "n/a"})
    assert Regex(r"^order \d+$", path="answer").evaluate(trace, make_case()).passed
    missing = Regex("x", path="nope").evaluate(trace, make_case())
    assert missing.reason == "output has no 'nope'"
    assert not missing.passed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"pattern": "("}, "invalid regex '\\('"),
        ({"pattern": "[a-"}, "invalid regex"),
        ({"pattern": ""}, "non-empty string"),
        ({"pattern": 3}, "non-empty string"),
        ({"pattern": "x", "mode": "find"}, "mode must be one of"),
        ({"pattern": "x", "ignore_case": "true"}, "ignore_case"),
        ({"pattern": "x", "path": ""}, "path"),
    ],
)
def test_regex_config_errors_at_build_time(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        Regex(**kwargs)


# -- json_schema ---------------------------------------------------------------

ORDER_SCHEMA: dict[str, JsonValue] = {
    "type": "object",
    "properties": {"id": {"type": "integer"}, "status": {"enum": ["open", "refunded"]}},
    "required": ["id", "status"],
}


@pytest.mark.parametrize(
    ("output", "passed"),
    [
        ({"id": 1, "status": "open"}, True),
        ('{"id": 1, "status": "refunded"}', True),
        ('```json\n{"id": 1, "status": "open"}\n```', True),
        ('```\n{"id": 1, "status": "open"}```', True),
        ({"id": "1", "status": "open"}, False),
        ({"id": 1}, False),
        ([], False),
        ("not json at all", False),
        (None, False),
        ('  {"id": 2, "status": "open"}  ', True),
    ],
)
def test_json_schema(output: JsonValue, passed: bool) -> None:
    result = JsonSchema(ORDER_SCHEMA).evaluate(make_trace(output), make_case())
    assert result.passed is passed
    assert result.score == (1.0 if passed else 0.0)
    assert result.evaluator == "json_schema"


def test_json_schema_error_listing() -> None:
    result = JsonSchema(ORDER_SCHEMA).evaluate(
        make_trace({"id": "x", "status": "lost"}), make_case()
    )
    assert result.reason == (
        "output does not match the JSON schema, 2 errors: "
        "$.id: 'x' is not of type 'integer'; "
        "$.status: 'lost' is not one of ['open', 'refunded']"
    )
    assert result.details == {
        "source": "schema",
        "error_count": 2,
        "errors": [
            {"path": "$.id", "message": "'x' is not of type 'integer'", "validator": "type"},
            {
                "path": "$.status",
                "message": "'lost' is not one of ['open', 'refunded']",
                "validator": "enum",
            },
        ],
    }
    one = JsonSchema(ORDER_SCHEMA).evaluate(make_trace({"id": 1}), make_case())
    assert one.reason == (
        "output does not match the JSON schema, 1 error: $: 'status' is a required property"
    )
    capped = JsonSchema({"type": "array", "items": {"type": "string"}}, max_errors=2).evaluate(
        make_trace([1, 2, 3]), make_case()
    )
    assert capped.details["error_count"] == 3
    assert isinstance(capped.details["errors"], list)
    assert len(capped.details["errors"]) == 2
    assert str(capped.reason).endswith("(+1 more)")


def test_json_schema_reasons_for_unparsable_and_valid_output() -> None:
    bad = JsonSchema(ORDER_SCHEMA).evaluate(make_trace("{oops"), make_case())
    assert bad.reason is not None
    assert bad.reason.startswith("output is not valid JSON (Expecting property name")
    assert bad.details == {"source": "schema"}
    ok = JsonSchema(ORDER_SCHEMA).evaluate(make_trace({"id": 1, "status": "open"}), make_case())
    assert ok.reason == "output matches the JSON schema"
    assert ok.details == {"source": "schema", "error_count": 0, "errors": []}


def test_json_schema_none_output_is_validated_as_null() -> None:
    assert JsonSchema({"type": "null"}).evaluate(make_trace(None), make_case()).passed


def test_json_schema_parse_json_false_validates_strings_as_strings() -> None:
    schema: dict[str, JsonValue] = {"type": "string", "maxLength": 5}
    assert JsonSchema(schema, parse_json=False).evaluate(make_trace("hello"), make_case()).passed
    assert not JsonSchema(schema).evaluate(make_trace("hello"), make_case()).passed
    assert (
        not JsonSchema(schema, parse_json=False)
        .evaluate(make_trace("too long"), make_case())
        .passed
    )


def test_json_schema_from_case_expected() -> None:
    case = make_case({"schema": ORDER_SCHEMA, "tools": ["lookup"]})
    result = JsonSchema().evaluate(make_trace({"id": 1, "status": "open"}), case)
    assert result.passed
    assert result.details["source"] == "case.expected.schema"
    with pytest.raises(EvaluatorConfigError, match="no schema for case 'c1'"):
        JsonSchema().evaluate(make_trace({}), make_case())
    with pytest.raises(EvaluatorConfigError, match="no schema"):
        JsonSchema().evaluate(make_trace({}), make_case("a string"))
    with pytest.raises(EvaluatorConfigError, match=r"case 'c1' expected\.schema is not a valid"):
        JsonSchema().evaluate(make_trace({}), make_case({"schema": {"type": "nope"}}))


def test_json_schema_path_and_draft() -> None:
    trace = make_trace({"data": {"id": 1, "status": "open"}, "extra": True})
    assert JsonSchema(ORDER_SCHEMA, path="data").evaluate(trace, make_case()).passed
    missing = JsonSchema(ORDER_SCHEMA, path="nope").evaluate(trace, make_case())
    assert missing.reason == "output has no 'nope'"
    # Draft 4 exclusiveMaximum is a boolean; the draft comes from $schema.
    draft4: dict[str, JsonValue] = {
        "$schema": "http://json-schema.org/draft-04/schema#",
        "type": "number",
        "maximum": 10,
        "exclusiveMaximum": True,
    }
    assert not JsonSchema(draft4).evaluate(make_trace(10), make_case()).passed
    assert JsonSchema(draft4).evaluate(make_trace(9), make_case()).passed


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"schema": {"type": "nope"}}, "schema is not a valid JSON schema"),
        ({"schema": {"required": "id"}}, "not a valid JSON schema"),
        ({"schema": "object"}, "must be a JSON schema \\(a mapping\\), got str"),
        ({"schema": {}, "max_errors": 0}, "max_errors"),
        ({"schema": {}, "max_errors": True}, "max_errors"),
        ({"schema": {}, "parse_json": "yes"}, "parse_json"),
        ({"schema": {}, "path": [1.5]}, "path"),
    ],
)
def test_json_schema_config_errors_at_build_time(kwargs: dict[str, Any], message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        JsonSchema(**kwargs)


# -- shared --------------------------------------------------------------------


@pytest.mark.parametrize(
    ("evaluator", "expected"),
    [
        (
            ExactMatch(),
            "ExactMatch(value=UNSET, case_sensitive=True, strip=True, "
            "normalise_whitespace=False, path=None)",
        ),
        (
            Contains("a", mode="any"),
            "Contains(text=['a'], mode='any', case_sensitive=True, "
            "normalise_whitespace=False, path=None)",
        ),
        (
            Regex("x"),
            "Regex(pattern='x', mode='search', ignore_case=False, multiline=False, "
            "dotall=False, path=None)",
        ),
        (
            JsonSchema({}, path="a"),
            "JsonSchema(schema={}, parse_json=True, max_errors=5, path='a')",
        ),
    ],
)
def test_repr(evaluator: object, expected: str) -> None:
    assert repr(evaluator) == expected
