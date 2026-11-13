"""Tests for :class:`agentlens.replay.ReplayDivergence` and :func:`request_diff`."""

from __future__ import annotations

import json

import pytest

from agentlens.replay import (
    DIFF_MAX_LINE_CHARS,
    DIFF_MAX_LINES,
    CassetteMissError,
    ReplayDivergence,
    request_diff,
)

KEY_A = "a" * 64
KEY_B = "b" * 64


def test_request_diff_is_empty_for_equal_documents_whatever_the_key_order() -> None:
    diff, truncated = request_diff({"a": 1, "b": [1, 2]}, {"b": [1, 2], "a": 1})
    assert (diff, truncated) == ("", False)


def test_request_diff_is_a_unified_diff_of_pretty_canonical_json() -> None:
    diff, truncated = request_diff(
        {"z": 1, "messages": [{"role": "user", "content": "hi"}]},
        {"messages": [{"content": "hello", "role": "user"}], "z": 1},
        expected_label="old",
        actual_label="new",
    )
    assert not truncated
    lines = diff.splitlines()
    assert lines[:2] == ["--- old", "+++ new"]
    assert '-      "content": "hi",' in lines
    assert '+      "content": "hello",' in lines
    assert not diff.endswith("\n")


def test_request_diff_of_an_absent_side_shows_the_whole_other_side() -> None:
    diff, _ = request_diff(None, {"a": 1}, present=(False, True))
    assert diff.splitlines()[3:] == ["+{", '+  "a": 1', "+}"]
    diff, _ = request_diff({"a": 1}, None, present=(True, False))
    assert diff.splitlines()[3:] == ["-{", '-  "a": 1', "-}"]


def test_request_diff_keeps_unicode_readable() -> None:
    diff, _ = request_diff({"q": "caffè"}, {"q": "tè"})
    assert '-  "q": "caffè"' in diff.splitlines()


def test_request_diff_bounds_the_number_of_lines() -> None:
    before = {f"k{n:03}": n for n in range(100)}
    after = {f"k{n:03}": -n for n in range(100)}
    diff, truncated = request_diff(before, after, max_lines=10)
    lines = diff.splitlines()
    assert truncated
    assert len(lines) == 11
    assert lines[-1].startswith("... (")
    assert lines[-1].endswith(" more diff lines)")


def test_request_diff_bounds_the_length_of_lines() -> None:
    diff, truncated = request_diff({"q": "x" * 1000}, {"q": "y" * 1000}, max_line_chars=50)
    assert truncated
    changed = [line for line in diff.splitlines() if line[:1] in "+-"]
    assert all(len(line) <= 50 for line in changed)
    assert changed[-1].endswith("...")


def test_default_bounds_are_used() -> None:
    before = {"messages": [f"m{n} " + "x" * 2000 for n in range(300)]}
    diff, truncated = request_diff(before, {"messages": []})
    lines = diff.splitlines()
    assert truncated
    assert len(lines) == DIFF_MAX_LINES + 1
    assert max(len(line) for line in lines) <= DIFF_MAX_LINE_CHARS


def test_create_request_changed_llm() -> None:
    divergence = ReplayDivergence.create(
        "llm",
        2,
        expected_request={"messages": [{"role": "user", "content": "hi"}]},
        actual_request={"messages": [{"role": "user", "content": "hello"}]},
        expected_key=KEY_A,
        actual_key=KEY_B,
        mode="strict",
    )
    assert divergence.reason == "request_changed"
    assert divergence.diff.startswith("--- recorded llm call #2\n+++ replayed llm call #2\n")
    assert divergence.summary() == "llm call #2: request differs from the recording [strict]"
    assert str(divergence) == f"{divergence.summary()}\n{divergence.diff}"


def test_create_tool_changed_diffs_name_and_arguments() -> None:
    divergence = ReplayDivergence.create(
        "tool",
        0,
        expected_request={"order_id": "1"},
        actual_request={"order_id": "1"},
        expected_key=KEY_A,
        actual_key=KEY_B,
        expected_name="lookup_order",
        actual_name="refund",
    )
    assert divergence.reason == "tool_changed"
    assert divergence.summary() == "tool call #0 ('refund'): recorded tool is 'lookup_order'"
    assert '-  "name": "lookup_order"' in divergence.diff.splitlines()
    assert '+  "name": "refund"' in divergence.diff.splitlines()


@pytest.mark.parametrize(
    ("expected_key", "actual_key", "reason", "text"),
    [
        (None, KEY_B, "extra_call", "not in the recording (the replay made an extra call)"),
        (KEY_A, None, "missing_call", "recorded but not made by the replay"),
    ],
)
def test_create_extra_and_missing_calls(
    expected_key: str | None, actual_key: str | None, reason: str, text: str
) -> None:
    divergence = ReplayDivergence.create(
        "tool",
        3,
        expected_request={"x": 1} if expected_key else None,
        actual_request={"x": 1} if actual_key else None,
        expected_key=expected_key,
        actual_key=actual_key,
        expected_name="t" if expected_key else None,
        actual_name="t" if actual_key else None,
    )
    assert divergence.reason == reason
    assert divergence.summary() == f"tool call #3 ('t'): {text}"
    assert divergence.diff


def test_str_without_diff_is_the_summary() -> None:
    divergence = ReplayDivergence(kind="llm", index=0, reason="request_changed")
    assert str(divergence) == "llm call #0: request differs from the recording"


def test_to_dict_is_json_ready_and_can_drop_requests() -> None:
    divergence = ReplayDivergence.create(
        "llm",
        1,
        expected_request={"messages": [{"role": "user", "content": "a"}]},
        actual_request={"messages": [{"role": "user", "content": "b"}]},
        expected_key=KEY_A,
        actual_key=KEY_B,
        mode="lenient",
        span_id="s1",
    )
    full = divergence.to_dict()
    json.dumps(full)
    assert full["summary"] == divergence.summary()
    assert full["span_id"] == "s1"
    assert full["actual_request"] == divergence.actual_request
    small = divergence.to_dict(include_requests=False)
    assert "expected_request" not in small
    assert "actual_request" not in small
    assert small["diff"] == divergence.diff
    assert (
        ReplayDivergence.model_validate({k: v for k, v in full.items() if k != "summary"})
        == divergence
    )


def test_miss_error_accepts_a_prebuilt_divergence_without_diff() -> None:
    divergence = ReplayDivergence(kind="llm", index=0, reason="extra_call", mode="strict")
    err = CassetteMissError(
        kind="llm",
        index=0,
        mode="strict",
        expected=None,
        actual_request=None,
        actual_key=KEY_A,
        divergence=divergence,
    )
    assert err.divergence is divergence
    assert str(err) == "strict cassette miss: no recorded llm call #0"
