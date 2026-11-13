"""Tests for secret redaction before persistence (NFR-5)."""

from __future__ import annotations

import logging
from pathlib import Path

import pytest
from pydantic import JsonValue

import agentlens
from agentlens.schema import Span, SpanKind, Trace
from agentlens.store import SQLiteStore
from agentlens.tracing import (
    DEFAULT_REDACTOR,
    REDACTED,
    PatternRedactor,
    redact_trace,
)

# Fake credentials in the shape of real ones (none of them is valid).
OPENAI_KEY = "sk-proj-AbCdEf0123456789XyZaBcDeF"
ANTHROPIC_KEY = "sk-ant-api03-AbCdEf0123456789_XyZ-aBcDeF"
AWS_KEY = "AKIAIOSFODNN7EXAMPLE"
GITHUB_TOKEN = "ghp_" + "a1B2c3D4e5" * 4
GITHUB_PAT = "github_pat_11ABCDEFG0123456789_abcdefghijKLMNOP"
GOOGLE_KEY = "AIza" + "SyA1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p6Q"
SLACK_TOKEN = "xoxb-123456789012-abcdefABCDEF"
JWT = "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.dozjgNryP4J3jVmNHl0w5N_XgL0n3I9PlFUP0THsR8U"
BEARER = "tok_9f8e7d6c5b4a3210"
PEM = "-----BEGIN RSA PRIVATE KEY-----\nMIIEowIBAAKCAQEA\nabc\n-----END RSA PRIVATE KEY-----"

ALL_SECRETS = [OPENAI_KEY, ANTHROPIC_KEY, AWS_KEY, GITHUB_TOKEN, GITHUB_PAT, GOOGLE_KEY]
ALL_SECRETS += [SLACK_TOKEN, JWT, BEARER, PEM, "hunter2hunter2", "s3cr3t-value"]


def _leaks(trace: Trace) -> list[str]:
    dumped = trace.model_dump_json()
    return [s for s in ALL_SECRETS if s in dumped or s.replace("\n", "\\n") in dumped]


# -- PatternRedactor -------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        (f"key is {OPENAI_KEY}.", f"key is {REDACTED}."),
        (f"key={ANTHROPIC_KEY}", f"key={REDACTED}"),
        (f"Authorization: Bearer {BEARER}", f"Authorization: Bearer {REDACTED}"),
        (f"bearer {BEARER}", f"bearer {REDACTED}"),
        (f"aws {AWS_KEY} here", f"aws {REDACTED} here"),
        (f"git {GITHUB_TOKEN}", f"git {REDACTED}"),
        (f"git {GITHUB_PAT}", f"git {REDACTED}"),
        (f"google {GOOGLE_KEY}", f"google {REDACTED}"),
        (f"slack {SLACK_TOKEN}", f"slack {REDACTED}"),
        (f"jwt {JWT}", f"jwt {REDACTED}"),
        (f"pem {PEM} end", f"pem {REDACTED} end"),
        ("api_key=hunter2hunter2&x=1", f"api_key={REDACTED}&x=1"),
        ('{"password": "hunter2hunter2"}', f'{{"password": "{REDACTED}"}}'),
        ("OPENAI_API_KEY: s3cr3t-value", f"OPENAI_API_KEY: {REDACTED}"),
    ],
)
def test_default_patterns_mask_secrets_in_text(text: str, expected: str) -> None:
    assert DEFAULT_REDACTOR(text) == expected


@pytest.mark.parametrize(
    "text",
    [
        "Where is order 1234?",
        "sk-short",  # too short to be a key
        "max_tokens: 1024, temperature=0.2",
        "token count: 42",
        "the best approach is a Bearer of good news",
        "task-1234567890abcdefghij",
    ],
)
def test_default_patterns_leave_ordinary_text_alone(text: str) -> None:
    assert DEFAULT_REDACTOR(text) == text


def test_sensitive_keys_are_masked_whatever_the_value() -> None:
    payload: JsonValue = {
        "api_key": "plain",
        "Authorization": {"scheme": "Basic", "value": "dXNlcjpwYXNz"},
        "X-API-Key": 123,
        "headers": [{"cookie": "session=1"}, {"accept": "json"}],
        "openai_api_key": "plain",
        "db_password": "plain",
        "max_tokens": 100,
        "input_tokens": 5,
        "model": "gpt-4o",
    }
    assert DEFAULT_REDACTOR(payload) == {
        "api_key": REDACTED,
        "Authorization": REDACTED,
        "X-API-Key": REDACTED,
        "headers": [{"cookie": REDACTED}, {"accept": "json"}],
        "openai_api_key": REDACTED,
        "db_password": REDACTED,
        "max_tokens": 100,
        "input_tokens": 5,
        "model": "gpt-4o",
    }


def test_redactor_does_not_mutate_input_and_keeps_scalars() -> None:
    payload: JsonValue = {"a": [OPENAI_KEY, 1, 2.5, True, None]}
    assert DEFAULT_REDACTOR(payload) == {"a": [REDACTED, 1, 2.5, True, None]}
    assert payload == {"a": [OPENAI_KEY, 1, 2.5, True, None]}


def test_custom_patterns_keys_and_replacement() -> None:
    redactor = PatternRedactor([r"\bINTERNAL-\d{4}\b"], keys={"Session-Id"}, replacement="***")
    assert redactor({"session_id": "x", "note": "ref INTERNAL-1234", "k": OPENAI_KEY}) == {
        "session_id": "***",
        "note": "ref ***",
        "k": OPENAI_KEY,  # default patterns replaced, not extended
    }


def test_secret_group_is_optional() -> None:
    redactor = PatternRedactor([r"pin(?: (?P<secret>\d{4}))?"])
    assert redactor("pin 1234") == f"pin {REDACTED}"
    assert redactor("pin") == REDACTED


# -- redact_trace -----------------------------------------------------------------


def _leaky_trace() -> Trace:
    trace = Trace(
        name="agent",
        input={"question": "hi", "config": {"api_key": "hunter2hunter2"}},
        output=[{"answer": f"used {ANTHROPIC_KEY}"}],
        metadata={"env": {"AWS": AWS_KEY}, "git_sha": "abc123"},
        error=f"AuthError: invalid key {OPENAI_KEY}",
    )
    root = Span(
        trace_id=trace.id,
        kind=SpanKind.AGENT_STEP,
        name="step",
        input={"messages": [{"role": "user", "content": f"token {GITHUB_TOKEN}"}]},
        attributes={"headers": {"Authorization": f"Bearer {BEARER}"}, "model": "m"},
    )
    child = Span(
        trace_id=trace.id,
        parent_id=root.id,
        kind=SpanKind.TOOL_CALL,
        name="http",
        input={"url": "https://x", "params": {"password": "s3cr3t-value"}},
        output={"body": {"nested": [[f"jwt={JWT}"]], "pem": PEM}},
        error=f"HTTPError: 401 for Bearer {BEARER}",
    )
    trace.spans.extend([root, child])
    return trace


def test_redact_trace_masks_nested_payloads_everywhere() -> None:
    trace = _leaky_trace()
    assert _leaks(trace)  # sanity: the fixture does leak

    redacted = redact_trace(trace)

    assert _leaks(redacted) == []
    assert redacted.input == {"question": "hi", "config": {"api_key": REDACTED}}
    assert redacted.output == [{"answer": f"used {REDACTED}"}]
    assert redacted.metadata == {"env": {"AWS": REDACTED}, "git_sha": "abc123"}
    assert redacted.error == f"AuthError: invalid key {REDACTED}"
    root, child = redacted.spans
    assert root.attributes == {"headers": {"Authorization": REDACTED}, "model": "m"}
    assert child.input == {"url": "https://x", "params": {"password": REDACTED}}
    assert child.error == f"HTTPError: 401 for Bearer {REDACTED}"
    # Structure, ids and timings are preserved.
    assert (redacted.id, redacted.name, redacted.start_time) == (
        trace.id,
        trace.name,
        trace.start_time,
    )
    assert [s.id for s in redacted.spans] == [s.id for s in trace.spans]
    assert child.parent_id == root.id
    Trace.model_validate(redacted.model_dump())  # still a valid trace


def test_redact_trace_does_not_modify_the_original() -> None:
    trace = _leaky_trace()
    before = trace.model_dump()
    redact_trace(trace)
    assert trace.model_dump() == before


def test_redact_trace_without_secrets_is_equal() -> None:
    trace = Trace(name="clean", input={"q": "hi"}, metadata={"case": "c1"})
    assert redact_trace(trace) == trace


def test_redact_trace_with_custom_callable_coerces_results() -> None:
    def shout(value: JsonValue) -> JsonValue:
        if isinstance(value, dict):
            return ["not", "a", "dict"]
        return 42 if isinstance(value, str) else value

    trace = Trace(name="t", input="x", metadata={"a": 1}, error="boom")
    redacted = redact_trace(trace, shout)
    assert redacted.input == 42
    assert redacted.metadata == {"redacted": ["not", "a", "dict"]}
    assert redacted.error == "42"


# -- integration with run() --------------------------------------------------------


@agentlens.trace(kind="tool_call")
def call_api(query: str, api_key: str) -> dict[str, str]:
    return {"result": query, "echo": f"Bearer {BEARER}"}


def test_run_redacts_trace_before_sink(sink: list[Trace]) -> None:
    with (
        agentlens.run(
            "agent", input={"q": OPENAI_KEY}, metadata={"token": "t"}, sink=sink.append
        ) as live,
        agentlens.span("s", attributes={"auth": {"secret": "x"}}),
    ):
        live.output = call_api("hi", api_key="hunter2hunter2")

    [persisted] = sink
    assert _leaks(persisted) == []
    assert persisted.input == {"q": REDACTED}
    assert persisted.metadata == {"token": REDACTED}
    assert persisted.output == {"result": "hi", "echo": f"Bearer {REDACTED}"}
    span, tool = persisted.spans
    assert span.attributes == {"auth": {"secret": REDACTED}}
    assert tool.input == {"query": "hi", "api_key": REDACTED}
    # The live trace handed to user code is not redacted.
    assert live.input == {"q": OPENAI_KEY}
    assert live.spans[1].input == {"query": "hi", "api_key": "hunter2hunter2"}


def test_run_redacts_error_message(sink: list[Trace]) -> None:
    with pytest.raises(PermissionError), agentlens.run("agent", sink=sink.append):
        raise PermissionError(f"rejected {ANTHROPIC_KEY}")
    assert sink[0].error == f"PermissionError: rejected {REDACTED}"


def test_run_with_custom_redactor(sink: list[Trace]) -> None:
    with agentlens.run("agent", input="customer 42", sink=sink.append, redactor=lambda v: "x"):
        pass
    assert sink[0].input == "x"


def test_run_with_redaction_disabled(sink: list[Trace]) -> None:
    with agentlens.run("agent", input=OPENAI_KEY, sink=sink.append, redactor=None) as t:
        pass
    assert sink == [t]
    assert sink[0] is t


def test_failing_redactor_drops_trace_and_never_raises(
    sink: list[Trace], caplog: pytest.LogCaptureFixture
) -> None:
    def broken(value: JsonValue) -> JsonValue:
        raise RuntimeError("redactor bug")

    with (
        caplog.at_level(logging.ERROR, "agentlens.tracing"),
        agentlens.run("agent", input=OPENAI_KEY, sink=sink.append, redactor=broken) as t,
    ):
        t.output = "done"
    assert sink == []  # fail closed: nothing unredacted is persisted
    assert t.output == "done"
    assert "redaction failed" in caplog.text


def test_store_never_receives_secrets(tmp_path: Path) -> None:
    with SQLiteStore(tmp_path / "traces.db") as store:
        with agentlens.run("agent", input=f"Bearer {BEARER}", sink=store.save_trace) as t:
            call_api("q", api_key=GITHUB_TOKEN)
        stored = store.get_trace(t.id)
    assert stored is not None
    assert _leaks(stored) == []
    assert stored.input == f"Bearer {REDACTED}"
