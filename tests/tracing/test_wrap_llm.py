"""Tests for :func:`agentlens.wrap_llm` (``llm_call`` spans)."""

from __future__ import annotations

import asyncio
import copy
import logging

import pytest

import agentlens
from agentlens.llm import LLMClient, LLMRequest, LLMResponse, Message, MockLLM, ToolCall, ToolSpec
from agentlens.schema import SpanKind, Usage
from agentlens.tracing import Recorder, TracedLLM


def _request(**kwargs: object) -> LLMRequest:
    return LLMRequest(messages=[Message.user("Where is order 1234?")], **kwargs)  # type: ignore[arg-type]


class FailingLLM:
    """A client whose calls always raise."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise ConnectionError("provider down")

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        raise ConnectionError("provider down")


class RawLLM:
    """A provider-like client: no ``model`` attribute, a ``provider`` and raw payloads."""

    provider = "acme"

    def complete(self, request: LLMRequest) -> LLMResponse:
        return LLMResponse(content="ok", raw={"id": "resp_1", "choices": [{"text": "ok"}]})

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return self.complete(request)


class NotAResponseLLM:
    """A client breaking the protocol by returning a plain string."""

    def complete(self, request: LLMRequest) -> LLMResponse:
        return "hello"  # type: ignore[return-value]

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return self.complete(request)


def test_wrapped_client_is_an_llm_client() -> None:
    llm = agentlens.wrap_llm(MockLLM(["hi"]))
    assert isinstance(llm, TracedLLM)
    assert isinstance(llm, LLMClient)
    assert "MockLLM" in repr(llm)


def test_complete_records_llm_call_span() -> None:
    mock = MockLLM(["It shipped yesterday."], model="mock-1")
    llm = agentlens.wrap_llm(mock)
    tools = [ToolSpec(name="lookup_order"), ToolSpec(name="refund")]
    request = _request(temperature=0.2, max_tokens=64, tools=tools)

    with agentlens.run("agent") as t:
        response = llm.complete(request)

    assert response.content == "It shipped yesterday."
    (s,) = t.spans
    assert s.kind is SpanKind.LLM_CALL
    assert s.name == "mock-1"
    assert s.parent_id is None
    assert s.end_time is not None
    assert s.error is None
    assert s.input == request.model_dump(mode="json", exclude_defaults=True)
    assert LLMRequest.model_validate(s.input) == request  # input round-trips
    assert s.output == {"content": "It shipped yesterday.", "tool_calls": [], "model": "mock-1"}
    assert s.usage == response.usage
    assert s.usage is not None
    assert s.usage.input_tokens == 4
    assert s.attributes == {
        "client": "agentlens.llm.mock.MockLLM",
        "model": "mock-1",
        "temperature": 0.2,
        "max_tokens": 64,
        "n_messages": 1,
        "n_tools": 2,
        "tool_names": ["lookup_order", "refund"],
        "request_hash": request.fingerprint(),
        "response_model": "mock-1",
        "n_tool_calls": 0,
    }
    assert t.total_tokens == response.usage.total_tokens  # type: ignore[union-attr]
    assert t.n_steps == 1


def test_request_model_wins_and_tool_calls_are_recorded() -> None:
    call = ToolCall(id="c1", name="lookup_order", arguments={"order_id": "1234"})
    llm = agentlens.wrap_llm(MockLLM([call]))
    request = _request(model="gpt-test", response_format={"type": "json_object"})

    with agentlens.run("agent") as t:
        llm.complete(request)

    (s,) = t.spans
    assert s.name == "gpt-test"
    assert s.attributes["model"] == "gpt-test"
    assert s.attributes["response_format"] == "json_object"
    assert s.attributes["n_tool_calls"] == 1
    assert "tool_names" not in s.attributes
    assert s.output == {
        "content": None,
        "tool_calls": [call.model_dump(mode="json")],
        "model": "gpt-test",
    }


def test_custom_name_provider_and_raw() -> None:
    plain = agentlens.wrap_llm(RawLLM())
    named = agentlens.wrap_llm(RawLLM(), name="planner", provider="override", include_raw=True)

    with agentlens.run("agent") as t:
        plain.complete(_request())
        named.complete(_request())

    first, second = t.spans
    assert first.name == "RawLLM"  # no model anywhere: class name
    assert first.attributes["provider"] == "acme"
    assert first.attributes["model"] is None
    assert first.output == {"content": "ok", "tool_calls": [], "model": None}  # no raw
    assert first.usage is None  # unreported usage stays unknown
    assert second.name == "planner"
    assert second.attributes["provider"] == "override"
    assert second.output == {
        "content": "ok",
        "tool_calls": [],
        "raw": {"id": "resp_1", "choices": [{"text": "ok"}]},
        "model": None,
    }


def test_no_provider_attribute_when_unknown() -> None:
    llm = agentlens.wrap_llm(MockLLM(["hi"]))
    with agentlens.run("agent") as t:
        llm.complete(_request())
    assert "provider" not in t.spans[0].attributes


def test_acomplete_records_span_and_nests_under_trace() -> None:
    llm = agentlens.wrap_llm(MockLLM(["a", "b"]))

    @agentlens.trace(kind="agent_step")
    async def step() -> list[str | None]:
        responses = await asyncio.gather(llm.acomplete(_request()), llm.acomplete(_request()))
        return [r.content for r in responses]

    async def main() -> agentlens.Trace:
        with agentlens.run("agent") as t:
            assert await step() == ["a", "b"]
        return t

    t = asyncio.run(main())
    outer, *calls = t.spans
    assert outer.kind is SpanKind.AGENT_STEP
    assert len(calls) == 2
    assert all(c.kind is SpanKind.LLM_CALL and c.parent_id == outer.id for c in calls)
    assert sorted(c.output["content"] for c in calls) == ["a", "b"]  # type: ignore[index,call-overload]
    assert all(c.usage is not None for c in calls)


def test_errors_are_recorded_and_reraised() -> None:
    llm = agentlens.wrap_llm(FailingLLM())
    with agentlens.run("agent") as t, pytest.raises(ConnectionError, match="provider down"):
        llm.complete(_request())
    (s,) = t.spans
    assert s.error == "ConnectionError: provider down"
    assert s.output is None
    assert s.end_time is not None
    assert agentlens.current_span() is None


def test_async_errors_are_recorded_and_reraised() -> None:
    llm = agentlens.wrap_llm(FailingLLM())

    async def main() -> agentlens.Trace:
        with agentlens.run("agent") as t, pytest.raises(ConnectionError):
            await llm.acomplete(_request())
        return t

    (s,) = asyncio.run(main()).spans
    assert s.error == "ConnectionError: provider down"


def test_passthrough_outside_run() -> None:
    mock = MockLLM(["one", "two"])
    llm = agentlens.wrap_llm(mock)
    assert llm.complete(_request()).content == "one"
    assert asyncio.run(llm.acomplete(_request())).content == "two"
    assert mock.calls == 2
    assert agentlens.current_trace() is None


def test_passthrough_does_not_touch_request(monkeypatch: pytest.MonkeyPatch) -> None:
    def fail(self: LLMRequest) -> str:
        raise AssertionError("fingerprint computed outside a run")

    monkeypatch.setattr(LLMRequest, "fingerprint", fail)
    llm = agentlens.wrap_llm(MockLLM(["one"]))
    assert llm.complete(_request()).content == "one"


def test_attribute_access_is_delegated() -> None:
    mock = MockLLM(["hi"])
    llm = agentlens.wrap_llm(mock)
    llm.complete(_request())
    assert llm.client is mock
    assert llm.calls == 1
    assert llm.requests == mock.requests
    with pytest.raises(AttributeError):
        _ = llm.does_not_exist


def test_copy_and_bare_instance_do_not_recurse() -> None:
    llm = agentlens.wrap_llm(MockLLM(["hi"]))
    clone = copy.copy(llm)
    assert clone.client is llm.client
    bare = TracedLLM.__new__(TracedLLM)
    with pytest.raises(AttributeError):
        _ = bare.client


def test_wrap_is_idempotent() -> None:
    llm = agentlens.wrap_llm(MockLLM(["hi"]))
    assert agentlens.wrap_llm(llm) is llm
    with agentlens.run("agent") as t:
        agentlens.wrap_llm(llm).complete(_request())
    assert len(t.spans) == 1


def test_attribute_failure_still_records_span(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(self: LLMRequest) -> str:
        raise RuntimeError("hash bug")

    monkeypatch.setattr(LLMRequest, "fingerprint", fail)
    llm = agentlens.wrap_llm(MockLLM(["hi"]))
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent") as t:
        assert llm.complete(_request()).content == "hi"
    (s,) = t.spans
    assert s.output == {"content": "hi", "tool_calls": [], "model": "mock"}
    assert "request_hash" not in s.attributes
    assert "could not build llm_call attributes" in caplog.text


def test_output_failure_is_logged_not_raised(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def fail(self: LLMResponse, **kwargs: object) -> object:
        raise RuntimeError("dump bug")

    llm = agentlens.wrap_llm(MockLLM(["hi"]))
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent") as t:
        monkeypatch.setattr(LLMResponse, "model_dump", fail)
        response = llm.complete(_request())
        monkeypatch.undo()
    assert response.content == "hi"
    assert len(t.spans) == 1
    assert "could not end span" in caplog.text
    assert agentlens.current_span() is None


def test_recorder_failure_never_breaks_the_call(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    def boom(*args: object, **kwargs: object) -> None:
        raise RuntimeError("recorder bug")

    monkeypatch.setattr(Recorder, "start_span", boom)
    llm = agentlens.wrap_llm(MockLLM(["hi", "there"]))
    with caplog.at_level(logging.ERROR, "agentlens.tracing"), agentlens.run("agent") as t:
        assert llm.complete(_request()).content == "hi"
        assert asyncio.run(llm.acomplete(_request())).content == "there"
    assert t.spans == []
    assert "could not start span" in caplog.text


def test_non_response_return_is_recorded_as_is() -> None:
    llm = agentlens.wrap_llm(NotAResponseLLM())
    with agentlens.run("agent") as t:
        assert llm.complete(_request()) == "hello"  # type: ignore[comparison-overlap]
    (s,) = t.spans
    assert s.output == "hello"
    assert s.usage is None
    assert "response_model" not in s.attributes


def test_usage_cost_flows_into_trace_totals() -> None:
    usage = Usage(input_tokens=10, output_tokens=5, cost_usd=0.002)
    llm = agentlens.wrap_llm(MockLLM([LLMResponse(content="x", usage=usage)] * 2))
    with agentlens.run("agent") as t:
        llm.complete(_request())
        llm.complete(_request())
    assert t.total_tokens == 30
    assert t.total_cost == pytest.approx(0.004)


def test_secrets_in_requests_are_redacted_before_the_sink(sink: list[agentlens.Trace]) -> None:
    llm = agentlens.wrap_llm(MockLLM(["ok"]))
    secret = "sk-abcdefghijklmnopqrstuvwxyz123456"
    with agentlens.run("agent", sink=sink.append):
        llm.complete(LLMRequest(messages=f"my key is {secret}"))
    (persisted,) = sink
    assert secret not in persisted.model_dump_json()
