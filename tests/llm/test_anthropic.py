"""Tests for the Anthropic Messages API adapter (HTTP mocked with respx)."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

import httpx
import pytest
import respx

import agentlens
from agentlens.llm import (
    ANTHROPIC_VERSION,
    DEFAULT_ANTHROPIC_BASE_URL,
    DEFAULT_MAX_TOKENS,
    AnthropicClient,
    LLMClient,
    LLMError,
    LLMHTTPError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    Message,
    RetryPolicy,
    ToolCall,
    ToolSpec,
)
from agentlens.llm.anthropic import parse_response
from agentlens.schema import SpanKind, Usage

BASE = "http://anthropic.test"
URL = f"{BASE}/v1/messages"
SECRET = "sk-ant-test-secret-key-123456"

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def message_body(
    *blocks: dict[str, Any],
    usage: dict[str, Any] | None = None,
    model: str = "claude-test-20250101",
    stop_reason: str = "end_turn",
) -> dict[str, Any]:
    return {
        "id": "msg_01",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": list(blocks) if blocks else [text("Hello!")],
        "stop_reason": stop_reason,
        "usage": usage or {"input_tokens": 12, "output_tokens": 3},
    }


def text(value: str) -> dict[str, Any]:
    return {"type": "text", "text": value}


def tool_use(call_id: str | None, name: str, arguments: Any) -> dict[str, Any]:
    block: dict[str, Any] = {"type": "tool_use", "name": name}
    if call_id is not None:
        block["id"] = call_id
    if arguments is not None:
        block["input"] = arguments
    return block


def client(**kwargs: Any) -> AnthropicClient:
    kwargs.setdefault("model", "claude-test")
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("api_key", SECRET)
    kwargs.setdefault(
        "retry_policy",
        RetryPolicy(max_retries=2, jitter=lambda d: d, sleep=lambda _: None),
    )
    return AnthropicClient(**kwargs)


def sent(route: respx.Route) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls.last.request.content)
    return body


class TestConstruction:
    def test_is_llm_client(self) -> None:
        assert isinstance(client(), LLMClient)

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        llm = AnthropicClient()
        assert llm.base_url == DEFAULT_ANTHROPIC_BASE_URL
        assert llm.url == "https://api.anthropic.com/v1/messages"
        assert llm.provider == "anthropic"
        assert llm.model is None
        assert llm.max_tokens == DEFAULT_MAX_TOKENS == 1024
        assert llm.retry_policy.max_retries == 2

    def test_base_url_trailing_slash_and_provider(self) -> None:
        llm = AnthropicClient(base_url="http://gateway.test/anthropic/", provider="gateway")
        assert llm.url == "http://gateway.test/anthropic/v1/messages"
        assert llm.provider == "gateway"

    def test_max_retries_shortcut(self) -> None:
        assert AnthropicClient(max_retries=5).retry_policy.max_retries == 5

    def test_invalid_max_tokens(self) -> None:
        with pytest.raises(ValueError, match="max_tokens"):
            AnthropicClient(max_tokens=0)

    def test_repr_hides_key(self) -> None:
        text_ = repr(client())
        assert SECRET not in text_
        assert "claude-test" in text_
        assert BASE in text_


class TestHeaders:
    def test_explicit_key_and_version(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=message_body())
        client().complete(LLMRequest(messages="hi"))
        headers = route.calls.last.request.headers
        assert headers["x-api-key"] == SECRET
        assert headers["anthropic-version"] == ANTHROPIC_VERSION == "2023-06-01"
        assert headers["content-type"] == "application/json"
        assert "authorization" not in headers

    def test_env_fallback(self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-0000000")
        route = respx_mock.post(URL).respond(json=message_body())
        client(api_key=None).complete(LLMRequest(messages="hi"))
        assert route.calls.last.request.headers["x-api-key"] == "sk-ant-from-env-0000000"

    def test_no_key(self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        route = respx_mock.post(URL).respond(json=message_body())
        client(api_key=None).complete(LLMRequest(messages="hi"))
        assert "x-api-key" not in route.calls.last.request.headers

    def test_empty_key_disables_env(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-from-env-0000000")
        route = respx_mock.post(URL).respond(json=message_body())
        client(api_key="").complete(LLMRequest(messages="hi"))
        assert "x-api-key" not in route.calls.last.request.headers

    def test_extra_headers(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=message_body())
        llm = client(headers={"anthropic-beta": "tools-2024", "anthropic-version": "2099-01-01"})
        llm.complete(LLMRequest(messages="hi"))
        headers = route.calls.last.request.headers
        assert headers["anthropic-beta"] == "tools-2024"
        assert headers["anthropic-version"] == "2099-01-01"  # caller may override


class TestPayload:
    def test_minimal_uses_default_max_tokens(self) -> None:
        assert client().build_payload(LLMRequest(messages="hi")) == {
            "model": "claude-test",
            "max_tokens": 1024,
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_client_max_tokens_and_request_overrides(self) -> None:
        llm = client(max_tokens=256)
        assert llm.build_payload(LLMRequest(messages="hi"))["max_tokens"] == 256
        assert llm.build_payload(LLMRequest(messages="hi", max_tokens=8))["max_tokens"] == 8

    def test_request_model_wins(self) -> None:
        payload = client().build_payload(LLMRequest(messages="hi", model="other"))
        assert payload["model"] == "other"

    def test_missing_model(self) -> None:
        with pytest.raises(LLMError, match="no model"):
            client(model=None).build_payload(LLMRequest(messages="hi"))

    def test_only_system_messages(self) -> None:
        with pytest.raises(LLMError, match="non-system message"):
            client().build_payload(LLMRequest(messages=[Message.system("alone")]))

    def test_system_messages_to_top_level(self) -> None:
        request = LLMRequest(
            messages=[
                Message.system("be brief"),
                Message.user("hi"),
                Message.system("and polite"),
            ],
            temperature=0.0,
        )
        assert client().build_payload(request) == {
            "model": "claude-test",
            "max_tokens": 1024,
            "system": "be brief\n\nand polite",
            "messages": [{"role": "user", "content": "hi"}],
            "temperature": 0.0,
        }

    def test_tools(self) -> None:
        request = LLMRequest(messages="weather?", tools=[WEATHER, ToolSpec(name="noop")])
        assert client().build_payload(request)["tools"] == [
            {
                "name": "get_weather",
                "description": "Current weather for a city",
                "input_schema": WEATHER.parameters,
            },
            {"name": "noop", "input_schema": {"type": "object", "properties": {}}},
        ]

    def test_tool_round_trip_messages(self) -> None:
        calls = [
            ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Rome"}),
            ToolCall(id="toolu_2", name="get_weather", arguments={"city": "Paris"}),
        ]
        request = LLMRequest(
            messages=[
                Message.system("use tools"),
                Message.user("weather in Rome and Paris?"),
                Message.assistant("Checking both.", tool_calls=calls),
                Message.tool('{"temp_c": 21}', tool_call_id="toolu_1"),
                Message.tool('{"temp_c": 17}', tool_call_id="toolu_2"),
                Message.user("and in Fahrenheit?"),
                Message.assistant(tool_calls=[ToolCall(id="toolu_3", name="noop")]),
                Message.tool("done", tool_call_id="toolu_3"),
                Message.assistant("Rome 70F, Paris 63F."),
                Message.user("thanks"),
            ]
        )
        payload = client().build_payload(request)
        assert payload["system"] == "use tools"
        assert payload["messages"] == [
            {"role": "user", "content": "weather in Rome and Paris?"},
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": "Checking both."},
                    {
                        "type": "tool_use",
                        "id": "toolu_1",
                        "name": "get_weather",
                        "input": {"city": "Rome"},
                    },
                    {
                        "type": "tool_use",
                        "id": "toolu_2",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    },
                ],
            },
            {
                "role": "user",
                "content": [
                    {"type": "tool_result", "tool_use_id": "toolu_1", "content": '{"temp_c": 21}'},
                    {"type": "tool_result", "tool_use_id": "toolu_2", "content": '{"temp_c": 17}'},
                    {"type": "text", "text": "and in Fahrenheit?"},
                ],
            },
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "toolu_3", "name": "noop", "input": {}}],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_3", "content": "done"}],
            },
            {"role": "assistant", "content": "Rome 70F, Paris 63F."},
            {"role": "user", "content": "thanks"},
        ]

    def test_consecutive_user_messages_stay_separate(self) -> None:
        request = LLMRequest(messages=[Message.user("a"), Message.user("b")])
        assert client().build_payload(request)["messages"] == [
            {"role": "user", "content": "a"},
            {"role": "user", "content": "b"},
        ]

    def test_empty_assistant_message(self) -> None:
        request = LLMRequest(messages=[Message.user("a"), Message.assistant()])
        assert client().build_payload(request)["messages"][1] == {
            "role": "assistant",
            "content": "",
        }

    def test_json_object_format_goes_to_system(self) -> None:
        request = LLMRequest(
            messages=[Message.system("be brief"), Message.user("hi")],
            response_format={"type": "json_object"},
        )
        payload = client().build_payload(request)
        assert "response_format" not in payload
        system = payload["system"]
        assert system.startswith("be brief\n\n")
        assert "valid JSON object" in system

    def test_json_schema_format_includes_schema(self) -> None:
        schema = {"type": "object", "properties": {"answer": {"type": "integer"}}}
        request = LLMRequest(
            messages="hi",
            response_format={"type": "json_schema", "json_schema": {"name": "a", "schema": schema}},
        )
        system = client().build_payload(request)["system"]
        assert "valid JSON object" in system
        assert json.dumps(schema, sort_keys=True) in system

    def test_json_schema_without_schema(self) -> None:
        request = LLMRequest(messages="hi", response_format={"type": "json_schema"})
        system = client().build_payload(request)["system"]
        assert "valid JSON object" in system
        assert "JSON Schema" not in system

    def test_text_format_ignored(self) -> None:
        request = LLMRequest(messages="hi", response_format={"type": "text"})
        assert "system" not in client().build_payload(request)

    def test_unsupported_format(self) -> None:
        request = LLMRequest(messages="hi", response_format={"type": "xml"})
        with pytest.raises(LLMError, match="unsupported response_format"):
            client().build_payload(request)


class TestComplete:
    def test_text_response(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=message_body(text("Hello!")))
        response = client().complete(LLMRequest(messages="hi"))
        assert response.content == "Hello!"
        assert response.tool_calls == []
        assert response.model == "claude-test-20250101"
        assert response.usage == Usage(input_tokens=12, output_tokens=3, cost_usd=None)
        assert response.raw == message_body(text("Hello!"))
        assert sent(route)["messages"] == [{"role": "user", "content": "hi"}]

    def test_tool_use_parsed(self, respx_mock: respx.Router) -> None:
        body = message_body(
            text("Let me check."),
            tool_use("toolu_1", "get_weather", {"city": "Rome"}),
            tool_use(None, "noop", None),
            stop_reason="tool_use",
        )
        respx_mock.post(URL).respond(json=body)
        response = client().complete(LLMRequest(messages="weather?", tools=[WEATHER]))
        assert response.content == "Let me check."
        assert response.tool_calls == [
            ToolCall(id="toolu_1", name="get_weather", arguments={"city": "Rome"}),
            ToolCall(id="toolu_2", name="noop", arguments={}),  # id synthesised from the index
        ]
        assert response.to_message().tool_calls == response.tool_calls

    def test_json_mode(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=message_body(text('{"answer": 42}')))
        request = LLMRequest(messages="answer in JSON", response_format={"type": "json_object"})
        response = client().complete(request)
        assert json.loads(response.content or "") == {"answer": 42}
        assert "valid JSON object" in sent(route)["system"]

    def test_acomplete(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=message_body(text("async hi")))
        response = asyncio.run(client().acomplete(LLMRequest(messages="hi")))
        assert response.content == "async hi"
        assert response.usage is not None
        assert route.calls.last.request.headers["x-api-key"] == SECRET

    def test_retries_overloaded_then_succeeds(self, respx_mock: respx.Router) -> None:
        delays: list[float] = []
        route = respx_mock.post(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "1"}),
                httpx.Response(529, json={"type": "error", "error": {"type": "overloaded_error"}}),
                httpx.Response(200, json=message_body()),
            ]
        )
        retry = RetryPolicy(initial_delay=0.5, jitter=lambda d: d, sleep=delays.append)
        assert client(retry_policy=retry).complete(LLMRequest(messages="hi")).content == "Hello!"
        assert route.call_count == 3
        assert delays == [1.0, 1.0]

    def test_rate_limit_exhausted(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(
            429, json={"type": "error", "error": {"type": "rate_limit_error", "message": "slow"}}
        )
        with pytest.raises(LLMRateLimitError, match="slow"):
            client().complete(LLMRequest(messages="hi"))

    def test_http_error_has_status_and_no_key(
        self, respx_mock: respx.Router, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx_mock.post(URL).respond(
            400,
            json={
                "type": "error",
                "error": {"type": "invalid_request_error", "message": "max_tokens: too large"},
            },
        )
        with (
            caplog.at_level(logging.DEBUG, logger="agentlens"),
            pytest.raises(LLMHTTPError, match="max_tokens: too large") as info,
        ):
            client().complete(LLMRequest(messages="hi"))
        assert info.value.status_code == 400
        assert SECRET not in str(info.value)
        assert SECRET not in info.value.body
        assert SECRET not in caplog.text

    def test_key_never_in_raw(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=message_body())
        response = client().complete(LLMRequest(messages="hi"))
        assert SECRET not in response.model_dump_json()

    def test_context_manager_closes(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=message_body())
        with client() as llm:
            llm.complete(LLMRequest(messages="hi"))
            pool = llm._http._client
        assert pool is not None
        assert pool.is_closed

    def test_traced_with_wrap_llm(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=message_body(text("traced")))
        llm = agentlens.wrap_llm(client())
        with agentlens.run("adapter-test") as trace:
            llm.complete(LLMRequest(messages="hi"))
        (span,) = trace.spans
        assert span.kind == SpanKind.LLM_CALL
        assert span.attributes["provider"] == "anthropic"
        assert span.attributes["response_model"] == "claude-test-20250101"
        assert span.usage == Usage(input_tokens=12, output_tokens=3)
        assert SECRET not in trace.model_dump_json()


class TestParseResponse:
    def test_missing_usage(self) -> None:
        body = message_body()
        del body["usage"]
        assert parse_response(body).usage is None

    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            ({"input_tokens": 5}, Usage(input_tokens=5, output_tokens=0)),
            ({"output_tokens": 7}, Usage(input_tokens=0, output_tokens=7)),
            (
                {
                    "input_tokens": 5,
                    "cache_creation_input_tokens": 100,
                    "cache_read_input_tokens": 20,
                    "output_tokens": 2,
                },
                Usage(input_tokens=125, output_tokens=2),
            ),
            ({"service_tier": "standard"}, None),
        ],
    )
    def test_usage(self, usage: dict[str, Any], expected: Usage | None) -> None:
        body = message_body()
        body["usage"] = usage
        assert parse_response(body).usage == expected

    def test_multiple_text_blocks_and_unknown_blocks(self) -> None:
        body = message_body(
            {"type": "thinking", "thinking": "hmm", "signature": "x"},
            text("Hello, "),
            {"type": "text"},  # malformed text block: skipped
            text("world"),
        )
        assert parse_response(body).content == "Hello, world"

    def test_tool_only_response_has_no_content(self) -> None:
        body = message_body(tool_use("toolu_1", "noop", {}))
        response = parse_response(body)
        assert response.content is None
        assert response.tool_calls == [ToolCall(id="toolu_1", name="noop")]

    def test_empty_content_and_non_string_model(self) -> None:
        body = message_body()
        body["content"] = []
        body["model"] = None
        response = parse_response(body)
        assert response.content is None
        assert response.tool_calls == []
        assert response.model is None

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({}, "no content list"),
            ({"content": "text"}, "no content list"),
            ({"content": ["oops"]}, "not an object"),
            ({"content": [{"type": "tool_use", "id": "t", "input": {}}]}, "has no name"),
            ({"content": [tool_use("t", "f", [1, 2])]}, "must be a JSON object"),
        ],
    )
    def test_malformed(self, body: dict[str, Any], match: str) -> None:
        with pytest.raises(LLMResponseError, match=match):
            parse_response(body)

    def test_returns_llm_response(self) -> None:
        assert isinstance(parse_response(message_body()), LLMResponse)


LIVE_MODEL = os.environ.get("AGENTLENS_LIVE_ANTHROPIC_MODEL")


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_MODEL, reason="set AGENTLENS_LIVE_ANTHROPIC_MODEL (and ANTHROPIC_API_KEY)"
)
def test_live_round_trip() -> None:
    """Real call against the Anthropic API."""
    with AnthropicClient(model=LIVE_MODEL) as llm:
        response = llm.complete(
            LLMRequest(
                messages=[
                    Message.system("You answer with exactly one word."),
                    Message.user("Reply with the single word: pong"),
                ],
                max_tokens=10,
                temperature=0,
            )
        )
    assert response.content
    assert "pong" in response.content.lower()
    assert response.usage is not None
