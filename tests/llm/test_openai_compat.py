"""Tests for the OpenAI-compatible adapter (HTTP mocked with respx)."""

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
    DEFAULT_OPENAI_BASE_URL,
    LLMClient,
    LLMError,
    LLMHTTPError,
    LLMRequest,
    LLMResponse,
    LLMResponseError,
    Message,
    OpenAICompatClient,
    RetryPolicy,
    ToolCall,
    ToolSpec,
)
from agentlens.llm.openai_compat import parse_response
from agentlens.schema import SpanKind, Usage

BASE = "http://llm.test/v1"
URL = f"{BASE}/chat/completions"
SECRET = "sk-test-secret-key-123456"

WEATHER = ToolSpec(
    name="get_weather",
    description="Current weather for a city",
    parameters={"type": "object", "properties": {"city": {"type": "string"}}},
)


def completion(
    content: str | None = "Hello!",
    *,
    tool_calls: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    model: str = "test-model-0613",
) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    body: dict[str, Any] = {
        "id": "chatcmpl-1",
        "object": "chat.completion",
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
    }
    body["usage"] = usage or {"prompt_tokens": 12, "completion_tokens": 3, "total_tokens": 15}
    return body


def fn_call(call_id: str | None, name: str, arguments: Any) -> dict[str, Any]:
    call: dict[str, Any] = {"type": "function", "function": {"name": name}}
    if call_id is not None:
        call["id"] = call_id
    if arguments is not None:
        call["function"]["arguments"] = arguments
    return call


def client(**kwargs: Any) -> OpenAICompatClient:
    kwargs.setdefault("model", "test-model")
    kwargs.setdefault("base_url", BASE)
    kwargs.setdefault("api_key", SECRET)
    kwargs.setdefault(
        "retry_policy",
        RetryPolicy(max_retries=2, jitter=lambda d: d, sleep=lambda _: None),
    )
    return OpenAICompatClient(**kwargs)


def sent(route: respx.Route) -> dict[str, Any]:
    body: dict[str, Any] = json.loads(route.calls.last.request.content)
    return body


class TestConstruction:
    def test_is_llm_client(self) -> None:
        assert isinstance(client(), LLMClient)

    def test_defaults(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        llm = OpenAICompatClient()
        assert llm.base_url == DEFAULT_OPENAI_BASE_URL
        assert llm.url == "https://api.openai.com/v1/chat/completions"
        assert llm.provider == "openai"
        assert llm.model is None
        assert llm.retry_policy.max_retries == 2

    def test_base_url_trailing_slash_and_provider(self) -> None:
        llm = OpenAICompatClient(base_url="http://localhost:11434/v1/")
        assert llm.url == "http://localhost:11434/v1/chat/completions"
        assert llm.provider == "openai_compat"
        assert OpenAICompatClient(base_url=BASE, provider="ollama").provider == "ollama"

    def test_max_retries_shortcut(self) -> None:
        assert OpenAICompatClient(max_retries=5).retry_policy.max_retries == 5

    def test_repr_hides_key(self) -> None:
        text = repr(client())
        assert SECRET not in text
        assert "test-model" in text
        assert BASE in text


class TestAuth:
    def test_explicit_key(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=completion())
        client().complete(LLMRequest(messages="hi"))
        assert route.calls.last.request.headers["authorization"] == f"Bearer {SECRET}"

    def test_env_fallback(self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-000000000")
        route = respx_mock.post(URL).respond(json=completion())
        client(api_key=None).complete(LLMRequest(messages="hi"))
        assert route.calls.last.request.headers["authorization"] == "Bearer sk-from-env-000000000"

    def test_no_key_for_local_servers(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        route = respx_mock.post(URL).respond(json=completion())
        client(api_key=None).complete(LLMRequest(messages="hi"))
        assert "authorization" not in route.calls.last.request.headers

    def test_empty_key_disables_env(
        self, respx_mock: respx.Router, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setenv("OPENAI_API_KEY", "sk-from-env-000000000")
        route = respx_mock.post(URL).respond(json=completion())
        client(api_key="").complete(LLMRequest(messages="hi"))
        assert "authorization" not in route.calls.last.request.headers

    def test_extra_headers(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=completion())
        client(headers={"OpenAI-Organization": "org-1"}).complete(LLMRequest(messages="hi"))
        assert route.calls.last.request.headers["openai-organization"] == "org-1"


class TestPayload:
    def test_minimal(self) -> None:
        assert client().build_payload(LLMRequest(messages="hi")) == {
            "model": "test-model",
            "messages": [{"role": "user", "content": "hi"}],
        }

    def test_request_model_wins(self) -> None:
        payload = client().build_payload(LLMRequest(messages="hi", model="other"))
        assert payload["model"] == "other"

    def test_missing_model(self) -> None:
        with pytest.raises(LLMError, match="no model"):
            client(model=None).build_payload(LLMRequest(messages="hi"))

    def test_all_options(self) -> None:
        request = LLMRequest(
            messages=[Message.system("be brief"), Message.user("weather?")],
            temperature=0.0,
            max_tokens=64,
            tools=[WEATHER, ToolSpec(name="noop")],
            response_format={"type": "json_object"},
        )
        assert client().build_payload(request) == {
            "model": "test-model",
            "messages": [
                {"role": "system", "content": "be brief"},
                {"role": "user", "content": "weather?"},
            ],
            "temperature": 0.0,
            "max_tokens": 64,
            "tools": [
                {
                    "type": "function",
                    "function": {
                        "name": "get_weather",
                        "description": "Current weather for a city",
                        "parameters": WEATHER.parameters,
                    },
                },
                {
                    "type": "function",
                    "function": {
                        "name": "noop",
                        "parameters": {"type": "object", "properties": {}},
                    },
                },
            ],
            "response_format": {"type": "json_object"},
        }

    def test_tool_round_trip_messages(self) -> None:
        call = ToolCall(id="call_9", name="get_weather", arguments={"city": "Rome"})
        request = LLMRequest(
            messages=[
                Message.user("weather in Rome?"),
                Message.assistant(tool_calls=[call]),
                Message.tool('{"temp_c": 21}', tool_call_id="call_9"),
            ]
        )
        messages = client().build_payload(request)["messages"]
        assert messages[1] == {
            "role": "assistant",
            "content": None,
            "tool_calls": [
                {
                    "id": "call_9",
                    "type": "function",
                    "function": {"name": "get_weather", "arguments": '{"city": "Rome"}'},
                }
            ],
        }
        assert messages[2] == {
            "role": "tool",
            "content": '{"temp_c": 21}',
            "tool_call_id": "call_9",
        }

    def test_json_schema_response_format_passthrough(self) -> None:
        schema_format: dict[str, Any] = {
            "type": "json_schema",
            "json_schema": {"name": "answer", "schema": {"type": "object"}, "strict": True},
        }
        request = LLMRequest(messages="hi", response_format=schema_format)
        assert client().build_payload(request)["response_format"] == schema_format


class TestComplete:
    def test_text_response(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=completion("Hello!"))
        response = client().complete(LLMRequest(messages="hi"))
        assert response.content == "Hello!"
        assert response.tool_calls == []
        assert response.model == "test-model-0613"
        assert response.usage == Usage(input_tokens=12, output_tokens=3, cost_usd=None)
        assert response.raw == completion("Hello!")
        assert sent(route)["messages"] == [{"role": "user", "content": "hi"}]

    def test_tool_calls_parsed(self, respx_mock: respx.Router) -> None:
        body = completion(
            None,
            tool_calls=[
                fn_call("call_1", "get_weather", '{"city": "Rome"}'),
                fn_call("call_b", "get_weather", {"city": "Paris"}),  # some servers send objects
                fn_call(None, "noop", ""),
            ],
        )
        respx_mock.post(URL).respond(json=body)
        response = client().complete(LLMRequest(messages="weather?", tools=[WEATHER]))
        assert response.content is None
        assert response.tool_calls == [
            ToolCall(id="call_1", name="get_weather", arguments={"city": "Rome"}),
            ToolCall(id="call_b", name="get_weather", arguments={"city": "Paris"}),
            ToolCall(id="call_2", name="noop", arguments={}),  # id synthesised from the index
        ]
        assert response.to_message().tool_calls == response.tool_calls

    def test_json_mode(self, respx_mock: respx.Router) -> None:
        route = respx_mock.post(URL).respond(json=completion('{"answer": 42}'))
        request = LLMRequest(messages="answer in JSON", response_format={"type": "json_object"})
        response = client().complete(request)
        assert json.loads(response.content or "") == {"answer": 42}
        assert sent(route)["response_format"] == {"type": "json_object"}

    def test_acomplete(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=completion("async hi"))
        response = asyncio.run(client().acomplete(LLMRequest(messages="hi")))
        assert response.content == "async hi"
        assert response.usage is not None

    def test_retries_then_succeeds(self, respx_mock: respx.Router) -> None:
        delays: list[float] = []
        route = respx_mock.post(URL).mock(
            side_effect=[
                httpx.Response(429, headers={"retry-after": "1"}),
                httpx.Response(503),
                httpx.Response(200, json=completion()),
            ]
        )
        retry = RetryPolicy(initial_delay=0.5, jitter=lambda d: d, sleep=delays.append)
        assert client(retry_policy=retry).complete(LLMRequest(messages="hi")).content == "Hello!"
        assert route.call_count == 3
        assert delays == [1.0, 1.0]

    def test_async_retries(self, respx_mock: respx.Router) -> None:
        delays: list[float] = []

        async def fake_sleep(delay: float) -> None:
            delays.append(delay)

        respx_mock.post(URL).mock(
            side_effect=[httpx.ConnectError("down"), httpx.Response(200, json=completion())]
        )
        retry = RetryPolicy(jitter=lambda d: d, async_sleep=fake_sleep)
        response = asyncio.run(client(retry_policy=retry).acomplete(LLMRequest(messages="hi")))
        assert response.content == "Hello!"
        assert delays == [0.5]

    def test_http_error_has_status_and_no_key(
        self, respx_mock: respx.Router, caplog: pytest.LogCaptureFixture
    ) -> None:
        respx_mock.post(URL).respond(
            400, json={"error": {"message": "Unknown model", "type": "invalid_request_error"}}
        )
        with (
            caplog.at_level(logging.DEBUG, logger="agentlens"),
            pytest.raises(LLMHTTPError, match="Unknown model") as info,
        ):
            client().complete(LLMRequest(messages="hi"))
        assert info.value.status_code == 400
        assert SECRET not in str(info.value)
        assert SECRET not in info.value.body
        assert SECRET not in caplog.text

    def test_key_never_in_raw(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=completion())
        response = client().complete(LLMRequest(messages="hi"))
        assert SECRET not in response.model_dump_json()

    def test_context_manager_closes(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=completion())
        with client() as llm:
            llm.complete(LLMRequest(messages="hi"))
            pool = llm._http._client
        assert pool is not None
        assert pool.is_closed

    def test_traced_with_wrap_llm(self, respx_mock: respx.Router) -> None:
        respx_mock.post(URL).respond(json=completion("traced"))
        llm = agentlens.wrap_llm(client(provider="ollama"))
        with agentlens.run("adapter-test") as trace:
            llm.complete(LLMRequest(messages="hi"))
        (span,) = trace.spans
        assert span.kind == SpanKind.LLM_CALL
        assert span.attributes["provider"] == "ollama"
        assert span.attributes["response_model"] == "test-model-0613"
        assert span.usage == Usage(input_tokens=12, output_tokens=3)
        assert SECRET not in trace.model_dump_json()


class TestParseResponse:
    def test_missing_usage(self) -> None:
        body = completion()
        del body["usage"]
        assert parse_response(body).usage is None

    @pytest.mark.parametrize(
        ("usage", "expected"),
        [
            ({"prompt_tokens": 5}, Usage(input_tokens=5, output_tokens=0)),
            ({"completion_tokens": 7}, Usage(input_tokens=0, output_tokens=7)),
            ({"total_tokens": 9}, None),
        ],
    )
    def test_partial_usage(self, usage: dict[str, Any], expected: Usage | None) -> None:
        body = completion()
        body["usage"] = usage
        assert parse_response(body).usage == expected

    def test_non_string_model_and_content(self) -> None:
        body = completion()
        body["model"] = None
        body["choices"][0]["message"]["content"] = ["parts"]
        response = parse_response(body)
        assert response.model is None
        assert response.content is None

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({}, "no choices"),
            ({"choices": []}, "no choices"),
            ({"choices": ["x"]}, "no choices"),
            ({"choices": [{"index": 0}]}, "no message"),
        ],
    )
    def test_malformed(self, body: dict[str, Any], match: str) -> None:
        with pytest.raises(LLMResponseError, match=match):
            parse_response(body)

    @pytest.mark.parametrize(
        ("tool_calls", "match"),
        [
            ({"not": "a list"}, "must be a list"),
            (["oops"], "no function name"),
            ([{"id": "c", "function": {"arguments": "{}"}}], "no function name"),
            ([fn_call("c", "f", "{not json")], "not valid JSON"),
            ([fn_call("c", "f", "[1, 2]")], "must be a JSON object"),
        ],
    )
    def test_bad_tool_calls(self, tool_calls: Any, match: str) -> None:
        body = completion(None, tool_calls=tool_calls)
        with pytest.raises(LLMResponseError, match=match):
            parse_response(body)

    def test_null_arguments_and_empty_tool_calls(self) -> None:
        body = completion("ok", tool_calls=[])
        assert parse_response(body).tool_calls == []
        body = completion(None, tool_calls=[fn_call("c1", "noop", None)])
        assert parse_response(body).tool_calls == [ToolCall(id="c1", name="noop")]

    def test_returns_llm_response(self) -> None:
        assert isinstance(parse_response(completion()), LLMResponse)


LIVE_MODEL = os.environ.get("AGENTLENS_LIVE_OPENAI_MODEL")


@pytest.mark.live
@pytest.mark.skipif(
    not LIVE_MODEL, reason="set AGENTLENS_LIVE_OPENAI_MODEL (and OPENAI_API_KEY/base URL)"
)
def test_live_round_trip() -> None:
    """Real call; optional ``AGENTLENS_LIVE_OPENAI_BASE_URL`` targets Ollama/vLLM/LM Studio."""
    base_url = os.environ.get("AGENTLENS_LIVE_OPENAI_BASE_URL", DEFAULT_OPENAI_BASE_URL)
    with OpenAICompatClient(model=LIVE_MODEL, base_url=base_url) as llm:
        response = llm.complete(
            LLMRequest(messages="Reply with the single word: pong", max_tokens=10, temperature=0)
        )
    assert response.content
    assert "pong" in response.content.lower()
