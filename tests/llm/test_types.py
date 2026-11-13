"""Tests for the provider-neutral LLM models and the LLMClient protocol."""

from __future__ import annotations

import json

import pytest
from pydantic import ValidationError

from agentlens.llm import (
    LLMClient,
    LLMRequest,
    LLMResponse,
    Message,
    MockLLM,
    ToolCall,
    ToolSpec,
)
from agentlens.schema import Usage

CALL = ToolCall(id="call_1", name="lookup_order", arguments={"order_id": "1234"})


class TestMessage:
    def test_constructors(self) -> None:
        assert Message.system("be nice") == Message(role="system", content="be nice")
        assert Message.user("hi").role == "user"
        assistant = Message.assistant(tool_calls=[CALL])
        assert assistant.content is None
        assert assistant.tool_calls == [CALL]
        assert Message.assistant("ok").tool_calls == []
        tool = Message.tool('{"status": "shipped"}', tool_call_id="call_1")
        assert tool.tool_call_id == "call_1"

    def test_tool_calls_only_on_assistant(self) -> None:
        with pytest.raises(ValidationError, match="only assistant messages"):
            Message(role="user", content="hi", tool_calls=[CALL])

    def test_tool_message_needs_tool_call_id(self) -> None:
        with pytest.raises(ValidationError, match="tool_call_id is required"):
            Message(role="tool", content="result")

    def test_tool_call_id_only_on_tool_messages(self) -> None:
        with pytest.raises(ValidationError, match="tool_call_id is required"):
            Message(role="user", content="hi", tool_call_id="call_1")

    @pytest.mark.parametrize("role", ["system", "user", "tool"])
    def test_content_required_except_assistant(self, role: str) -> None:
        extra = {"tool_call_id": "c"} if role == "tool" else {}
        with pytest.raises(ValidationError, match="need content"):
            Message.model_validate({"role": role, **extra})

    def test_unknown_role_rejected(self) -> None:
        with pytest.raises(ValidationError):
            Message.model_validate({"role": "developer", "content": "x"})


class TestToolSpec:
    def test_defaults_to_empty_object_schema(self) -> None:
        spec = ToolSpec(name="ping")
        assert spec.description == ""
        assert spec.parameters == {"type": "object", "properties": {}}

    def test_name_required_non_empty(self) -> None:
        with pytest.raises(ValidationError):
            ToolSpec(name="")


class TestLLMRequest:
    def test_string_shorthand_is_a_user_message(self) -> None:
        request = LLMRequest.model_validate({"messages": "hello"})
        assert request.messages == [Message.user("hello")]

    def test_defaults(self) -> None:
        request = LLMRequest(messages=[Message.user("hi")])
        assert request.model is None
        assert request.temperature is None
        assert request.max_tokens is None
        assert request.tools == []
        assert request.response_format is None

    def test_validation(self) -> None:
        with pytest.raises(ValidationError):
            LLMRequest(messages=[])
        with pytest.raises(ValidationError):
            LLMRequest(messages=[Message.user("hi")], temperature=-0.1)
        with pytest.raises(ValidationError):
            LLMRequest(messages=[Message.user("hi")], max_tokens=0)
        with pytest.raises(ValidationError):
            LLMRequest.model_validate({"messages": "hi", "top_k": 3})

    def test_last_user_message(self) -> None:
        request = LLMRequest(
            messages=[
                Message.system("sys"),
                Message.user("first"),
                Message.assistant(tool_calls=[CALL]),
                Message.tool("{}", tool_call_id="call_1"),
                Message.user("second"),
                Message.assistant("done"),
            ]
        )
        assert request.last_user_message == "second"
        assert LLMRequest(messages=[Message.system("only")]).last_user_message is None

    def test_json_round_trip(self) -> None:
        request = LLMRequest(
            messages=[Message.user("hi"), Message.assistant(tool_calls=[CALL])],
            model="gpt-4o-mini",
            temperature=0,
            max_tokens=64,
            tools=[ToolSpec(name="lookup_order", parameters={"type": "object"})],
            response_format={"type": "json_object"},
        )
        assert LLMRequest.model_validate_json(request.model_dump_json()) == request

    def test_canonical_json_is_order_independent(self) -> None:
        a = LLMRequest(
            messages=[Message.user("hi")],
            tools=[ToolSpec(name="t", parameters={"type": "object", "properties": {"x": {}}})],
            response_format={"type": "json_object", "strict": True},
        )
        b = LLMRequest(
            messages=[Message.user("hi")],
            tools=[ToolSpec(name="t", parameters={"properties": {"x": {}}, "type": "object"})],
            response_format={"strict": True, "type": "json_object"},
        )
        assert a.canonical_json() == b.canonical_json()
        assert a.fingerprint() == b.fingerprint()
        assert len(a.fingerprint()) == 64
        assert " " not in a.canonical_json()
        assert json.loads(a.canonical_json())["messages"][0]["content"] == "hi"

    def test_fingerprint_changes_with_content(self) -> None:
        assert (
            LLMRequest(messages=[Message.user("a")]).fingerprint()
            != LLMRequest(messages=[Message.user("b")]).fingerprint()
        )
        unicode = LLMRequest(messages=[Message.user("caffè")])
        assert "caffè" in unicode.canonical_json()


class TestLLMResponse:
    def test_defaults(self) -> None:
        response = LLMResponse()
        assert response.content is None
        assert response.tool_calls == []
        assert response.usage is None
        assert response.raw is None
        assert response.model is None

    def test_to_message(self) -> None:
        response = LLMResponse(content="calling", tool_calls=[CALL], usage=Usage(input_tokens=1))
        assert response.to_message() == Message.assistant("calling", [CALL])

    def test_json_round_trip(self) -> None:
        response = LLMResponse(
            content=None,
            tool_calls=[CALL],
            usage=Usage(input_tokens=3, output_tokens=2, cost_usd=0.001),
            raw={"id": "chatcmpl-1", "choices": []},
            model="gpt-4o-mini-2024-07-18",
        )
        assert LLMResponse.model_validate_json(response.model_dump_json()) == response


class TestProtocol:
    def test_mock_is_an_llm_client(self) -> None:
        assert isinstance(MockLLM(["x"]), LLMClient)

    def test_structural_typing(self) -> None:
        class Echo:
            def complete(self, request: LLMRequest) -> LLMResponse:
                return LLMResponse(content=request.last_user_message)

            async def acomplete(self, request: LLMRequest) -> LLMResponse:
                return self.complete(request)

        client: LLMClient = Echo()
        assert isinstance(client, LLMClient)
        assert client.complete(LLMRequest(messages=[Message.user("ping")])).content == "ping"
        assert not isinstance(object(), LLMClient)
