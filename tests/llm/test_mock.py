"""Tests for the scripted MockLLM client."""

from __future__ import annotations

import asyncio
import re
import threading

import pytest

from agentlens.llm import (
    LLMRequest,
    LLMResponse,
    Message,
    MockExhaustedError,
    MockLLM,
    MockLLMError,
    MockNoMatchError,
    ScriptItem,
    ToolCall,
    count_tokens,
)
from agentlens.schema import Usage

CALL = ToolCall(id="call_1", name="lookup_order", arguments={"order_id": "1234"})

REGEX_SCRIPT: dict[str | re.Pattern[str], ScriptItem] = {
    r"refund": "Refund issued.",
    re.compile(r"order \d+", re.IGNORECASE): CALL,
    r"": "catch-all",
}


def ask(text: str, **kwargs: object) -> LLMRequest:
    return LLMRequest.model_validate({"messages": text, **kwargs})


class TestCountTokens:
    @pytest.mark.parametrize(
        ("text", "expected"),
        [(None, 0), ("", 0), ("   ", 0), ("one", 1), ("one two\nthree\tfour", 4)],
    )
    def test_whitespace_words(self, text: str | None, expected: int) -> None:
        assert count_tokens(text) == expected


class TestListScript:
    def test_returns_items_in_order(self) -> None:
        llm = MockLLM(["first", "second"])
        assert llm.complete(ask("a")).content == "first"
        assert llm.complete(ask("b")).content == "second"
        assert llm.calls == 2
        assert [r.last_user_message for r in llm.requests] == ["a", "b"]

    def test_exhausted_raises(self) -> None:
        llm = MockLLM(["only"])
        llm.complete(ask("a"))
        with pytest.raises(MockExhaustedError, match="1 responses but call #2"):
            llm.complete(ask("b"))
        assert issubclass(MockExhaustedError, MockLLMError)

    def test_empty_script_raises_immediately(self) -> None:
        with pytest.raises(MockExhaustedError):
            MockLLM([]).complete(ask("a"))

    def test_cycle(self) -> None:
        llm = MockLLM(["a", "b"], cycle=True)
        assert [llm.complete(ask("x")).content for _ in range(5)] == ["a", "b", "a", "b", "a"]

    def test_empty_script_with_cycle_still_raises(self) -> None:
        with pytest.raises(MockExhaustedError):
            MockLLM([], cycle=True).complete(ask("a"))

    def test_default_after_exhaustion(self) -> None:
        llm = MockLLM(["a"], default="fallback")
        assert llm.complete(ask("x")).content == "a"
        assert llm.complete(ask("x")).content == "fallback"

    def test_reset_restarts(self) -> None:
        llm = MockLLM(["a", "b"])
        llm.complete(ask("x"))
        llm.reset()
        assert llm.calls == 0
        assert llm.complete(ask("x")).content == "a"

    def test_script_is_copied(self) -> None:
        items: list[str] = ["a"]
        llm = MockLLM(items)
        items.append("b")
        llm.complete(ask("x"))
        with pytest.raises(MockExhaustedError):
            llm.complete(ask("x"))

    def test_single_string_is_rejected(self) -> None:
        with pytest.raises(TypeError, match="not a single string"):
            MockLLM("hello")


class TestToolCalls:
    def test_tool_call_item(self) -> None:
        response = MockLLM([CALL]).complete(ask("where is 1234?"))
        assert response.content is None
        assert response.tool_calls == [CALL]

    def test_list_of_tool_calls_item(self) -> None:
        other = ToolCall(id="call_2", name="get_weather", arguments={})
        response = MockLLM([[CALL, other]]).complete(ask("x"))
        assert response.tool_calls == [CALL, other]

    def test_full_response_with_text_and_tool_call(self) -> None:
        item = LLMResponse(content="Let me check.", tool_calls=[CALL])
        response = MockLLM([item]).complete(ask("x"))
        assert response.content == "Let me check."
        assert response.tool_calls == [CALL]

    def test_agent_loop(self) -> None:
        llm = MockLLM([CALL, "Order 1234 has shipped."])
        messages = [Message.user("Where is order 1234?")]
        first = llm.complete(LLMRequest(messages=messages))
        messages.append(first.to_message())
        messages.append(Message.tool('{"status": "shipped"}', tool_call_id=first.tool_calls[0].id))
        final = llm.complete(LLMRequest(messages=messages))
        assert final.content == "Order 1234 has shipped."
        assert [m.role for m in llm.requests[1].messages] == ["user", "assistant", "tool"]


class TestCallableScript:
    def test_computes_from_request(self) -> None:
        llm = MockLLM(lambda request: (request.last_user_message or "").upper())
        assert llm.complete(ask("hello")).content == "HELLO"
        assert llm.complete(ask("again")).content == "AGAIN"

    def test_may_return_any_item(self) -> None:
        llm = MockLLM(lambda request: CALL if request.tools else LLMResponse(content="no tools"))
        assert llm.complete(ask("x")).content == "no tools"
        assert llm.complete(ask("x", tools=[{"name": "t"}])).tool_calls == [CALL]

    def test_invalid_return_type(self) -> None:
        llm = MockLLM(lambda request: 42)  # type: ignore[arg-type, return-value]
        with pytest.raises(TypeError, match="got int"):
            llm.complete(ask("x"))

    def test_list_with_non_tool_calls_is_invalid(self) -> None:
        with pytest.raises(TypeError, match="got list"):
            MockLLM([[CALL, "x"]]).complete(ask("x"))  # type: ignore[list-item]


class TestRegexScript:
    def test_first_matching_pattern_wins(self) -> None:
        llm = MockLLM(REGEX_SCRIPT)
        assert llm.complete(ask("I want a refund for order 12")).content == "Refund issued."
        assert llm.complete(ask("Where is ORDER 12?")).tool_calls == [CALL]
        assert llm.complete(ask("hello")).content == "catch-all"

    def test_matches_last_user_message_only(self) -> None:
        llm = MockLLM({r"refund": "Refund issued."}, default="no match")
        request = LLMRequest(
            messages=[
                Message.system("You handle refunds."),
                Message.user("refund please"),
                Message.assistant("Which order?"),
                Message.user("order 7"),
            ]
        )
        assert llm.complete(request).content == "no match"

    def test_no_match_without_default_raises(self) -> None:
        llm = MockLLM({r"refund": "ok"})
        with pytest.raises(MockNoMatchError, match="'hello'"):
            llm.complete(ask("hello"))

    def test_no_user_message_matches_empty_string(self) -> None:
        llm = MockLLM({r"^$": "empty"})
        assert llm.complete(LLMRequest(messages=[Message.system("sys")])).content == "empty"

    def test_empty_mapping_uses_default(self) -> None:
        assert MockLLM({}, default="d").complete(ask("x")).content == "d"
        with pytest.raises(MockNoMatchError):
            MockLLM({}).complete(ask("x"))


class TestUsageAndModel:
    def test_deterministic_usage_from_word_counts(self) -> None:
        llm = MockLLM(["three word answer"])
        request = LLMRequest(messages=[Message.system("be brief"), Message.user("hi there you")])
        response = llm.complete(request)
        assert response.usage == Usage(input_tokens=5, output_tokens=3, cost_usd=0.0)

    def test_tool_calls_count_towards_usage(self) -> None:
        request = LLMRequest(
            messages=[
                Message.user("where"),
                Message.assistant(tool_calls=[CALL]),
                Message.tool("shipped", tool_call_id="call_1"),
            ]
        )
        response = MockLLM([CALL]).complete(request)
        # 'lookup_order {"order_id": "1234"}' -> 3 words
        assert response.usage == Usage(input_tokens=5, output_tokens=3, cost_usd=0.0)

    def test_same_request_same_response(self) -> None:
        script: list[ScriptItem] = ["a b", CALL]
        first = [MockLLM(script).complete(ask("x y z")) for _ in range(2)]
        assert first[0] == first[1]

    def test_explicit_usage_is_kept(self) -> None:
        usage = Usage(input_tokens=100, output_tokens=7, cost_usd=0.5)
        response = MockLLM([LLMResponse(content="x", usage=usage)]).complete(ask("x"))
        assert response.usage == usage

    def test_model_resolution(self) -> None:
        assert MockLLM(["x"]).complete(ask("q")).model == "mock"
        assert MockLLM(["x"], model="mock-2").complete(ask("q")).model == "mock-2"
        assert MockLLM(["x"]).complete(ask("q", model="gpt-4o")).model == "gpt-4o"
        pinned = LLMResponse(content="x", model="from-script")
        assert MockLLM([pinned]).complete(ask("q", model="gpt-4o")).model == "from-script"

    def test_script_items_are_not_mutated(self) -> None:
        item = LLMResponse(content="x")
        MockLLM([item]).complete(ask("q"))
        assert item.usage is None
        assert item.model is None


class TestAsyncAndThreads:
    def test_acomplete(self) -> None:
        llm = MockLLM(["a", "b"])

        async def main() -> list[str | None]:
            first = await llm.acomplete(ask("x"))
            second = await llm.acomplete(ask("y"))
            return [first.content, second.content]

        assert asyncio.run(main()) == ["a", "b"]

    def test_concurrent_calls_each_get_one_item(self) -> None:
        n = 200
        llm = MockLLM([str(i) for i in range(n)])
        results: list[str | None] = []
        lock = threading.Lock()

        def worker() -> None:
            for _ in range(n // 8):
                content = llm.complete(ask("x")).content
                with lock:
                    results.append(content)

        threads = [threading.Thread(target=worker) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sorted(results, key=lambda s: int(s or 0)) == [str(i) for i in range(n)]
        assert llm.calls == n
