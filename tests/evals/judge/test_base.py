"""The shared judge flow of :class:`LLMJudge`: request, parsing, retry, cache, client resolution."""

from __future__ import annotations

import asyncio

import pytest
from pydantic import JsonValue

from agentlens.evals import EvaluatorConfigError, arun_evaluator, run_evaluator
from agentlens.evals.judge import (
    JSON_RESPONSE_FORMAT,
    JUDGE_MODEL_ENV,
    JudgeParseError,
    JudgePrompt,
    JudgeRun,
    LLMJudge,
    MemoryJudgeCache,
    judge_cache_key,
    value_text,
)
from agentlens.evals.judge.base import _add_usage
from agentlens.llm import LLMConfigError, LLMRequest, LLMResponse, MockLLM
from agentlens.schema import Case, EvalResult, Trace, Usage

TRACE = Trace(name="agent", input="q", output="the answer")
CASE = Case(id="c1", input="q")


class EchoJudge(LLMJudge[int]):
    """Minimal judge: the verdict is the integer under ``n``."""

    name = "echo_judge"

    def build_prompt(self, trace: Trace, case: Case) -> JudgePrompt:
        output = value_text(trace.output)
        return JudgePrompt(system="sys", user=f"judge {output}", output=output)

    def parse_reply(self, data: dict[str, JsonValue]) -> int:
        value = data.get("n")
        if not isinstance(value, int):
            raise JudgeParseError("'n' must be an integer")
        return value

    def make_result(self, verdict: int, run: JudgeRun[int]) -> EvalResult:
        return EvalResult(evaluator=self.name, score=1.0, passed=verdict > 0, details=run.details())


class NoUsageClient:
    """A client whose responses carry no usage (and a provider/model to label it)."""

    provider = "acme"
    model = "judge-1"

    def __init__(self, replies: list[LLMResponse]) -> None:
        self.replies = replies
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return self.replies[len(self.requests) - 1]

    async def acomplete(self, request: LLMRequest) -> LLMResponse:
        return self.complete(request)


def test_value_text() -> None:
    assert value_text(None) == "(none)"
    assert value_text("é") == "é"
    assert value_text({"b": 1, "a": "é"}) == '{"a": "é", "b": 1}'
    assert value_text([1, True]) == "[1, true]"


def test_judge_prompt() -> None:
    prompt = JudgePrompt(system="S", user="U", output="O")
    assert prompt.text == "S\n\nU"
    assert [(m.role, m.content) for m in prompt.messages()] == [("system", "S"), ("user", "U")]


def test_request_is_temperature_zero_json_mode() -> None:
    llm = MockLLM(['{"n": 1}'])
    judge = EchoJudge(client=llm, cache=False, max_tokens=64)
    result = judge.evaluate(TRACE, CASE)
    assert result.passed
    (request,) = llm.requests
    assert request.temperature == 0.0
    assert request.response_format == JSON_RESPONSE_FORMAT == {"type": "json_object"}
    assert request.max_tokens == 64
    assert request.model is None  # the client's configured model
    assert [m.role for m in request.messages] == ["system", "user"]
    assert result.details["attempts"] == 1
    assert result.details["cached"] is False
    assert "cache_key" not in result.details
    assert result.details["judge_model"] == "mock"


def test_retry_after_unparseable_reply() -> None:
    llm = MockLLM(["I think it is fine.", '{"n": 2}'])
    judge = EchoJudge(client=llm, cache=False)
    result = judge.evaluate(TRACE, CASE)
    assert result.passed
    assert result.details["attempts"] == 2
    retry = llm.requests[1]
    assert retry.temperature == 0.0
    assert [m.role for m in retry.messages] == ["system", "user", "assistant", "user"]
    assert retry.messages[2].content == "I think it is fine."
    assert "could not be used" in (retry.messages[3].content or "")
    usage = result.details["judge_usage"]
    assert isinstance(usage, dict)
    assert usage["output_tokens"] == 5 + 2  # both attempts counted


def test_unparseable_after_retries_is_a_failed_result() -> None:
    llm = MockLLM(["nope", '{"n": "x"}'])
    judge = EchoJudge(client=llm, cache=MemoryJudgeCache())
    result = judge.evaluate(TRACE, CASE)
    assert not result.passed
    assert result.score == 0.0
    assert result.reason == (
        "judge reply could not be parsed after 2 attempts: 'n' must be an integer"
    )
    assert result.details["raw_reply"] == '{"n": "x"}'
    assert result.details["error"] == {
        "type": "JudgeParseError",
        "message": "'n' must be an integer",
    }
    assert judge.cache is not None
    assert len(judge.cache) == 0  # type: ignore[arg-type]  # failures are not cached


def test_no_retry_and_long_raw_reply_is_clipped() -> None:
    llm = MockLLM(["x" * 5000])
    result = EchoJudge(client=llm, cache=False, retries=0).evaluate(TRACE, CASE)
    assert llm.calls == 1
    assert "after 1 attempt:" in (result.reason or "")
    raw = result.details["raw_reply"]
    assert isinstance(raw, str)
    assert raw.endswith("…")
    assert len(raw) == 4001


def test_none_content_counts_as_empty_reply() -> None:
    client = NoUsageClient([LLMResponse(content=None)])
    result = EchoJudge(client=client, cache=False, retries=0).evaluate(TRACE, CASE)
    assert result.details["raw_reply"] == ""
    assert "empty reply" in (result.reason or "")
    assert "judge_usage" not in result.details


def test_cache_hit_avoids_second_call() -> None:
    cache = MemoryJudgeCache()
    llm = MockLLM(['{"n": 3}'])  # a second call would raise MockExhaustedError
    first = EchoJudge(client=llm, cache=cache).evaluate(TRACE, CASE)
    second = EchoJudge(client=llm, cache=cache).evaluate(TRACE, CASE)
    assert llm.calls == 1
    assert first.details["cached"] is False
    assert second.details["cached"] is True
    assert second.details["attempts"] == 0
    assert "judge_usage" not in second.details
    key = judge_cache_key("mock", "sys\n\njudge the answer", "the answer")
    assert first.details["cache_key"] == second.details["cache_key"] == key
    assert cache.get(key) == {"reply": '{"n": 3}', "judge_model": "mock", "model": "mock"}


def test_cache_misses_on_different_output_or_model() -> None:
    cache = MemoryJudgeCache()
    llm = MockLLM(['{"n": 1}', '{"n": 2}', '{"n": 3}'])
    EchoJudge(client=llm, cache=cache).evaluate(TRACE, CASE)
    EchoJudge(client=llm, cache=cache).evaluate(TRACE.model_copy(update={"output": "x"}), CASE)
    EchoJudge(client=MockLLM(['{"n": 1}'], model="other"), cache=cache).evaluate(TRACE, CASE)
    other = MockLLM(['{"n": 4}'])
    other.model = "other-judge"
    EchoJudge(client=other, cache=cache).evaluate(TRACE, CASE)
    assert llm.calls == 2
    assert len(cache) == 4


def test_stale_cache_entry_is_ignored() -> None:
    cache = MemoryJudgeCache()
    key = judge_cache_key("mock", "sys\n\njudge the answer", "the answer")
    cache.set(key, {"reply": "garbage"})
    llm = MockLLM(['{"n": 5}'])
    result = EchoJudge(client=llm, cache=cache).evaluate(TRACE, CASE)
    assert llm.calls == 1
    assert result.details["cached"] is False
    assert cache.get(key) == {"reply": '{"n": 5}', "judge_model": "mock", "model": "mock"}
    cache.set(key, {"no_reply": True})
    assert EchoJudge(client=MockLLM(['{"n": 6}']), cache=cache).evaluate(TRACE, CASE).passed


def test_async_path_uses_acomplete_and_cache() -> None:
    cache = MemoryJudgeCache()
    llm = MockLLM(["bad", '{"n": 1}'])
    judge = EchoJudge(client=llm, cache=cache)
    result = asyncio.run(judge.aevaluate(TRACE, CASE))
    assert result.passed
    assert result.details["attempts"] == 2
    again = asyncio.run(arun_evaluator(judge, TRACE, CASE))
    assert again.details["cached"] is True
    assert llm.calls == 2


def test_usage_without_cost_and_client_label() -> None:
    client = NoUsageClient(
        [
            LLMResponse(content="?", usage=Usage(input_tokens=3, output_tokens=1, cost_usd=0.1)),
            LLMResponse(content='{"n": 1}', usage=Usage(input_tokens=5, output_tokens=2)),
        ]
    )
    result = EchoJudge(client=client, cache=False).evaluate(TRACE, CASE)
    assert result.details["judge_model"] == "acme:judge-1"
    assert result.details["judge_usage"] == {
        "input_tokens": 8,
        "output_tokens": 3,
        "cost_usd": None,
    }


def test_add_usage() -> None:
    a = Usage(input_tokens=1, output_tokens=2, cost_usd=0.5)
    assert _add_usage(None, None) is None
    assert _add_usage(a, None) is a
    assert _add_usage(None, a) is a
    assert _add_usage(a, a) == Usage(input_tokens=2, output_tokens=4, cost_usd=1.0)


def test_client_label_falls_back_to_class_name() -> None:
    class Bare:
        def complete(self, request: LLMRequest) -> LLMResponse:
            return LLMResponse(content='{"n": 1}')

        async def acomplete(self, request: LLMRequest) -> LLMResponse:
            return self.complete(request)

    assert EchoJudge(client=Bare(), cache=False).judge_model == "Bare"


# -- client resolution -------------------------------------------------------------


def test_no_model_configured_is_an_evaluator_error() -> None:
    judge = EchoJudge(cache=False)  # building needs no environment
    with pytest.raises(LLMConfigError, match=JUDGE_MODEL_ENV):
        _ = judge.client
    result = run_evaluator(judge, TRACE, CASE)
    assert not result.passed
    assert result.details["error"]["type"] == "LLMConfigError"  # type: ignore[index,call-overload]


def test_model_spec_is_resolved_lazily() -> None:
    judge = EchoJudge(model="mock:judge", cache=False)
    assert judge._client is None
    assert judge.judge_model == "mock:judge"
    client = judge.client
    assert isinstance(client, MockLLM)
    assert client.model == "judge"
    assert judge.client is client


def test_model_from_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("AGENTLENS_MODEL", "mock:general")
    judge = EchoJudge(cache=False)
    assert judge.judge_model == "mock:general"
    monkeypatch.setenv(JUDGE_MODEL_ENV, "Mock:judge")
    assert judge.judge_model == "mock:judge"
    assert isinstance(judge.client, MockLLM)


def test_explicit_model_wins_over_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(JUDGE_MODEL_ENV, "mock:env")
    assert EchoJudge(model="mock:param", cache=False).judge_model == "mock:param"


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"model": "mock", "client": MockLLM([])}, "not both"),
        ({"model": ""}, "spec"),
        ({"model": 3}, "spec"),
        ({"model": "nope:x"}, "unknown LLM provider"),
        ({"client": "mock"}, "LLMClient"),
        ({"retries": -1}, "retries"),
        ({"retries": 6}, "retries"),
        ({"retries": True}, "retries"),
        ({"retries": 1.0}, "retries"),
        ({"max_tokens": 0}, "max_tokens"),
        ({"max_tokens": True}, "max_tokens"),
        ({"cache": 3}, "cache must be"),
    ],
)
def test_invalid_parameters(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        EchoJudge(**kwargs)  # type: ignore[arg-type]


def test_llm_errors_propagate_to_run_evaluator() -> None:
    def boom(request: LLMRequest) -> str:
        raise RuntimeError("provider down")

    result = run_evaluator(EchoJudge(client=MockLLM(boom), cache=False), TRACE, CASE)
    assert result.reason == "evaluator error: RuntimeError: provider down"


def test_missing_verdict_without_error() -> None:
    judge = EchoJudge(client=MockLLM([]), cache=False)
    run: JudgeRun[int] = JudgeRun(judge_model="m", cache_key=None)
    result = judge._result(run)
    assert (
        result.reason
        == "judge reply could not be parsed after 0 attempts: the judge gave no verdict"
    )
