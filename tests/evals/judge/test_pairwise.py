"""The pairwise A/B judge: both orders, mapping back, averaging, consistency, cache, async."""

from __future__ import annotations

import asyncio
import json
from collections.abc import Callable

import pytest
from pydantic import JsonValue

from agentlens.evals import EvaluatorConfigError, PairwiseJudge, PairwiseResult
from agentlens.evals.judge import (
    JudgeParseError,
    MemoryJudgeCache,
    OrderVerdict,
    parse_position,
)
from agentlens.llm import LLMRequest, MockLLM
from agentlens.schema import Case, Trace

CRITERIA = "Confirms the refund amount"
A = Trace(name="agent", input="Refund order 1234", output="Refund done.")
B = Trace(name="agent", input="Refund order 1234", output="Your refund of $42 is on its way.")
CASE = Case(id="refund", input="Refund order 1234", expected="Confirms a $42 refund")


def reply(winner: JsonValue, reasoning: str | None = None) -> str:
    data: dict[str, JsonValue] = {"winner": winner}
    if reasoning is not None:
        data["reasoning"] = reasoning
    return json.dumps(data)


def first_output(request: LLMRequest) -> str:
    text = request.messages[1].content or ""
    return text.split("<output_1>\n", 1)[1].split("\n</output_1>", 1)[0]


def by_order(ab: str, ba: str) -> Callable[[LLMRequest], str]:
    """A responder answering ``ab`` when A is shown first and ``ba`` when B is."""

    def respond(request: LLMRequest) -> str:
        return ab if first_output(request) == A.output else ba

    return respond


def judge(script: object, **kwargs: object) -> tuple[PairwiseJudge, MockLLM]:
    llm = MockLLM(script)  # type: ignore[arg-type]
    kwargs.setdefault("cache", False)
    return PairwiseJudge(CRITERIA, client=llm, **kwargs), llm  # type: ignore[arg-type]


# -- parsing -----------------------------------------------------------------------


@pytest.mark.parametrize(
    ("data", "expected"),
    [
        ({"winner": "1"}, "1"),
        ({"winner": 2}, "2"),
        ({"Winner": " Output 2. "}, "2"),
        ({"choice": "first"}, "1"),
        ({"better": "second"}, "2"),
        ({"winner": "tie"}, "tie"),
        ({"verdict": "Draw"}, "tie"),
        ({"winner": "neither"}, "tie"),
        ({"winner": '"1"'}, "1"),
    ],
)
def test_parse_position(data: dict[str, JsonValue], expected: str) -> None:
    assert parse_position(data) == expected


@pytest.mark.parametrize(
    ("data", "message"),
    [
        ({"reasoning": "x"}, "no 'winner' field"),
        ({"winner": 3}, "must be '1', '2' or 'tie'"),
        ({"winner": True}, "must be '1', '2' or 'tie'"),
        ({"winner": "A"}, "must be '1', '2' or 'tie'"),
        ({"winner": None}, "must be '1', '2' or 'tie'"),
    ],
)
def test_parse_position_errors(data: dict[str, JsonValue], message: str) -> None:
    with pytest.raises(JudgeParseError, match=message):
        parse_position(data)


# -- both orders, mapping and averaging ---------------------------------------------


def test_asks_both_orders_with_swapped_outputs() -> None:
    pj, llm = judge([reply("2", "B states $42."), reply("1", "B states $42.")])
    result = pj.compare(A, B, CASE)
    ab, ba = llm.requests
    assert first_output(ab) == A.output
    assert first_output(ba) == B.output
    for request in (ab, ba):
        assert request.temperature == 0.0
        assert request.response_format == {"type": "json_object"}
        user = request.messages[1].content or ""
        assert f"<criteria>\n{CRITERIA}\n</criteria>" in user
        assert "<input>\nRefund order 1234\n</input>" in user
        assert "<reference>\nConfirms a $42 refund\n</reference>" in user
        assert '"winner": "1", "2" or "tie"' in user
        assert "impartial" in (request.messages[0].content or "")
    assert result.winner == "b"
    assert result.preference == 0.0
    assert result.consistent
    assert result.confidence == 1.0
    assert result.ok
    assert result.error is None
    assert [o.position for o in result.orders] == ["2", "1"]
    assert [o.winner for o in result.orders] == ["b", "b"]
    assert result.reasons == ["ab: B states $42.", "ba: B states $42."]
    assert result.reason == "B is better (preference for A 0.00, consistent across orders)"


def test_consistent_a_wins() -> None:
    pj, _ = judge(by_order(reply("1"), reply("2")))
    result = pj.compare(A, B, CASE)
    assert (result.winner, result.preference, result.consistent) == ("a", 1.0, True)
    assert result.reasons == []


def test_consistent_tie() -> None:
    pj, _ = judge([reply("tie"), reply("tie")])
    result = pj.compare(A, B, CASE)
    assert (result.winner, result.preference, result.consistent, result.confidence) == (
        "tie",
        0.5,
        True,
        1.0,
    )
    assert result.reason.startswith("tie (preference for A 0.50, consistent")


def test_position_bias_opposite_verdicts_become_a_tie() -> None:
    # The judge always picks whatever is shown first: pure position bias.
    pj, _ = judge([reply("1"), reply("1")])
    result = pj.compare(A, B, CASE)
    assert result.winner == "tie"
    assert result.preference == 0.5
    assert not result.consistent
    assert result.confidence == 0.0
    assert [o.winner for o in result.orders] == ["a", "b"]
    assert "the orders disagree: ab → a, ba → b" in result.reason
    assert "position bias" in result.reason
    details = result.details()
    assert details["consistent"] is False
    assert details["confidence"] == 0.0


def test_partly_inconsistent_keeps_winner_with_lower_confidence() -> None:
    pj, _ = judge(by_order(reply("2"), reply("tie")))
    result = pj.compare(A, B, CASE)
    assert result.winner == "b"
    assert result.preference == 0.25
    assert not result.consistent
    assert result.confidence == 0.5
    assert "confidence 0.50" in result.reason


# -- conversion to EvalResult -----------------------------------------------------


def test_to_eval_result_for_candidate() -> None:
    pj, _ = judge(by_order(reply("2", "B is precise"), reply("1", "B is precise")))
    result = pj.compare(A, B, CASE)
    as_b = result.to_eval_result()
    assert as_b.evaluator == "pairwise"
    assert as_b.passed
    assert as_b.score == 1.0
    assert (as_b.reason or "").startswith("candidate B wins: B is better")
    assert as_b.details["candidate"] == "b"
    assert as_b.details["winner"] == "b"
    orders = as_b.details["orders"]
    assert isinstance(orders, list)
    assert len(orders) == 2
    assert orders[0] == {
        "order": "ab",
        "position": "2",
        "winner": "b",
        "reasoning": "B is precise",
        "judge_model": "mock",
        "cached": False,
        "attempts": 1,
        "judge_usage": orders[0]["judge_usage"],  # type: ignore[index,call-overload]
    }
    as_a = result.to_eval_result("a", name="pairwise_vs_baseline")
    assert as_a.evaluator == "pairwise_vs_baseline"
    assert not as_a.passed
    assert as_a.score == 0.0
    assert (as_a.reason or "").startswith("candidate A loses")


def test_to_eval_result_tie_passes() -> None:
    pj, _ = judge([reply("1"), reply("1")])
    as_b = pj.compare(A, B, CASE).to_eval_result("b")
    assert as_b.passed
    assert as_b.score == 0.5
    assert (as_b.reason or "").startswith("candidate B ties")


def test_to_eval_result_rejects_bad_candidate() -> None:
    pj, _ = judge([reply("1"), reply("2")])
    with pytest.raises(ValueError, match="candidate must be"):
        pj.compare(A, B).to_eval_result("c")  # type: ignore[arg-type]


# -- parse failures -----------------------------------------------------------------


def test_retry_then_success_in_one_order() -> None:
    pj, llm = judge(["no idea", reply("2"), reply("1")])
    result = pj.compare(A, B, CASE)
    assert llm.calls == 3
    assert result.ok
    assert result.winner == "b"
    assert result.orders[0].judge["attempts"] == 2


def test_unparseable_order_gives_error_result() -> None:
    pj, _ = judge(["nope", "x" * 5000, reply("1")])
    result = pj.compare(A, B, CASE)
    assert not result.ok
    assert result.winner == "tie"
    assert result.preference == 0.5
    assert not result.consistent
    assert result.confidence == 0.0
    assert result.error == "order ab: no JSON object found in the judge's reply"
    bad = result.orders[0]
    assert bad.winner is None
    assert bad.preference is None
    assert bad.raw_reply is not None
    assert bad.raw_reply.endswith("…")
    assert result.orders[1].winner == "b"
    assert result.reason.startswith("pairwise judge reply could not be parsed: order ab:")
    evaluated = result.to_eval_result()
    assert not evaluated.passed
    assert evaluated.score == 0.0
    assert evaluated.details["error"] == {
        "type": "JudgeParseError",
        "message": "order ab: no JSON object found in the judge's reply",
    }
    orders = evaluated.details["orders"]
    assert isinstance(orders, list)
    assert isinstance(orders[0], dict)
    assert orders[0]["error"] == "no JSON object found in the judge's reply"


def test_order_verdict_without_error_object() -> None:
    from agentlens.evals.judge import JudgeRun

    verdict = OrderVerdict.from_run("ba", JudgeRun(judge_model="m", cache_key=None))
    assert verdict.error == "the judge gave no verdict"
    assert verdict.raw_reply == ""


def test_both_orders_unparseable() -> None:
    pj, _ = judge(["?", "?"], retries=0)
    result = pj.compare(A, B)
    assert result.error is not None
    assert "order ab:" in result.error
    assert "order ba:" in result.error


# -- cache ----------------------------------------------------------------------------


def test_cache_key_includes_order_and_is_reused() -> None:
    cache = MemoryJudgeCache()
    pj, llm = judge([reply("2"), reply("1")], cache=cache)
    first = pj.compare(A, B, CASE)
    assert len(cache) == 2  # one entry per order
    keys = [o.judge["cache_key"] for o in first.orders]
    assert keys[0] != keys[1]
    again, llm2 = judge([], cache=cache)  # an empty script: any call would fail
    second = again.compare(A, B, CASE)
    assert llm2.calls == 0
    assert all(o.judge["cached"] for o in second.orders)
    assert (second.winner, second.preference) == (first.winner, first.preference)
    assert llm.calls == 2


def test_identical_outputs_ask_once() -> None:
    cache = MemoryJudgeCache()
    pj, llm = judge([reply("tie")], cache=cache)
    result = pj.compare(A, A, CASE)
    assert llm.calls == 1  # both orders show the judge the same prompt
    assert len(cache) == 1
    assert [o.judge["cached"] for o in result.orders] == [False, True]
    assert result.winner == "tie"


def test_swapping_arguments_reuses_the_same_two_verdicts() -> None:
    cache = MemoryJudgeCache()
    pj, _ = judge(by_order(reply("2"), reply("1")), cache=cache)
    assert pj.compare(A, B, CASE).winner == "b"
    swapped, llm = judge([], cache=cache)
    result = swapped.compare(B, A, CASE)
    assert llm.calls == 0
    assert result.winner == "a"  # still the trace with output B


# -- async ----------------------------------------------------------------------------


def test_acompare_matches_compare() -> None:
    pj, llm = judge(by_order(reply("2", "r"), reply("tie", "r")))
    result = asyncio.run(pj.acompare(A, B, CASE))
    assert llm.calls == 2
    assert (result.winner, result.preference, result.consistent) == ("b", 0.25, False)
    assert isinstance(result, PairwiseResult)


def test_acompare_uses_cache() -> None:
    cache = MemoryJudgeCache()
    pj, _ = judge(by_order(reply("1"), reply("2")), cache=cache)
    asyncio.run(pj.acompare(A, B, CASE))
    again, llm = judge([], cache=cache)
    result = asyncio.run(again.acompare(A, B, CASE))
    assert llm.calls == 0
    assert result.winner == "a"


# -- prompt and configuration ----------------------------------------------------------


def test_input_falls_back_to_case_and_reference_can_be_hidden() -> None:
    a = Trace(name="agent", output={"amount": 42})
    b = Trace(name="agent", output=None)
    pj, llm = judge([reply("1"), reply("2")], include_expected=False)
    pj.compare(a, b, CASE)
    user = llm.requests[0].messages[1].content or ""
    assert "<input>\nRefund order 1234\n</input>" in user
    assert '<output_1>\n{"amount": 42}\n</output_1>' in user
    assert "<output_2>\n(none)\n</output_2>" in user
    assert "<reference>" not in user


def test_no_case_and_no_input() -> None:
    a = Trace(name="agent", output="x")
    pj, llm = judge([reply("1"), reply("2")])
    pj.compare(a, a)
    assert "<input>\n(none)\n</input>" in (llm.requests[0].messages[1].content or "")


def test_custom_template() -> None:
    pj, llm = judge(
        [reply("1"), reply("2")],
        template="Pick: $criteria | $input | 1=$output_1 | 2=$output_2 | $reference",
        include_expected=False,
    )
    pj.compare(A, B, CASE)
    assert llm.requests[1].messages[1].content == (
        f"Pick: {CRITERIA} | Refund order 1234 | 1={B.output} | 2={A.output} | "
    )
    assert "template='custom'" in repr(pj)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"criteria": ""}, "criteria must be a non-empty string"),
        ({"criteria": 3}, "criteria must be a non-empty string"),
        ({"include_expected": "yes"}, "include_expected must be true or false"),
        ({"template": "$output"}, "unknown placeholder"),
        ({"retries": 9}, "retries must be"),
        ({"model": "nope"}, "nope"),
    ],
)
def test_config_errors(kwargs: dict[str, object], message: str) -> None:
    kwargs.setdefault("criteria", CRITERIA)
    with pytest.raises(EvaluatorConfigError, match=message):
        PairwiseJudge(**kwargs)  # type: ignore[arg-type]


def test_repr_and_lazy_model() -> None:
    pj = PairwiseJudge("  Better  ", model="openai:gpt-4o-mini", cache=False)
    assert pj.criteria == "Better"
    assert repr(pj) == (
        "PairwiseJudge(criteria='Better', include_expected=True, "
        "model='openai:gpt-4o-mini', cache=None, retries=1)"
    )
    assert pj.judge_model == "openai:gpt-4o-mini"
    assert pj.name == "pairwise"
