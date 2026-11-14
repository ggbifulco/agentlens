"""``llm_rubric`` and ``llm_pass_fail`` with a scripted MockLLM judge."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest
import yaml

from agentlens.evals import (
    DiskJudgeCache,
    EvaluatorConfigError,
    LLMPassFail,
    LLMRubric,
    MemoryJudgeCache,
    aevaluate_case,
    evaluate_case,
    get_evaluator,
)
from agentlens.llm import MockLLM
from agentlens.schema import Case, EvaluatorSpec, Trace

TRACE = Trace(name="agent", input="Refund order 1234", output="Your refund of $42 is on its way.")
CASE = Case(id="refund", input="Refund order 1234", expected="Confirms a $42 refund")


def reply(**fields: object) -> str:
    return json.dumps(fields)


def rubric(*replies: str, **kwargs: object) -> tuple[LLMRubric, MockLLM]:
    llm = MockLLM(list(replies))
    kwargs.setdefault("cache", False)
    return LLMRubric("Polite and confirms the refund amount", client=llm, **kwargs), llm  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("score", "normalised", "passed"),
    [(1, 0.0, False), (2, 0.25, False), (3, 0.5, False), (4, 0.75, True), (5, 1.0, True)],
)
def test_rubric_normalises_score(score: int, normalised: float, passed: bool) -> None:
    judge, _ = rubric(reply(reasoning="Polite; states $42.", score=score))
    result = judge.evaluate(TRACE, CASE)
    assert result.evaluator == "llm_rubric"
    assert result.score == normalised
    assert result.passed is passed
    assert result.details["rubric_score"] == float(score)
    assert result.details["reasoning"] == "Polite; states $42."
    assert result.details["criteria"] == "Polite and confirms the refund amount"
    op = ">=" if passed else "<"
    assert result.reason == f"judge score {score}/5 ({op} 4): Polite; states $42."


def test_rubric_min_score_and_fractional_scores() -> None:
    judge, _ = rubric(reply(score="3.5"), min_score=3)
    result = judge.evaluate(TRACE, CASE)
    assert result.passed
    assert result.score == 0.625
    assert result.reason == "judge score 3.5/5 (>= 3)"  # no reasoning given
    assert result.details["min_score"] == 3.0


def test_rubric_prompt_contents() -> None:
    judge, llm = rubric(reply(reasoning="ok", score=5))
    judge.evaluate(TRACE, CASE)
    (request,) = llm.requests
    assert request.temperature == 0.0
    assert request.response_format == {"type": "json_object"}
    system, user = request.messages
    assert system.role == "system"
    assert "impartial" in (system.content or "")
    text = user.content or ""
    assert "<criteria>\nPolite and confirms the refund amount\n</criteria>" in text
    assert "<input>\nRefund order 1234\n</input>" in text
    assert "<output>\nYour refund of $42 is on its way.\n</output>" in text
    assert "<reference>\nConfirms a $42 refund\n</reference>" in text
    assert "1 to 5" in text


def test_reference_omitted_without_expected_or_when_disabled() -> None:
    for case, kwargs in [
        (Case(id="c", input="q"), {}),
        (CASE, {"include_expected": False}),
    ]:
        judge, llm = rubric(reply(score=5), **kwargs)
        judge.evaluate(TRACE, case)
        assert "<reference>" not in (llm.requests[0].messages[1].content or "")


def test_structured_input_output_and_case_input_fallback() -> None:
    trace = Trace(name="agent", input=None, output={"b": [1, 2], "a": "é"})
    case = Case(id="c", input={"question": "q"}, expected={"tools": ["x"]})
    judge, llm = rubric(reply(score=4))
    judge.evaluate(trace, case)
    text = llm.requests[0].messages[1].content or ""
    assert '<input>\n{"question": "q"}\n</input>' in text
    assert '<output>\n{"a": "é", "b": [1, 2]}\n</output>' in text
    assert '<reference>\n{"tools": ["x"]}\n</reference>' in text


def test_none_output_is_shown_as_none() -> None:
    judge, llm = rubric(reply(score=1))
    judge.evaluate(Trace(name="agent", input="q"), CASE)
    assert "<output>\n(none)\n</output>" in (llm.requests[0].messages[1].content or "")


@pytest.mark.parametrize(
    "raw",
    [
        '```json\n{"reasoning": "fine", "score": "4"}\n```',
        'My evaluation: {"reasoning": "fine", "score": 4}. Done.',
        '{"Reasoning": "fine", "Score": 4.0}',
    ],
)
def test_rubric_tolerant_parsing(raw: str) -> None:
    judge, llm = rubric(raw)
    result = judge.evaluate(TRACE, CASE)
    assert result.passed
    assert result.score == 0.75
    assert llm.calls == 1


@pytest.mark.parametrize(
    "first",
    ['{"score": 7}', '{"score": "great"}', "Score: 4", '{"reasoning": "no score"}', ""],
)
def test_rubric_invalid_reply_is_retried(first: str) -> None:
    judge, llm = rubric(first, reply(score=5))
    result = judge.evaluate(TRACE, CASE)
    assert llm.calls == 2
    assert result.score == 1.0
    assert result.details["attempts"] == 2


def test_rubric_unparseable_is_a_failed_result_with_raw_reply() -> None:
    judge, llm = rubric('{"score": 9}', '{"score": 0}')
    result = judge.evaluate(TRACE, CASE)
    assert llm.calls == 2
    assert not result.passed
    assert result.score == 0.0
    assert result.details["raw_reply"] == '{"score": 0}'
    assert "between 1 and 5" in (result.reason or "")


def test_pass_fail_verdicts() -> None:
    llm = MockLLM(
        [
            reply(reasoning="States the amount.", verdict="pass"),
            '```json\n{"verdict": "FAIL", "reasoning": "Rude."}\n```',
        ]
    )
    judge = get_evaluator(
        {"llm_pass_fail": "Polite and confirms the amount"}, client=llm, cache=False
    )
    assert isinstance(judge, LLMPassFail)
    ok = judge.evaluate(TRACE, CASE)
    assert ok.passed
    assert ok.score == 1.0
    assert ok.reason == "judge verdict: pass: States the amount."
    assert ok.details["verdict"] == "pass"
    bad = judge.evaluate(TRACE, CASE)
    assert not bad.passed
    assert bad.score == 0.0
    assert bad.reason == "judge verdict: fail: Rude."
    assert bad.details["reasoning"] == "Rude."
    text = llm.requests[0].messages[1].content or ""
    assert '"verdict"' in text
    assert "Polite and confirms the amount" in text
    assert all(r.temperature == 0.0 for r in llm.requests)


def test_pass_fail_retry_then_failure() -> None:
    llm = MockLLM(['{"verdict": "maybe"}', "no idea"])
    result = LLMPassFail("x", client=llm, cache=False).evaluate(TRACE, CASE)
    assert not result.passed
    assert result.details["raw_reply"] == "no idea"
    assert result.details["attempts"] == 2


def test_cache_hit_avoids_second_judge_call() -> None:
    cache = MemoryJudgeCache()
    llm = MockLLM([reply(reasoning="ok", score=4)])
    first = LLMRubric("crit", client=llm, cache=cache).evaluate(TRACE, CASE)
    second = LLMRubric("crit", client=llm, cache=cache).evaluate(TRACE, CASE)
    assert llm.calls == 1
    assert second.details["cached"] is True
    assert (first.score, first.passed, first.reason) == (second.score, second.passed, second.reason)
    # different criteria = different prompt = cache miss
    other = MockLLM([reply(score=2)])
    assert LLMRubric("other", client=other, cache=cache).evaluate(TRACE, CASE).score == 0.25
    assert other.calls == 1


def test_disk_cache_survives_new_judge_instances(tmp_path: Path) -> None:
    llm = MockLLM([reply(verdict="pass")])
    LLMPassFail("crit", client=llm, cache=str(tmp_path)).evaluate(TRACE, CASE)
    again = LLMPassFail("crit", client=MockLLM([]), cache=DiskJudgeCache(tmp_path))
    result = again.evaluate(TRACE, CASE)
    assert result.passed
    assert result.details["cached"] is True
    assert len(list(tmp_path.glob("*/*.json"))) == 1


def test_rubric_and_pass_fail_do_not_share_cache_entries() -> None:
    cache = MemoryJudgeCache()
    LLMRubric("crit", client=MockLLM([reply(score=5)]), cache=cache).evaluate(TRACE, CASE)
    llm = MockLLM([reply(verdict="fail")])
    assert not LLMPassFail("crit", client=llm, cache=cache).evaluate(TRACE, CASE).passed
    assert llm.calls == 1


def test_async_evaluation() -> None:
    llm = MockLLM([reply(score=5), reply(verdict="pass")])
    judges = [
        LLMRubric("crit", client=llm, cache=False),
        LLMPassFail("crit", client=llm, cache=False),
    ]
    results = asyncio.run(aevaluate_case(TRACE, CASE, judges))
    assert [r.evaluator for r in results] == ["llm_rubric", "llm_pass_fail"]
    assert all(r.passed for r in results)
    single = LLMRubric("crit", client=MockLLM([reply(score=1)]), cache=False)
    assert asyncio.run(single.aevaluate(TRACE, CASE)).score == 0.0


def test_custom_template() -> None:
    template = "Criteria: $criteria\nOutput: $output$reference\nAnswer JSON with score."
    judge, llm = rubric(reply(score=5), template=template)
    judge.evaluate(TRACE, CASE)
    text = llm.requests[0].messages[1].content or ""
    assert text.startswith("Criteria: Polite and confirms the refund amount\nOutput: Your refund")
    assert "<reference>" in text


def test_suite_yaml_spec_builds_judges() -> None:
    specs = yaml.safe_load(
        """
        - {llm_rubric: "Polite, confirms refund amount"}
        - {llm_rubric: {criteria: "Concise", min_score: 3, model: "mock:judge", cache: off}}
        - {llm_pass_fail: {criteria: "Mentions the order id", include_expected: false}}
        """
    )
    built = [get_evaluator(EvaluatorSpec.model_validate(s)) for s in specs]
    assert isinstance(built[0], LLMRubric)
    assert built[0].criteria == "Polite, confirms refund amount"
    assert built[0].min_score == 4.0
    assert isinstance(built[1], LLMRubric)
    assert built[1].min_score == 3.0
    assert built[1].judge_model == "mock:judge"
    assert built[1].cache is None
    assert isinstance(built[2], LLMPassFail)
    assert built[2].include_expected is False


def test_evaluate_case_without_model_reports_config_error() -> None:
    case = Case(id="c", input="q", evaluators=[{"llm_rubric": "x"}])  # type: ignore[list-item]
    (result,) = evaluate_case(TRACE, case)
    assert not result.passed
    assert "no judge model configured" in (result.reason or "")


@pytest.mark.parametrize(
    ("factory", "kwargs", "message"),
    [
        (LLMRubric, {"criteria": ""}, "criteria"),
        (LLMRubric, {"criteria": None}, "criteria"),
        (LLMPassFail, {"criteria": 3}, "criteria"),
        (LLMRubric, {"criteria": "c", "min_score": 0}, "min_score"),
        (LLMRubric, {"criteria": "c", "min_score": 6}, "min_score"),
        (LLMRubric, {"criteria": "c", "min_score": True}, "min_score"),
        (LLMRubric, {"criteria": "c", "min_score": "4"}, "min_score"),
        (LLMRubric, {"criteria": "c", "min_score": float("nan")}, "min_score"),
        (LLMRubric, {"criteria": "c", "include_expected": "no"}, "include_expected"),
        (LLMPassFail, {"criteria": "c", "template": "$criteria $bogus"}, "placeholder"),
        (LLMPassFail, {"criteria": "c", "template": ""}, "template"),
    ],
)
def test_invalid_parameters(factory: type, kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(EvaluatorConfigError, match=message):
        factory(**kwargs)


def test_registry_config_errors() -> None:
    with pytest.raises(EvaluatorConfigError):
        get_evaluator("llm_rubric")  # criteria is required
    with pytest.raises(EvaluatorConfigError, match="unknown LLM provider"):
        get_evaluator("llm_pass_fail", criteria="c", model="bogus:x")


def test_repr() -> None:
    judge = LLMRubric("c", model="mock:j", cache=False, template="$criteria JSON")
    assert repr(judge) == (
        "LLMRubric(criteria='c', include_expected=True, template='custom', model='mock:j', "
        "cache=None, retries=1, min_score=4.0)"
    )
    assert repr(LLMPassFail("  c  ", cache=False)).startswith("LLMPassFail(criteria='c', ")
