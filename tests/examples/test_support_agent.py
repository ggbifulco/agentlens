"""The example support-desk agent (``examples/support_agent``) and its golden dataset."""

from __future__ import annotations

import json
import os
from collections import Counter
from pathlib import Path
from typing import Any

import pytest
from examples.support_agent import (
    DEFAULT_MODEL,
    GIVE_UP_MESSAGE,
    MAX_ITERATIONS,
    TOOL_NAMES,
    TOOL_SPECS,
    build_llm,
    call_tool,
    detect_intents,
    make_mock_llm,
    make_tools,
    run_agent,
    support_policy,
    system_prompt,
)
from examples.support_agent import agent as agent_module
from examples.support_agent.__main__ import main
from examples.support_agent.data import FAQ, new_database
from examples.support_agent.mock_policy import Intent, Rules

import agentlens
from agentlens.llm import LLMRequest, Message, MockLLM, ToolCall
from agentlens.suite import load_suite, run_suite

ROOT = Path(__file__).resolve().parents[2]
EXAMPLE = ROOT / "examples" / "support_agent"
SUITE = EXAMPLE / "suite.yaml"


@pytest.fixture(autouse=True)
def _offline(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("AGENTLENS_MODEL", raising=False)


def _traced(question: str, **kwargs: Any) -> tuple[str, list[str], int]:
    with agentlens.run("support-agent", input=question) as trace:
        trace.output = run_agent(question, **kwargs)
    assert trace.error is None
    assert isinstance(trace.output, str)
    return trace.output, trace.tool_calls, trace.n_steps


# --- the golden dataset ----------------------------------------------------------------------


def test_golden_dataset_passes_offline() -> None:
    suite = load_suite(SUITE)
    result = run_suite(suite)
    assert len(result.runs) >= 20
    failed = {r.case_id: [e.reason for e in r.eval_results if not e.passed] for r in result.runs}
    assert all(run.passed for run in result.runs), {k: v for k, v in failed.items() if v}
    assert all(run.trace.error is None for run in result.runs)


def test_golden_dataset_covers_every_tool_and_category() -> None:
    cases = load_suite(SUITE).resolved_cases()
    tags = Counter(tag for case in cases for tag in case.tags)
    for tag in ("refund", "tracking", "address", "faq", "escalation", "unknown-order", "policy"):
        assert tags[tag] >= 2, tag
    assert tags["multi-step"] >= 2
    expected_tools = {
        tool for case in cases if isinstance(case.expected, dict) for tool in case.expected["tools"]
    }
    assert expected_tools == set(TOOL_NAMES)
    for case in cases:
        names = {spec.name for spec in case.evaluators}
        assert {"no_error", "max_steps", "tool_sequence"} <= names, case.id
        assert {"contains", "regex"} & names, case.id


def test_mock_calls_exactly_the_expected_tools() -> None:
    """Stronger than the suite's ``ordered`` checks: the mock is the reference behaviour."""
    for case in load_suite(SUITE).resolved_cases():
        assert isinstance(case.input, str)
        assert isinstance(case.expected, dict)
        _, tools, _ = _traced(case.input)
        assert tools == case.expected["tools"], case.id


def test_golden_dataset_is_isolated_under_concurrency() -> None:
    result = run_suite(load_suite(SUITE), concurrency=8, repetitions=2)
    assert all(run.passed for run in result.runs)


def test_judged_suite_shares_the_dataset_and_adds_a_rubric() -> None:
    judged = load_suite(EXAMPLE / "suite.judges.yaml")
    plain = load_suite(SUITE)
    assert [c.id for c in judged.cases] == [c.id for c in plain.cases]
    assert judged.agent == plain.agent == "examples.support_agent:run_agent"
    for case in judged.resolved_cases():
        assert "llm_rubric" in {spec.name for spec in case.evaluators}


def test_prompt_change_is_caught_as_a_regression(monkeypatch: pytest.MonkeyPatch) -> None:
    """Dropping the refund limit from the prompt makes the mock refund the $899 laptop."""
    prompt = system_prompt().replace("of up to $200 yourself", "yourself")
    assert prompt != system_prompt()
    monkeypatch.setattr(agent_module, "system_prompt", lambda version: prompt)
    result = run_suite(load_suite(SUITE))
    failed = {run.case_id for run in result.runs if not run.passed}
    assert failed == {"refund-over-limit", "multi-broken-over-limit"}


# --- the agent loop --------------------------------------------------------------------------


def test_run_agent_defaults_to_the_offline_mock() -> None:
    answer, tools, steps = _traced("I want a refund for order 1001")
    assert tools == ["lookup_order", "issue_refund"]
    assert steps == 3
    assert "$49.99" in answer
    assert DEFAULT_MODEL.startswith("mock")


def test_each_run_gets_a_fresh_database() -> None:
    first = _traced("Refund order 1001")[0]
    assert _traced("Refund order 1001")[0] == first  # not "already refunded"


def test_run_agent_executes_tool_calls_of_any_client() -> None:
    llm = MockLLM(
        [
            [
                ToolCall(id="a", name="lookup_order", arguments={"order_id": "#1003"}),
                ToolCall(id="b", name="track_order", arguments={"order_id": "1003"}),
            ],
            ToolCall(id="c", name="no_such_tool", arguments={}),
            ToolCall(id="d", name="lookup_order", arguments={"id": "1003"}),
            "Order 1003 is being prepared.",
        ]
    )
    with agentlens.run("support-agent") as trace:
        answer = run_agent("Status of 1003?", llm=llm)
    assert answer == "Order 1003 is being prepared."
    # The call with bad arguments is traced too (with its error).
    assert trace.tool_calls == ["lookup_order", "track_order", "lookup_order"]
    assert llm.requests[0].tools == TOOL_SPECS
    assert llm.requests[0].temperature == 0
    assert llm.requests[0].messages[0].content == system_prompt()
    results = [json.loads(m.content or "") for m in llm.requests[-1].messages if m.role == "tool"]
    assert results[0]["customer"]["name"] == "Chiara Bianchi"
    assert results[1]["status"] == "processing"
    assert results[2]["error"] == "unknown_tool"
    assert results[3]["error"] == "invalid_arguments"


def test_run_agent_gives_up_after_max_iterations() -> None:
    llm = MockLLM([ToolCall(id="x", name="search_faq", arguments={"query": "hours"})], cycle=True)
    assert run_agent("Loop forever", llm=llm) == GIVE_UP_MESSAGE
    assert llm.calls == MAX_ITERATIONS
    assert run_agent("Loop forever", llm=llm, max_iterations=2) == GIVE_UP_MESSAGE


def test_run_agent_returns_empty_text_for_an_empty_answer() -> None:
    assert run_agent("Hi", llm=MockLLM([""])) == ""


def test_build_llm_resolution(monkeypatch: pytest.MonkeyPatch) -> None:
    mock = build_llm()
    assert isinstance(mock, MockLLM)
    assert mock.model == "support"
    assert isinstance(build_llm("mock"), MockLLM)
    monkeypatch.setenv("AGENTLENS_MODEL", "mock:other")
    env_mock = build_llm()
    assert isinstance(env_mock, MockLLM)
    assert env_mock.model == "other"
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    real = build_llm("openai:gpt-4o-mini")
    assert not isinstance(real, MockLLM)
    assert getattr(real, "model", None) == "gpt-4o-mini"


def test_suite_model_is_passed_to_the_agent() -> None:
    result = run_suite(load_suite(SUITE), model="mock:support-v2")
    assert all(run.passed for run in result.runs)
    llm_spans = [s for run in result.runs for s in run.trace.spans if s.kind == "llm_call"]
    assert {s.attributes.get("response_model") for s in llm_spans} == {"support-v2"}


def test_main_prints_answer_and_tools(capsys: pytest.CaptureFixture[str]) -> None:
    assert main(["Where is my order 1002?"]) == 0
    out = capsys.readouterr().out
    assert "UPS" in out
    assert "tools: track_order | llm calls: 2" in out
    assert main(["What's the weather in Rome?", "--model", "mock"]) == 0
    assert "tools: (none)" in capsys.readouterr().out


# --- the mock policy ------------------------------------------------------------------------


def test_rules_are_read_from_the_prompt() -> None:
    rules = Rules.from_prompt(system_prompt())
    assert rules == Rules(refund_limit=200.0, refund_window_days=30)
    assert Rules.from_prompt("Be nice.") == Rules()
    assert Rules.from_prompt("refunds of up to $99.50").refund_limit == 99.5


def test_without_rules_the_mock_refunds_everything_delivered() -> None:
    llm = make_mock_llm()
    answer = run_agent("Refund order 1005", llm=llm, prompt_version="v1")
    assert "outside our refund window" in answer
    no_rules = MockLLM(
        lambda request: support_policy(
            request.model_copy(update={"messages": request.messages[1:]})
        )
    )
    answer = run_agent("Refund order 1005", llm=no_rules)
    assert "$25.00" in answer
    answer = run_agent("Refund order 1004", llm=no_rules)
    assert "$899.00" in answer
    assert "issued" in answer


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("Refund order 1001", [Intent("refund", "1001", "Refund order 1001")]),
        ("What is your refund policy?", [Intent("faq", text="What is your refund policy?")]),
        ("Hello there", [Intent("decline", text="Hello there")]),
        (
            "Order 1004 arrived broken. Refund it",
            [Intent("refund", "1004", "Refund it")],
        ),
        (
            "Change the address of 1003 to 1 Main St, Rome.",
            [
                Intent(
                    "address",
                    "1003",
                    "Change the address of 1003 to 1 Main St, Rome.",
                    "1 Main St, Rome",
                )
            ],
        ),
        (
            "Get me a manager about order 1006",
            [Intent("escalate", "1006", "Get me a manager about order 1006")],
        ),
        ("I want a human", [Intent("escalate", None, "I want a human")]),
        ("Track 1002 and track 1002", [Intent("track", "1002", "Track 1002")]),
    ],
)
def test_detect_intents(message: str, expected: list[Intent]) -> None:
    assert detect_intents(message) == expected


@pytest.mark.parametrize(
    ("question", "needle"),
    [
        ("Where is order 1003?", "still being prepared"),
        ("Where is order 1008?", "cancelled, so it will not be shipped"),
        ("Change the address of order 1004 to 2 Elm Road, York", "already delivered"),
        ("Change the address of order 7777 to 2 Elm Road, York", "couldn't find order 7777"),
        ("Change my address to 2 Elm Road, York", "order number"),
        ("Can I pay with a voucher on Sunday?", "Gift cards"),
    ],
)
def test_more_mock_answers(question: str, needle: str) -> None:
    assert needle in run_agent(question)


def test_mock_reports_tool_errors_and_bad_results() -> None:
    def request(*results: str, question: str = "Refund order 1001") -> LLMRequest:
        names = ["lookup_order", "issue_refund"][: len(results)]
        messages = [Message.system(system_prompt()), Message.user(question)]
        for i, (name, content) in enumerate(zip(names, results, strict=True), start=1):
            call = ToolCall(id=f"call_{i}", name=name, arguments={})
            messages += [Message.assistant(None, [call]), Message.tool(content, f"call_{i}")]
        return LLMRequest(messages=messages)

    order = json.dumps(
        {"total": 10.0, "status": "delivered", "refunded": False, "delivered_on": "2026-10-30"}
    )
    failed = support_policy(request(order, json.dumps({"error": "x", "message": "boom"})))
    assert failed == "I couldn't refund order 1001: boom."
    assert support_policy(request("not json")) == (
        "I couldn't find order 1001. Could you double-check the order number?"
    )
    assert "couldn't find order 1001" in str(support_policy(request("[1, 2]")))
    update_failed = support_policy(
        LLMRequest(
            messages=[
                Message.system(system_prompt()),
                Message.user("Change the address of order 1003 to 1 Main St"),
                Message.assistant(None, [ToolCall(id="call_1", name="lookup_order")]),
                Message.tool(json.dumps({"status": "processing"}), "call_1"),
                Message.assistant(None, [ToolCall(id="call_2", name="update_address")]),
                Message.tool(json.dumps({"error": "e", "message": "nope"}), "call_2"),
            ]
        )
    )
    assert update_failed == "I couldn't update order 1003: nope."


def test_faq_without_results() -> None:
    llm = MockLLM(
        lambda request: (
            support_policy(request)
            if not any(m.role == "tool" for m in request.messages)
            else support_policy(
                request.model_copy(
                    update={
                        "messages": [
                            *request.messages[:-1],
                            Message.tool(json.dumps({"results": []}), "call_1"),
                        ]
                    }
                )
            )
        )
    )
    assert "couldn't find an answer" in run_agent("What are your opening hours?", llm=llm)


# --- the tools -------------------------------------------------------------------------------


def test_tool_specs_match_the_tools() -> None:
    tools = make_tools(new_database())
    assert tuple(tools) == TOOL_NAMES
    assert all(spec.parameters["type"] == "object" for spec in TOOL_SPECS)
    assert all(getattr(fn, "__agentlens_tool__", None) == name for name, fn in tools.items())


def test_tool_invariants() -> None:
    db = new_database()
    tools = make_tools(db)
    assert tools["issue_refund"](order_id="0000", amount=1)["error"] == "order_not_found"
    assert tools["issue_refund"](order_id="1003", amount=1)["error"] == "not_refundable"
    assert tools["issue_refund"](order_id="1007", amount=1)["error"] == "already_refunded"
    assert tools["issue_refund"](order_id="1001", amount=50)["error"] == "invalid_amount"
    assert tools["issue_refund"](order_id="1001", amount=0)["error"] == "invalid_amount"
    refund = tools["issue_refund"](order_id=" #1001 ", amount="49.99", reason="broken")
    assert refund == {
        "refund_id": "R-1001",
        "order_id": "1001",
        "amount": 49.99,
        "reason": "broken",
        "status": "issued",
    }
    assert db.refunds == [refund]
    assert tools["issue_refund"](order_id="1001", amount=1)["error"] == "already_refunded"
    assert tools["track_order"](order_id="0000")["error"] == "order_not_found"
    assert tools["update_address"](order_id="0000", new_address="x")["error"] == "order_not_found"
    assert tools["update_address"](order_id="1003", new_address=" ")["error"] == "invalid_address"
    assert tools["update_address"](order_id="1001", new_address="x")["error"] == "already_shipped"
    assert (
        tools["update_address"](order_id="1003", new_address=" 1 Main St ")["status"] == "updated"
    )
    assert db.orders["1003"]["shipping_address"] == "1 Main St"
    first = tools["escalate_to_human"](reason="help")
    second = tools["escalate_to_human"](reason="again", order_id="#1002")
    assert (first["ticket_id"], first["order_id"]) == ("ESC-1001", None)
    assert (second["ticket_id"], second["order_id"]) == ("ESC-1002", "1002")
    assert new_database().orders["1003"]["shipping_address"] == "5 Via Roma, Milano"


def test_search_faq_ranks_by_keyword_hits() -> None:
    search = make_tools(new_database())["search_faq"]
    assert search(query="What about the weather?")["results"] == []
    results = search(query="Can I pay by card or PayPal?")["results"]
    assert results[0]["id"] == "payment-methods"
    assert len(search(query="ship to canada")["results"]) == 2
    assert {article.id for article in FAQ} >= {"returns-policy", "warranty"}


def test_call_tool_turns_failures_into_error_results() -> None:
    def boom(**_: Any) -> dict[str, Any]:
        raise RuntimeError("database down")

    tools = {"boom": boom}
    result = json.loads(call_tool(tools, ToolCall(id="1", name="boom")))
    assert result == {"error": "tool_failed", "message": "RuntimeError: database down"}


def test_tool_exceptions_are_traced() -> None:
    tools = make_tools(new_database())
    with agentlens.run("t") as trace:
        out = json.loads(
            call_tool(
                tools,
                ToolCall(
                    id="1", name="issue_refund", arguments={"order_id": "1001", "amount": "x"}
                ),
            )
        )
    assert out["error"] == "tool_failed"
    assert trace.spans[0].error is not None


# --- live ------------------------------------------------------------------------------------

LIVE_MODEL = os.environ.get("AGENTLENS_LIVE_SUPPORT_MODEL")


@pytest.mark.live
@pytest.mark.skipif(not LIVE_MODEL, reason="set AGENTLENS_LIVE_SUPPORT_MODEL (and its API key)")
def test_live_golden_dataset() -> None:
    """A real model on the golden dataset: most cases should pass (wording varies)."""
    result = run_suite(load_suite(SUITE), model=LIVE_MODEL)
    passed = sum(run.passed for run in result.runs)
    assert passed / len(result.runs) >= 0.7, [r.case_id for r in result.runs if not r.passed]
