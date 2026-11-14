from __future__ import annotations

from agentlens.suite import AgentImportError, SuiteError, SuiteIssue, format_loc


def test_format_loc() -> None:
    assert format_loc(()) == ""
    assert format_loc(("name",)) == "name"
    assert format_loc(("cases", 2, "evaluators", 0)) == "cases[2].evaluators[0]"
    assert format_loc((3, "id")) == "[3].id"


def test_issue_str_with_full_position() -> None:
    issue = SuiteIssue("unknown evaluator 'foo'", ("cases", 2, "evaluators", 0), "s.yaml", 12, 5)
    assert str(issue) == "s.yaml:12:5: cases[2].evaluators[0]: unknown evaluator 'foo'"
    assert issue.path == "cases[2].evaluators[0]"


def test_issue_str_partial_positions() -> None:
    assert str(SuiteIssue("boom")) == "boom"
    assert str(SuiteIssue("boom", ("name",))) == "name: boom"
    assert str(SuiteIssue("boom", (), "s.yaml")) == "s.yaml: boom"
    assert str(SuiteIssue("boom", (), "s.yaml", 3)) == "s.yaml:3: boom"


def test_suite_error_lists_issues() -> None:
    issues = [SuiteIssue("a", ("name",), "s.yaml", 1, 1), SuiteIssue("b")]
    err = SuiteError("invalid suite s.yaml", issues)
    assert err.issues == issues
    assert err.summary == "invalid suite s.yaml"
    assert str(err) == "invalid suite s.yaml (2 problems):\n  s.yaml:1:1: name: a\n  b"
    one = SuiteError("bad", [SuiteIssue("x")])
    assert str(one) == "bad (1 problem):\n  x"


def test_suite_error_without_issues() -> None:
    err = SuiteError("just a message")
    assert str(err) == "just a message"
    assert err.issues == []


def test_agent_import_error_is_a_suite_error() -> None:
    err = AgentImportError("cannot import")
    assert isinstance(err, SuiteError)
    assert str(err) == "cannot import"
