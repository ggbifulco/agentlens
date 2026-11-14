from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import pytest
from pydantic import ValidationError

from agentlens.evals import EvaluatorRegistry, NoError
from agentlens.schema import Case, EvaluatorSpec
from agentlens.suite import AgentImportError, Suite, SuiteError, SuiteIssue, merge_evaluators
from tests.suite.conftest import WriteFile


def echo(text: str) -> str:
    return text


def make(**overrides: Any) -> Suite:
    fields: dict[str, Any] = {
        "name": "demo",
        "agent": "pkg.mod:run",
        "cases": [{"id": "a", "input": "hi"}],
    }
    fields.update(overrides)
    return Suite.model_validate(fields)


def specs(*items: Any) -> list[EvaluatorSpec]:
    return [EvaluatorSpec.model_validate(item) for item in items]


def test_defaults() -> None:
    suite = make()
    assert suite.repetitions == 1
    assert suite.concurrency == 1
    assert suite.timeout_s is None
    assert suite.model is None
    assert suite.description is None
    assert suite.imports == []
    assert suite.metadata == {}
    assert suite.defaults.evaluators == []
    assert suite.source_path is None
    assert suite.base_dir is None
    assert suite.cases == [Case(id="a", input="hi")]


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("repetitions", 0),
        ("concurrency", 0),
        ("timeout_s", 0),
        ("name", ""),
        ("cases", []),
        ("agent", "no_colon"),
        ("agent", 42),
        ("imports", ["bad-name"]),
        ("unknown", 1),
    ],
)
def test_invalid_fields(field: str, value: Any) -> None:
    with pytest.raises(ValidationError):
        make(**{field: value})


def test_python_suite_with_callable_agent() -> None:
    suite = Suite(name="py", agent=echo, cases=[Case(id="a", input="x")])
    assert suite.load_agent() is echo
    dumped = suite.model_dump(mode="json")
    assert dumped["agent"] == f"{__name__}:echo"


def test_callable_object_agent_serialises_by_type() -> None:
    class Agent:
        def __call__(self, text: str) -> str:
            return text

    suite = make(agent=Agent())
    assert suite.model_dump()["agent"].endswith(
        ":test_callable_object_agent_serialises_by_type.<locals>.Agent"
    )


def test_string_agent_round_trips() -> None:
    suite = make(defaults={"evaluators": ["no_error", {"max_steps": 8}], "tags": ["smoke"]})
    again = Suite.model_validate(suite.model_dump(mode="json"))
    assert again == suite


def test_duplicate_ids_rejected_in_python() -> None:
    cases = [{"id": "a", "input": 1}, {"id": "b", "input": 2}, {"id": "a", "input": 3}]
    with pytest.raises(ValidationError, match=r"duplicate case id 'a' \(first: cases\[0\]\)"):
        make(cases=cases)


def test_merge_evaluators_overrides_by_name() -> None:
    defaults = specs("no_error", {"max_steps": 8}, {"contains": "x"})
    own = specs({"max_steps": 3}, {"regex": "y"})
    merged = merge_evaluators(defaults, own)
    assert merged == specs("no_error", {"contains": "x"}, {"max_steps": 3}, {"regex": "y"})
    assert merge_evaluators(defaults, merged) == merged  # idempotent
    assert merge_evaluators([], own) == own
    assert merge_evaluators(defaults, []) == defaults


def test_resolved_cases_merge_defaults_without_mutating() -> None:
    suite = make(
        defaults={"evaluators": ["no_error", {"max_steps": 8}], "tags": ["smoke", "support"]},
        cases=[
            {
                "id": "a",
                "input": "hi",
                "evaluators": [{"max_steps": 3}],
                "tags": ["refund", "smoke"],
            },
            {"id": "b", "input": "yo"},
        ],
    )
    a, b = suite.resolved_cases()
    assert a.evaluators == specs("no_error", {"max_steps": 3})
    assert a.tags == ["smoke", "support", "refund"]
    assert b.evaluators == specs("no_error", {"max_steps": 8})
    assert b.tags == ["smoke", "support"]
    assert suite.cases[0].evaluators == specs({"max_steps": 3})  # declared form kept
    assert suite.cases[1].tags == []
    assert suite.case("b") == b
    with pytest.raises(KeyError):
        suite.case("zzz")


def test_python_cases_are_not_mutated() -> None:
    case = Case(id="a", input="x")
    suite = Suite(name="py", agent=echo, cases=[case], defaults={"evaluators": ["no_error"]})
    assert suite.resolved_cases()[0].evaluators == specs("no_error")
    assert case.evaluators == []


def test_check_passes_and_chains(tiny_registry: EvaluatorRegistry) -> None:
    suite = make(defaults={"evaluators": ["no_error"]})
    assert suite.check(registry=tiny_registry) is suite
    assert suite.issues(registry=tiny_registry) == []


def test_check_uses_default_registry() -> None:
    assert make(defaults={"evaluators": ["no_error", {"max_steps": 8}]}).check()


def test_check_reports_every_evaluator_problem(registry: EvaluatorRegistry) -> None:
    suite = make(
        defaults={"evaluators": ["no_errr"]},
        cases=[
            {"id": "a", "input": 1, "evaluators": [{"regex": "("}, "no_errr"]},
            {"id": "b", "input": 2, "evaluators": [{"max_steps": {"nope": 1}}]},
        ],
    )
    with pytest.raises(SuiteError) as info:
        suite.check(registry=registry)
    issues = info.value.issues
    assert [i.path for i in issues] == [
        "defaults.evaluators[0]",
        "cases[0].evaluators[0]",
        "cases[0].evaluators[1]",
        "cases[1].evaluators[0]",
    ]
    assert issues[0].message == "unknown evaluator 'no_errr' (did you mean 'no_error'?)"
    assert "invalid regex" in issues[1].message
    assert issues[2].message == issues[0].message
    assert "invalid parameters for evaluator 'max_steps'" in issues[3].message
    assert str(info.value).startswith("invalid suite 'demo' (4 problems)")


def test_unknown_evaluator_without_suggestion(tiny_registry: EvaluatorRegistry) -> None:
    (issue,) = make(defaults={"evaluators": ["zzzzzz"]}).issues(registry=tiny_registry)
    assert issue.message == "unknown evaluator 'zzzzzz'"


def test_factory_crashing_unexpectedly_is_reported() -> None:
    def factory() -> NoError:
        raise KeyError("boom")

    reg = EvaluatorRegistry()
    reg.register("crashy", factory)
    (issue,) = make(defaults={"evaluators": ["crashy"]}).issues(registry=reg)
    assert issue.message == "cannot build evaluator 'crashy': KeyError: 'boom'"


def test_each_distinct_spec_is_built_once() -> None:
    built: list[int] = []

    def factory() -> NoError:
        built.append(1)
        return NoError()

    reg = EvaluatorRegistry()
    reg.register("counted", factory)
    cases = [{"id": str(i), "input": i, "evaluators": ["counted"]} for i in range(5)]
    assert make(defaults={"evaluators": ["counted"]}, cases=cases).issues(registry=reg) == []
    assert len(built) == 1


def test_imports_are_imported_before_evaluators(
    tmp_path: Path, write: WriteFile, monkeypatch: pytest.MonkeyPatch
) -> None:
    write(
        "custom_evals.py",
        """
        from agentlens.evals import NoError, default_registry
        default_registry.register("custom_check", NoError, replace=True)
        """,
    )
    monkeypatch.chdir(tmp_path)
    suite = make(imports=["custom_evals"], defaults={"evaluators": ["custom_check"]})
    try:
        assert suite.issues() == []
    finally:
        from agentlens.evals import default_registry

        default_registry.unregister("custom_check")


def test_bad_import_is_an_issue(tiny_registry: EvaluatorRegistry) -> None:
    (issue,) = make(imports=["nope_module_xyz"]).issues(registry=tiny_registry)
    assert issue.loc == ("imports", 0)
    assert "cannot import module 'nope_module_xyz'" in issue.message


def test_duplicate_ids_in_issues(tiny_registry: EvaluatorRegistry) -> None:
    cases = [{"id": "a", "input": 1}, {"id": "a", "input": 2}]
    suite = Suite.model_validate(
        {"name": "d", "agent": "m:f", "cases": cases}, context={"agentlens.suite.skip_checks": True}
    )
    assert suite.issues(registry=tiny_registry) == [
        SuiteIssue("duplicate case id 'a' (first: cases[0])", ("cases", 1, "id"))
    ]


def test_search_paths(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(tmp_path)
    suite = make()
    assert suite.search_paths() == [Path(os.getcwd())]
    suite._source = tmp_path / "suites" / "s.yaml"
    assert suite.source_path == tmp_path / "suites" / "s.yaml"
    assert suite.base_dir == tmp_path / "suites"
    assert suite.search_paths() == [Path(os.getcwd()), tmp_path / "suites"]
    suite._source = tmp_path / "s.yaml"  # suite dir == cwd: listed once
    assert suite.search_paths() == [Path(os.getcwd())]


def test_load_agent_errors_are_agent_import_errors() -> None:
    with pytest.raises(AgentImportError):
        make(agent="nope_module_xyz:run").load_agent()


async def _async_echo(text: str) -> str:
    return text


def test_agent_is_async() -> None:
    assert make(agent=echo).agent_is_async() is False
    assert make(agent=_async_echo).agent_is_async() is True


def test_content_hash_is_stable_and_hex() -> None:
    digest = make().content_hash()
    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert make().content_hash() == digest


def test_content_hash_ignores_description_concurrency_and_defaults_spelling() -> None:
    base = make(cases=[{"id": "a", "input": "hi", "evaluators": ["no_error"], "tags": ["t"]}])
    via_defaults = make(
        description="docs only",
        concurrency=8,
        defaults={"evaluators": ["no_error"], "tags": ["t"]},
        cases=[{"id": "a", "input": "hi"}],
    )
    assert via_defaults.content_hash() == base.content_hash()


@pytest.mark.parametrize(
    "change",
    [
        {"repetitions": 3},
        {"model": "openai:gpt-4o-mini"},
        {"agent": "pkg.mod:other"},
        {"cases": [{"id": "a", "input": "hello"}]},
        {"defaults": {"evaluators": [{"max_steps": 8}]}},
        {"timeout_s": 5.0},
    ],
)
def test_content_hash_changes_with_what_runs(change: dict[str, Any]) -> None:
    assert make(**change).content_hash() != make().content_hash()


def test_content_hash_of_callable_agent_uses_its_name() -> None:
    assert make(agent=echo).content_hash() == make(agent=f"{__name__}:echo").content_hash()
