from __future__ import annotations

from pathlib import Path

import pytest

from agentlens.evals import EvaluatorRegistry
from agentlens.schema import Case, EvaluatorSpec
from agentlens.suite import SuiteError, load_cases, load_suite
from tests.suite.conftest import WriteFile

DESIGN_EXAMPLE = """
name: support-agent
agent: support_agent:run_agent   # import path, callable(input) -> output
repetitions: 3
defaults:
  evaluators: [no_error, {max_steps: 8}]
cases:
  - id: refund-simple
    input: "I want a refund for order 1234"
    expected: {tools: [lookup_order, issue_refund]}
    evaluators:
      - {tool_sequence: {mode: ordered}}
      - {llm_rubric: {criteria: "Polite, confirms refund amount"}}
"""


def problems(path: Path, registry: EvaluatorRegistry | None = None) -> list[str]:
    """Issues of a broken suite as ``line:col: path: message`` (file name stripped)."""
    with pytest.raises(SuiteError) as info:
        load_suite(path, registry=registry)
    return [str(issue).removeprefix(f"{path}:") for issue in info.value.issues]


def test_design_example_loads(write: WriteFile, registry: EvaluatorRegistry) -> None:
    write("support_agent.py", "def run_agent(text):\n    return 'ok ' + text\n")
    path = write("suite.yaml", DESIGN_EXAMPLE)
    suite = load_suite(path, registry=registry)
    assert suite.name == "support-agent"
    assert suite.repetitions == 3
    assert suite.source_path == path
    (case,) = suite.resolved_cases()
    assert [s.name for s in case.evaluators] == [
        "no_error",
        "max_steps",
        "tool_sequence",
        "llm_rubric",
    ]
    assert case.expected == {"tools": ["lookup_order", "issue_refund"]}
    assert suite.load_agent()("hi") == "ok hi"  # found next to the suite file


def test_default_registry_is_used(write: WriteFile) -> None:
    path = write("suite.yaml", DESIGN_EXAMPLE)
    assert load_suite(path).name == "support-agent"


def test_all_optional_fields(write: WriteFile, registry: EvaluatorRegistry) -> None:
    path = write(
        "suite.yaml",
        """
        name: full
        description: Every field
        agent: pkg.mod:Agent.run
        model: openai:gpt-4o-mini
        repetitions: 2
        concurrency: 4
        timeout_s: 30
        metadata: {owner: support-team, version: 2}
        defaults: {tags: [smoke]}
        cases:
          - {id: a, input: {question: hi}, tags: [x]}
        """,
    )
    suite = load_suite(path, registry=registry)
    assert suite.concurrency == 4
    assert suite.timeout_s == 30.0
    assert suite.model == "openai:gpt-4o-mini"
    assert suite.metadata == {"owner": "support-team", "version": 2}
    assert suite.resolved_cases()[0].tags == ["smoke", "x"]


def test_error_with_line_context_for_unknown_evaluator(
    write: WriteFile, registry: EvaluatorRegistry
) -> None:
    path = write(
        "suite.yaml",
        """
        name: demo
        agent: m:f
        cases:
          - id: a
            input: hi
          - id: b
            input: yo
            evaluators:
              - no_error
              - foo
        """,
    )
    assert problems(path, registry) == ["10:9: cases[1].evaluators[1]: unknown evaluator 'foo'"]


def test_all_errors_are_listed_in_file_order(write: WriteFile, registry: EvaluatorRegistry) -> None:
    path = write(
        "suite.yaml",
        """
        name: demo
        agent: m:f
        repetitions: 0
        evalutors: [no_error]
        timeout_s: -1
        cases:
          - id: a
            input: hi
            evaluators:
              - tool_sequence:
                mode: ordered
          - input: missing id
          - id: c
            input: 3
            expectd: 4
        """,
    )
    assert problems(path, registry) == [
        "3:14: repetitions: Input should be greater than or equal to 1",
        "4:1: evalutors: unknown field 'evalutors' (did you mean 'evaluators'?)",
        "5:12: timeout_s: Input should be greater than 0",
        "10:9: cases[0].evaluators[0]: an evaluator is 'name', {name: value} or "
        "{name: {param: value}}, got a mapping with keys 'tool_sequence', 'mode' "
        "(are the parameters indented under the name?)",
        "12:5: cases[1].id: missing required field 'id'",
        "15:5: cases[2].expectd: unknown field 'expectd' (did you mean 'expected'?)",
    ]


def test_semantic_errors_after_schema(write: WriteFile, registry: EvaluatorRegistry) -> None:
    path = write(
        "suite.yaml",
        """
        name: demo
        agent: m:f
        imports: [nope_module_xyz]
        defaults:
          evaluators: [{regex: "("}]
        cases:
          - {id: a, input: 1, evaluators: [contain]}
          - {id: a, input: 2}
        """,
    )
    found = problems(path, registry)
    assert found[0].startswith("3:11: imports[0]: cannot import module 'nope_module_xyz'")
    assert found[1].startswith("5:16: defaults.evaluators[0]: cannot build evaluator 'regex'")
    assert found[2:] == [
        "7:36: cases[0].evaluators[0]: unknown evaluator 'contain' (did you mean 'contains'?)",
        "8:10: cases[1].id: duplicate case id 'a' (first: cases[0])",
    ]


def test_check_false_skips_imports_and_evaluators(write: WriteFile) -> None:
    path = write(
        "suite.yaml",
        """
        name: demo
        agent: m:f
        imports: [nope_module_xyz]
        cases: [{id: a, input: 1, evaluators: [does_not_exist]}]
        """,
    )
    assert load_suite(path, check=False).cases[0].evaluators[0].name == "does_not_exist"


def test_duplicate_ids_checked_even_without_check(write: WriteFile) -> None:
    path = write(
        "suite.yaml", "name: d\nagent: m:f\ncases: [{id: a, input: 1}, {id: a, input: 2}]\n"
    )
    with pytest.raises(SuiteError) as info:
        load_suite(path, check=False)
    assert [i.path for i in info.value.issues] == ["cases[1].id"]


@pytest.mark.parametrize(
    ("agent", "message"),
    [
        ("no_colon", "agent must look like 'package.module:callable'"),
        ("42", "agent must be an import path 'package.module:callable'"),
    ],
)
def test_bad_agent(write: WriteFile, agent: str, message: str) -> None:
    path = write("suite.yaml", f"name: d\nagent: {agent}\ncases: [{{id: a, input: 1}}]\n")
    (found,) = problems(path)
    assert found.startswith("2:8: agent: ")
    assert message.split("must")[1].split("'")[0] in found


def test_not_a_mapping(write: WriteFile) -> None:
    path = write("suite.yaml", "- just\n- a list\n")
    assert problems(path) == ["1:1: a suite must be a mapping with 'name', 'agent' and 'cases'"]


def test_missing_top_level_fields(write: WriteFile) -> None:
    path = write("suite.yaml", "description: nothing else\n")
    assert problems(path) == [
        "1:1: name: missing required field 'name'",
        "1:1: agent: missing required field 'agent'",
        "1:1: cases: missing required field 'cases'",
    ]


def test_yaml_syntax_error_has_line(write: WriteFile) -> None:
    path = write("suite.yaml", "name: demo\ncases:\n  - id: a\n   input: x\n")
    (found,) = problems(path)
    assert found.startswith("4:")
    assert "invalid YAML" in found


def test_json_value_errors_point_at_known_prefix(write: WriteFile) -> None:
    path = write("suite.yaml", "name: d\nagent: m:f\ncases: [{id: a, input: 2024-01-01}]\n")
    found = problems(path)
    assert found
    assert all(f.startswith("3:") and "cases[0].input" in f for f in found)


# ------------------------------------------------------------------ external datasets


def test_external_jsonl_dataset(write: WriteFile, registry: EvaluatorRegistry) -> None:
    write(
        "data/cases.jsonl",
        '{"id": "a", "input": "hi", "evaluators": [{"max_steps": 3}]}\n'
        '{"id": "b", "input": "yo", "tags": ["t"]}\n',
    )
    path = write(
        "suites/suite.yaml",
        """
        name: ext
        agent: m:f
        defaults: {evaluators: [no_error, {max_steps: 8}]}
        cases: ../data/cases.jsonl
        """,
    )
    suite = load_suite(path, registry=registry)
    assert [c.id for c in suite.cases] == ["a", "b"]
    a, b = suite.resolved_cases()
    assert a.evaluators == [
        EvaluatorSpec(name="no_error"),
        EvaluatorSpec.model_validate({"max_steps": 3}),
    ]
    assert b.tags == ["t"]


@pytest.mark.parametrize("suffix", [".yaml", ".yml", ".json"])
def test_external_yaml_or_json_dataset(
    write: WriteFile, registry: EvaluatorRegistry, suffix: str
) -> None:
    write(f"cases{suffix}", '[{"id": "a", "input": "hi"}]\n')
    path = write("suite.yaml", f"name: ext\nagent: m:f\ncases: cases{suffix}\n")
    assert load_suite(path, registry=registry).cases == [Case(id="a", input="hi")]


def test_external_dataset_absolute_path(
    tmp_path: Path, write: WriteFile, registry: EvaluatorRegistry
) -> None:
    data = write("elsewhere/cases.yaml", "- {id: a, input: 1}\n")
    path = write("suite.yaml", f"name: ext\nagent: m:f\ncases: {data}\n")
    assert len(load_suite(path, registry=registry).cases) == 1


def test_errors_in_external_dataset_point_into_it(
    write: WriteFile, registry: EvaluatorRegistry
) -> None:
    data = write("cases.yaml", "- id: a\n  input: 1\n- id: a\n  input: 2\n  evaluators: [nope]\n")
    path = write("suite.yaml", "name: ext\nagent: m:f\ncases: cases.yaml\n")
    with pytest.raises(SuiteError) as info:
        load_suite(path, registry=registry)
    assert [str(i) for i in info.value.issues] == [
        f"{data}:3:7: cases[1].id: duplicate case id 'a' (first: cases[0])",
        f"{data}:5:16: cases[1].evaluators[0]: unknown evaluator 'nope'",
    ]


def test_schema_errors_in_jsonl_point_at_the_row(write: WriteFile) -> None:
    data = write("cases.jsonl", '{"id": "a", "input": 1}\n{"input": 2}\n')
    path = write("suite.yaml", "name: ext\nagent: m:f\ncases: cases.jsonl\n")
    with pytest.raises(SuiteError) as info:
        load_suite(path)
    assert [str(i) for i in info.value.issues] == [
        f"{data}:2:1: cases[1].id: missing required field 'id'"
    ]


def test_missing_external_dataset(write: WriteFile) -> None:
    path = write("suite.yaml", "name: ext\nagent: m:f\ncases: nope.jsonl\n")
    (found,) = problems(path)
    assert found.startswith("3:8: cases: dataset not found: ")


def test_unsupported_dataset_format(write: WriteFile) -> None:
    write("cases.csv", "id,input\n")
    path = write("suite.yaml", "name: ext\nagent: m:f\ncases: cases.csv\n")
    (found,) = problems(path)
    assert "unsupported dataset format '.csv'" in found


def test_bad_json_lines_in_external_dataset(write: WriteFile) -> None:
    data = write("cases.jsonl", '{"id": "a", "input": 1}\n{oops\n')
    path = write("suite.yaml", "name: ext\nagent: m:f\ncases: cases.jsonl\n")
    with pytest.raises(SuiteError) as info:
        load_suite(path)
    (issue,) = info.value.issues
    assert (issue.file, issue.line) == (str(data), 2)


def test_empty_external_dataset(write: WriteFile) -> None:
    write("cases.jsonl", "\n")
    path = write("suite.yaml", "name: ext\nagent: m:f\ncases: cases.jsonl\n")
    assert problems(path) == [
        "3:8: cases: List should have at least 1 item after validation, not 0"
    ]


# ------------------------------------------------------------------ load_cases


def test_load_cases(write: WriteFile) -> None:
    path = write("cases.jsonl", '{"id": "a", "input": 1}\n{"id": "b", "input": 2, "tags": ["x"]}\n')
    assert load_cases(path) == [Case(id="a", input=1), Case(id="b", input=2, tags=["x"])]


def test_load_cases_errors(write: WriteFile) -> None:
    path = write("cases.yaml", "- {id: a, input: 1}\n- {id: a, input: 2}\n- {input: 3}\n")
    with pytest.raises(SuiteError) as info:
        load_cases(path)
    assert [str(i) for i in info.value.issues] == [
        f"{path}:3:3: [2].id: missing required field 'id'"
    ]
    path = write("dups.yaml", "- {id: a, input: 1}\n- {id: a, input: 2}\n")
    with pytest.raises(SuiteError) as info:
        load_cases(path)
    assert [str(i) for i in info.value.issues] == [
        f"{path}:2:8: [1].id: duplicate case id 'a' (first: [0])"
    ]


def test_load_cases_not_a_list(write: WriteFile) -> None:
    path = write("cases.yaml", "id: a\ninput: 1\n")
    with pytest.raises(SuiteError) as info:
        load_cases(path)
    assert str(info.value.issues[0]) == f"{path}:1:1: a dataset must be a list of cases, got dict"


def test_load_cases_unsupported_suffix(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match=r"unsupported dataset format '\(none\)'"):
        load_cases(tmp_path / "cases")


def test_private_helpers_edge_cases() -> None:
    from agentlens.suite.loader import _known_prefix, _raw_at
    from agentlens.suite.source import SourceMap

    assert _raw_at({"a": [1]}, ("a", 0)) == 1
    assert _raw_at({"a": [1]}, ("a", 5)) is None
    assert _raw_at({"a": 1}, ("b",)) is None
    assert _raw_at(3, ("a",)) is None
    assert _known_prefix(("a", "b"), SourceMap()) == ()
