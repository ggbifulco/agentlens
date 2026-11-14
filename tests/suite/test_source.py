from __future__ import annotations

from pathlib import Path

import pytest

from agentlens.suite import SuiteError
from agentlens.suite.errors import SuiteIssue
from agentlens.suite.source import Position, SourceMap, collect, load_json_lines, load_yaml
from tests.suite.conftest import WriteFile


def test_load_yaml_records_value_and_key_positions(write: WriteFile) -> None:
    path = write(
        "s.yaml",
        """
        name: demo
        cases:
          - id: a
            evaluators: [no_error, {max_steps: 8}]
        """,
    )
    data, smap = load_yaml(path)
    assert data == {
        "name": "demo",
        "cases": [{"id": "a", "evaluators": ["no_error", {"max_steps": 8}]}],
    }
    file = str(path)
    assert smap.values[()] == Position(file, 1, 1)
    assert smap.values[("name",)] == Position(file, 1, 7)
    assert smap.keys[("name",)] == Position(file, 1, 1)
    assert smap.values[("cases", 0)] == Position(file, 3, 5)
    assert smap.values[("cases", 0, "evaluators", 1)] == Position(file, 4, 28)
    assert smap.values[("cases", 0, "evaluators", 1, "max_steps")] == Position(file, 4, 40)


def test_locate_falls_back_to_longest_prefix(write: WriteFile) -> None:
    path = write("s.yaml", "cases:\n  - id: a\n")
    _, smap = load_yaml(path)
    assert smap.locate(("cases", 0, "input")) == Position(str(path), 2, 5)
    assert smap.locate(("cases", 0, "id"), key=True) == Position(str(path), 2, 5)
    assert smap.locate(("cases", 0, "nope"), key=True) == Position(str(path), 2, 5)
    assert SourceMap().locate(("x",)) is None
    assert SourceMap().issue(("x",), "m") == SuiteIssue("m", ("x",))


def test_mount_grafts_under_prefix() -> None:
    outer, inner = SourceMap(), SourceMap()
    inner.values[()] = Position("d.jsonl", 1, 1)
    inner.values[(0,)] = Position("d.jsonl", 3, 1)
    inner.keys[(0, "id")] = Position("d.jsonl", 3, 2)
    outer.mount(("cases",), inner)
    assert outer.values == {("cases", 0): Position("d.jsonl", 3, 1)}
    assert outer.keys == {("cases", 0, "id"): Position("d.jsonl", 3, 2)}


def test_merge_keys_and_anchors(write: WriteFile) -> None:
    path = write(
        "s.yaml",
        """
        base: &base {input: hi, tags: [x]}
        cases:
          - <<: *base
            id: a
            input: override
        """,
    )
    data, smap = load_yaml(path)
    assert data["cases"][0] == {"input": "override", "tags": ["x"], "id": "a"}
    assert ("cases", 0, "id") in smap.values


def test_recursive_alias_does_not_loop(write: WriteFile) -> None:
    path = write("s.yaml", "a: &x [1, *x]\n")
    data, _ = load_yaml(path)
    assert data["a"][1] is data["a"]


def test_complex_keys_are_skipped(write: WriteFile) -> None:
    path = write("s.yaml", "? [a, b]\n: 1\nname: x\n")
    with pytest.raises(SuiteError) as info:  # walked fine, then PyYAML refuses the list key
        load_yaml(path)
    assert "unhashable key" in str(info.value)


def test_duplicate_keys_are_errors(write: WriteFile) -> None:
    path = write("s.yaml", "name: a\ncases: []\nname: b\n")
    with pytest.raises(SuiteError) as info:
        load_yaml(path)
    (issue,) = info.value.issues
    assert str(issue) == f"{path}:3:1: name: duplicate key 'name'"


def test_syntax_error_has_position(write: WriteFile) -> None:
    path = write("s.yaml", "name: demo\ncases:\n  - id: a\n   input: [unclosed\n")
    with pytest.raises(SuiteError) as info:
        load_yaml(path)
    (issue,) = info.value.issues
    assert issue.file == str(path)
    assert issue.line == 4
    assert issue.message.startswith("invalid YAML:")
    assert str(info.value).startswith(f"invalid YAML in {path} (1 problem)")


def test_multiple_documents_are_rejected(write: WriteFile) -> None:
    path = write("s.yaml", "a: 1\n---\nb: 2\n")
    with pytest.raises(SuiteError) as info:
        load_yaml(path)
    assert "single document" in info.value.issues[0].message


@pytest.mark.parametrize("text", ["", "# only a comment\n"])
def test_empty_file(write: WriteFile, text: str) -> None:
    path = write("s.yaml", text)
    with pytest.raises(SuiteError, match="suite file is empty"):
        load_yaml(path)


def test_missing_file(tmp_path: Path) -> None:
    with pytest.raises(SuiteError, match="suite file not found"):
        load_yaml(tmp_path / "nope.yaml")
    with pytest.raises(SuiteError, match="dataset not found"):
        load_json_lines(tmp_path / "nope.jsonl")


def test_unreadable_file(tmp_path: Path) -> None:
    path = tmp_path / "bin.yaml"
    path.write_bytes(b"\xff\xfe\x00bad")
    with pytest.raises(SuiteError, match="cannot read suite file"):
        load_yaml(path)


def test_json_lines(write: WriteFile) -> None:
    path = write("d.jsonl", '{"id": "a", "input": 1}\n\n{"id": "b", "input": 2}\n')
    rows, smap = load_json_lines(path)
    assert rows == [{"id": "a", "input": 1}, {"id": "b", "input": 2}]
    assert smap.values[(1,)] == Position(str(path), 3, 1)
    assert smap.locate((1, "input")) == Position(str(path), 3, 1)


def test_json_lines_reports_every_bad_line(write: WriteFile) -> None:
    path = write("d.jsonl", '{"id": "a"}\n{"id": \n{"id": "b"}\nnope\n')
    with pytest.raises(SuiteError) as info:
        load_json_lines(path)
    assert [(i.line, i.message.split(":")[0]) for i in info.value.issues] == [
        (2, "invalid JSON"),
        (4, "invalid JSON"),
    ]


def test_collect_drops_repeats() -> None:
    a, b = SuiteIssue("a", ("x",)), SuiteIssue("b")
    assert collect([a, b, SuiteIssue("a", ("x",))]) == [a, b]
