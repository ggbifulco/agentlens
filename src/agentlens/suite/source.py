"""YAML/JSONL parsing that remembers where every value came from.

``yaml.safe_load`` returns plain data and forgets positions. To report
``suite.yaml:12:9: cases[2].evaluators[0]: ...`` we compose the YAML node tree
with :class:`yaml.SafeLoader`, record the start mark of every node under its
path (``("cases", 2, "evaluators", 0)``) in a :class:`SourceMap`, then construct
the data from the same tree. Validation errors carry a path, and the source
map turns it back into a file position (the longest recorded prefix of it).
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from agentlens.suite.errors import Loc, SuiteError, SuiteIssue

__all__ = ["Position", "SourceMap", "load_json_lines", "load_yaml"]


@dataclass(frozen=True)
class Position:
    """A 1-based position in a file."""

    file: str
    line: int
    column: int


@dataclass
class SourceMap:
    """Positions of the values (and mapping keys) of a parsed document, by path."""

    values: dict[Loc, Position] = field(default_factory=dict)
    keys: dict[Loc, Position] = field(default_factory=dict)
    """Position of the *key* of a mapping entry, by the entry's path (used for
    "unknown field" errors, which are about the key, not its value)."""

    def locate(self, loc: Loc, *, key: bool = False) -> Position | None:
        """Position of ``loc``, else of its longest recorded prefix (``None`` if none).

        With ``key=True`` the position of the mapping key at ``loc`` is preferred.
        """
        if key and loc in self.keys:
            return self.keys[loc]
        for end in range(len(loc), -1, -1):
            found = self.values.get(loc[:end])
            if found is not None:
                return found
        return None

    def mount(self, prefix: Loc, other: SourceMap) -> None:
        """Graft ``other`` (e.g. an external dataset) under ``prefix``; its root is skipped."""
        self.values.update({prefix + loc: pos for loc, pos in other.values.items() if loc})
        self.keys.update({prefix + loc: pos for loc, pos in other.keys.items() if loc})

    def issue(self, loc: Loc, message: str, *, key: bool = False) -> SuiteIssue:
        """A :class:`SuiteIssue` for ``loc``, positioned with :meth:`locate`."""
        pos = self.locate(loc, key=key)
        if pos is None:
            return SuiteIssue(message, loc)
        return SuiteIssue(message, loc, pos.file, pos.line, pos.column)


def _position(file: str, mark: yaml.Mark) -> Position:
    return Position(file, mark.line + 1, mark.column + 1)


def _walk(
    node: yaml.Node,
    loc: Loc,
    file: str,
    smap: SourceMap,
    issues: list[SuiteIssue],
    ancestors: frozenset[int] = frozenset(),
) -> None:
    smap.values[loc] = _position(file, node.start_mark)
    if id(node) in ancestors:  # recursive alias (&a [*a]): the data is invalid anyway
        return
    inner = ancestors | {id(node)}
    if isinstance(node, yaml.MappingNode):
        seen: set[str] = set()
        for key_node, value_node in node.value:
            if not isinstance(key_node, yaml.ScalarNode) or key_node.tag.endswith(":merge"):
                continue  # complex keys / merge keys (<<): no stable path to record
            key = str(key_node.value)
            child = (*loc, key)
            if key in seen:
                pos = _position(file, key_node.start_mark)
                issues.append(
                    SuiteIssue(f"duplicate key {key!r}", child, pos.file, pos.line, pos.column)
                )
                continue
            seen.add(key)
            smap.keys[child] = _position(file, key_node.start_mark)
            _walk(value_node, child, file, smap, issues, inner)
    elif isinstance(node, yaml.SequenceNode):
        for index, item in enumerate(node.value):
            _walk(item, (*loc, index), file, smap, issues, inner)


def _read(path: Path, label: str) -> str:
    try:
        return path.read_text(encoding="utf-8")
    except FileNotFoundError:
        raise SuiteError(f"{label} not found: {path}") from None
    except (OSError, UnicodeDecodeError) as exc:
        raise SuiteError(f"cannot read {label} {path}: {exc}") from exc


def load_yaml(path: Path, *, label: str = "suite file") -> tuple[Any, SourceMap]:
    """Parse one YAML (or JSON) document, returning ``(data, source_map)``.

    Raises:
        SuiteError: if the file cannot be read, is empty, is not valid YAML
            (positioned at the syntax error) or has duplicate mapping keys.
    """
    file = str(path)
    text = _read(path, label)
    smap = SourceMap()
    issues: list[SuiteIssue] = []
    loader = yaml.SafeLoader(text)
    try:
        node = loader.get_single_node()
        if node is None:
            raise SuiteError(f"{label} is empty: {file}")
        # Walk before constructing: construction flattens merge keys (<<) in place.
        _walk(node, (), file, smap, issues)
        data = loader.construct_document(node)
    except yaml.MarkedYAMLError as exc:
        mark = exc.problem_mark or exc.context_mark
        problem = " ".join(p for p in (exc.context, exc.problem) if p) or "invalid YAML"
        if mark is None:  # pragma: no cover - PyYAML always sets one of the marks
            raise SuiteError(f"invalid YAML in {file}", [SuiteIssue(problem)]) from exc
        pos = _position(file, mark)
        issue = SuiteIssue(f"invalid YAML: {problem}", (), pos.file, pos.line, pos.column)
        raise SuiteError(f"invalid YAML in {file}", [issue]) from exc
    except yaml.YAMLError as exc:  # pragma: no cover - reader errors without a mark
        raise SuiteError(f"invalid YAML in {file}: {exc}") from exc
    finally:
        loader.dispose()
    if issues:
        raise SuiteError(f"invalid YAML in {file}", issues)
    return data, smap


def load_json_lines(path: Path, *, label: str = "dataset") -> tuple[list[Any], SourceMap]:
    """Parse a JSON Lines file (one JSON value per line; blank lines skipped).

    Each row is mapped to its line; positions inside a row are not tracked.

    Raises:
        SuiteError: if the file cannot be read or some lines are not valid JSON
            (all bad lines are reported).
    """
    file = str(path)
    rows: list[Any] = []
    smap = SourceMap()
    smap.values[()] = Position(file, 1, 1)
    issues: list[SuiteIssue] = []
    for lineno, line in enumerate(_read(path, label).splitlines(), start=1):
        if not line.strip():
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError as exc:
            issues.append(SuiteIssue(f"invalid JSON: {exc.msg}", (), file, lineno, exc.colno))
            continue
        smap.values[(len(rows),)] = Position(file, lineno, 1)
        rows.append(row)
    if issues:
        raise SuiteError(f"invalid JSON Lines in {file}", issues)
    return rows, smap


def collect(issues: Iterable[SuiteIssue]) -> list[SuiteIssue]:
    """Drop repeated issues (same position and message), keeping order."""
    seen: set[tuple[Any, ...]] = set()
    out: list[SuiteIssue] = []
    for issue in issues:
        key = (issue.file, issue.line, issue.column, issue.loc, issue.message)
        if key not in seen:
            seen.add(key)
            out.append(issue)
    return out
