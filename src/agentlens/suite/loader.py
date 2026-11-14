"""Loading suites and datasets from files, with errors at ``file:line:column``.

:func:`load_suite` parses the YAML of ``docs/design.md`` §6 into a
:class:`~agentlens.suite.Suite`, validates it (schema, unique case ids, imports,
evaluator names and parameters against the registry) and raises one
:class:`~agentlens.suite.SuiteError` listing every problem, each at its line::

    suites/support.yaml:12:9: cases[2].evaluators[0]: unknown evaluator 'contain' (did you
    mean 'contains'?)

The dataset may live in its own file: ``cases: data/support.jsonl`` (or
``.yaml``/``.yml``/``.json`` holding a list of cases), relative to the suite file.
"""

from __future__ import annotations

import difflib
import os
from pathlib import Path
from typing import Any

from pydantic import TypeAdapter, ValidationError

from agentlens.evals import EvaluatorRegistry
from agentlens.schema import Case, EvaluatorSpec
from agentlens.suite.errors import Loc, SuiteError, SuiteIssue
from agentlens.suite.model import SKIP_CHECKS, Suite, SuiteDefaults
from agentlens.suite.source import SourceMap, collect, load_json_lines, load_yaml

__all__ = ["DATASET_SUFFIXES", "load_cases", "load_suite"]

DATASET_SUFFIXES = (".jsonl", ".yaml", ".yml", ".json")
"""File types accepted for an external dataset (``cases: path``) and by :func:`load_cases`."""

_FIELDS = sorted(
    {
        *Suite.model_fields,
        *SuiteDefaults.model_fields,
        *Case.model_fields,
        *EvaluatorSpec.model_fields,
    }
)
_CASES = TypeAdapter(list[Case])


def load_suite(
    path: str | os.PathLike[str],
    *,
    registry: EvaluatorRegistry | None = None,
    check: bool = True,
) -> Suite:
    """Load and validate a suite YAML file.

    Args:
        path: The suite file.
        registry: Registry to check evaluator specs against (default: the
            process-wide one).
        check: Also run :meth:`Suite.check` (import ``imports`` modules and build
            every evaluator spec). Schema and unique-id checks always run. The
            agent is *not* imported here; see :meth:`Suite.load_agent`.

    Raises:
        SuiteError: listing every problem found, each with its file position.
    """
    suite_path = Path(path)
    data, smap = load_yaml(suite_path)
    if not isinstance(data, dict):
        raise SuiteError(
            f"invalid suite {suite_path}",
            [smap.issue((), "a suite must be a mapping with 'name', 'agent' and 'cases'")],
        )
    if isinstance(data.get("cases"), str):
        data = {**data, "cases": _external_cases(suite_path, data["cases"], smap)}
    try:
        suite = Suite.model_validate(data, context={SKIP_CHECKS: True})
    except ValidationError as exc:
        raise SuiteError(
            f"invalid suite {suite_path}", _by_position(_validation_issues(exc, data, smap))
        ) from None
    suite._source = suite_path
    issues = suite.issues(registry=registry) if check else suite._duplicate_ids()
    if issues:
        raise SuiteError(
            f"invalid suite {suite_path}",
            _by_position(collect(smap.issue(issue.loc, issue.message) for issue in issues)),
        )
    return suite


def load_cases(path: str | os.PathLike[str]) -> list[Case]:
    """Load a dataset file (``.jsonl``, ``.yaml``/``.yml`` or ``.json`` list of cases).

    Raises:
        SuiteError: listing every invalid row or duplicate id, with positions.
    """
    dataset_path = Path(path)
    rows, smap = _read_dataset(dataset_path)
    try:
        cases = _CASES.validate_python(rows)
    except ValidationError as exc:
        raise SuiteError(
            f"invalid dataset {dataset_path}", _validation_issues(exc, rows, smap)
        ) from None
    first: dict[str, int] = {}
    issues: list[SuiteIssue] = []
    for index, case in enumerate(cases):
        if case.id in first:
            message = f"duplicate case id {case.id!r} (first: [{first[case.id]}])"
            issues.append(smap.issue((index, "id"), message))
        first.setdefault(case.id, index)
    if issues:
        raise SuiteError(f"invalid dataset {dataset_path}", issues)
    return cases


def _read_dataset(path: Path) -> tuple[list[Any], SourceMap]:
    suffix = path.suffix.lower()
    if suffix not in DATASET_SUFFIXES:
        raise SuiteError(
            f"unsupported dataset format {path.suffix or '(none)'!r} for {path}: "
            f"use one of {', '.join(DATASET_SUFFIXES)}"
        )
    if suffix == ".jsonl":
        return load_json_lines(path)
    data, smap = load_yaml(path, label="dataset")  # JSON is read as YAML to keep positions
    if not isinstance(data, list):
        raise SuiteError(
            f"invalid dataset {path}",
            [smap.issue((), f"a dataset must be a list of cases, got {type(data).__name__}")],
        )
    return data, smap


def _external_cases(suite_path: Path, ref: str, smap: SourceMap) -> list[Any]:
    path = Path(ref)
    if not path.is_absolute():
        path = suite_path.parent / path
    try:
        rows, dataset_map = _read_dataset(path)
    except SuiteError as exc:
        issues = exc.issues or [smap.issue(("cases",), exc.summary)]
        raise SuiteError(f"invalid suite {suite_path}", issues) from None
    smap.mount(("cases",), dataset_map)
    return rows


def _by_position(issues: list[SuiteIssue]) -> list[SuiteIssue]:
    return sorted(issues, key=lambda i: (i.file or "", i.line or 0, i.column or 0))


def _raw_at(data: Any, loc: Loc) -> Any:
    for part in loc:
        try:
            data = data[part]
        except (KeyError, IndexError, TypeError):
            return None
    return data


def _evaluator_loc(loc: Loc) -> Loc | None:
    for i in range(len(loc) - 1):
        if loc[i] == "evaluators" and isinstance(loc[i + 1], int):
            return loc[: i + 2]
    return None


def _known_prefix(loc: Loc, smap: SourceMap) -> Loc:
    for end in range(len(loc), -1, -1):
        if loc[:end] in smap.values:
            return loc[:end]
    return ()


def _validation_issues(exc: ValidationError, data: Any, smap: SourceMap) -> list[SuiteIssue]:
    issues: list[SuiteIssue] = []
    for err in exc.errors(include_url=False):
        loc: Loc = tuple(err["loc"])
        kind = err["type"]
        spec_loc = _evaluator_loc(loc)
        raw_spec = _raw_at(data, spec_loc) if spec_loc is not None else None
        if (
            spec_loc is not None
            and isinstance(raw_spec, dict)
            and len(raw_spec) > 1
            and not set(raw_spec) <= {"name", "params"}
        ):
            keys = ", ".join(repr(str(k)) for k in raw_spec)
            issues.append(
                smap.issue(
                    spec_loc,
                    "an evaluator is 'name', {name: value} or {name: {param: value}}, "
                    f"got a mapping with keys {keys} (are the parameters indented under the name?)",
                )
            )
        elif loc[:1] == ("agent",) and kind != "value_error":
            issues.append(
                smap.issue(("agent",), "agent must be an import path 'package.module:callable'")
            )
        elif kind == "extra_forbidden":
            field = str(loc[-1])
            close = difflib.get_close_matches(field, _FIELDS, n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            issues.append(smap.issue(loc, f"unknown field {field!r}{hint}", key=True))
        elif kind == "missing":
            issues.append(smap.issue(loc, f"missing required field {loc[-1]!r}"))
        else:
            message = str(err["msg"]).removeprefix("Value error, ")
            issues.append(smap.issue(_known_prefix(loc, smap), message))
    return collect(issues)
