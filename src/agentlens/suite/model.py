"""The :class:`Suite` model: an agent, a dataset of cases and how to run them (FR-S1).

A suite is usually loaded from YAML with :func:`~agentlens.suite.load_suite`,
but it is a plain pydantic model, so it can be built in Python too::

    from agentlens.suite import Suite

    suite = Suite(
        name="support-agent",
        agent="examples.support_agent:run_agent",   # or the function itself
        defaults={"evaluators": ["no_error", {"max_steps": 8}]},
        cases=[{"id": "refund", "input": "refund order 1234"}],
    )
    for case in suite.resolved_cases():   # defaults merged in
        ...
"""

from __future__ import annotations

import os
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from pydantic import (
    Field,
    JsonValue,
    PrivateAttr,
    ValidationInfo,
    field_serializer,
    field_validator,
    model_validator,
)

from agentlens.evals import EvaluatorError, EvaluatorRegistry, UnknownEvaluatorError
from agentlens.evals.registry import default_registry
from agentlens.schema import Case, EvaluatorSpec
from agentlens.schema._base import SchemaModel
from agentlens.suite.agent import check_import_path, import_module, resolve_agent
from agentlens.suite.errors import AgentImportError, Loc, SuiteError, SuiteIssue

__all__ = ["SKIP_CHECKS", "Suite", "SuiteDefaults", "merge_evaluators"]

SKIP_CHECKS = "agentlens.suite.skip_checks"
"""Validation-context key: skip the cross-case checks in the model validator
(the loader runs them itself to report each problem at its line)."""


def merge_evaluators(
    defaults: Iterable[EvaluatorSpec], own: Iterable[EvaluatorSpec]
) -> list[EvaluatorSpec]:
    """Suite defaults first, then the case's own evaluators.

    A case evaluator **overrides** every default with the same name, so a case
    can tighten ``{max_steps: 8}`` to ``{max_steps: 3}`` without running both.
    Different names accumulate. Merging is idempotent.
    """
    own = list(own)
    names = {spec.name for spec in own}
    return [spec for spec in defaults if spec.name not in names] + own


def _merge_tags(*groups: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(tag for group in groups for tag in group))


class SuiteDefaults(SchemaModel):
    """Settings applied to every case of a suite (``defaults:`` in YAML)."""

    evaluators: list[EvaluatorSpec] = Field(default_factory=list)
    """Evaluators run on every case (see :func:`merge_evaluators` for overrides)."""
    tags: list[str] = Field(default_factory=list)
    """Tags added to every case (before the case's own, without duplicates)."""


class Suite(SchemaModel):
    """A test suite: the agent under test, its dataset and run settings.

    ``cases`` keep exactly what was declared; :meth:`resolved_cases` returns them
    with the suite :attr:`defaults` merged in, which is what the runner executes.
    Case ids must be unique (they pair cases across experiments when comparing).
    """

    name: str = Field(min_length=1)
    """Suite name; experiments are grouped by it."""
    agent: str | Callable[..., Any]
    """The agent under test: ``"package.module:callable"`` (see :func:`resolve_agent`)
    or, for Python-defined suites, the callable itself. Called with each case input."""
    description: str | None = None
    """Free text for humans and reports."""
    model: str | None = None
    """The LLM under test (``"openai:gpt-4o-mini"``), recorded with the experiment;
    informational for the runner and agents that read it."""
    repetitions: int = Field(default=1, ge=1)
    """How many times each case is run (FR-S2), to measure non-determinism."""
    concurrency: int = Field(default=1, ge=1)
    """Maximum runs executing at the same time."""
    timeout_s: float | None = Field(default=None, gt=0)
    """Per-run timeout in seconds; ``None`` means no limit."""
    imports: list[str] = Field(default_factory=list)
    """Modules imported before the evaluators are checked, e.g. the module that
    defines custom ``@agentlens.evaluator`` functions used by the cases."""
    metadata: dict[str, JsonValue] = Field(default_factory=dict)
    """Anything else worth recording with the experiment."""
    defaults: SuiteDefaults = Field(default_factory=SuiteDefaults)
    cases: list[Case] = Field(min_length=1)
    """The dataset, as declared (see :meth:`resolved_cases`)."""

    _source: Path | None = PrivateAttr(default=None)

    @field_validator("agent")
    @classmethod
    def _check_agent(cls, value: str | Callable[..., Any]) -> str | Callable[..., Any]:
        if isinstance(value, str):
            check_import_path(value)
        return value

    @field_validator("imports")
    @classmethod
    def _check_imports(cls, value: list[str]) -> list[str]:
        for name in value:
            if not name or not all(part.isidentifier() for part in name.split(".")):
                raise ValueError(f"imports must be dotted module names, got {name!r}")
        return value

    @field_serializer("agent")
    def _serialise_agent(self, value: str | Callable[..., Any]) -> str:
        if isinstance(value, str):
            return value
        module = getattr(value, "__module__", None) or "__main__"
        qualname = getattr(value, "__qualname__", None) or type(value).__qualname__
        return f"{module}:{qualname}"

    @model_validator(mode="after")
    def _unique_ids(self, info: ValidationInfo) -> Suite:
        if info.context and info.context.get(SKIP_CHECKS):
            return self
        duplicates = [issue.message for issue in self._duplicate_ids()]
        if duplicates:
            raise ValueError("; ".join(duplicates))
        return self

    # ------------------------------------------------------------------ paths

    @property
    def source_path(self) -> Path | None:
        """The YAML file this suite was loaded from (``None`` for Python suites)."""
        return self._source

    @property
    def base_dir(self) -> Path | None:
        """Directory of :attr:`source_path`: relative paths in the suite resolve here."""
        return self._source.parent if self._source is not None else None

    def search_paths(self) -> list[Path]:
        """Directories appended to ``sys.path`` before importing the agent:
        the current working directory, then :attr:`base_dir` (if any)."""
        paths = [Path(os.getcwd())]
        if self.base_dir is not None and self.base_dir.resolve() != paths[0].resolve():
            paths.append(self.base_dir)
        return paths

    # ------------------------------------------------------------------ cases

    def resolved_cases(self) -> list[Case]:
        """The cases with :attr:`defaults` merged in (new objects; ``cases`` is unchanged).

        Evaluators: defaults first, then the case's own; a case evaluator replaces
        defaults of the same name (:func:`merge_evaluators`). Tags: defaults first,
        then the case's own, without duplicates.
        """
        return [
            case.model_copy(
                update={
                    "evaluators": merge_evaluators(self.defaults.evaluators, case.evaluators),
                    "tags": _merge_tags(self.defaults.tags, case.tags),
                }
            )
            for case in self.cases
        ]

    def case(self, case_id: str) -> Case:
        """The resolved case with id ``case_id`` (``KeyError`` if none)."""
        for case in self.resolved_cases():
            if case.id == case_id:
                return case
        raise KeyError(case_id)

    # ------------------------------------------------------------------ agent

    def load_agent(self) -> Callable[..., Any]:
        """Import and return the agent callable (see :func:`resolve_agent`).

        Raises:
            AgentImportError: if the import path cannot be resolved.
        """
        return resolve_agent(self.agent, search_paths=self.search_paths())

    # ------------------------------------------------------------------ checks

    def check(self, *, registry: EvaluatorRegistry | None = None) -> Suite:
        """Check what the model alone cannot: imports and evaluators.

        Imports the :attr:`imports` modules, then builds every evaluator spec
        (defaults and per case) from ``registry`` (default: the process-wide
        one), so unknown names and bad parameters fail now rather than mid-run.
        Built evaluators are discarded; LLM judges need no API key to be built.

        Returns:
            The suite itself, for chaining.

        Raises:
            SuiteError: listing every problem found.
        """
        issues = self.issues(registry=registry)
        if issues:
            raise SuiteError(f"invalid suite {self.name!r}", issues)
        return self

    def issues(self, *, registry: EvaluatorRegistry | None = None) -> list[SuiteIssue]:
        """Every problem :meth:`check` would raise, without raising (no file positions)."""
        issues = self._duplicate_ids()
        for index, name in enumerate(self.imports):
            try:
                import_module(name, search_paths=self.search_paths())
            except AgentImportError as exc:
                issues.append(SuiteIssue(str(exc), ("imports", index)))
        issues.extend(self._evaluator_issues(registry or default_registry))
        return issues

    def _duplicate_ids(self) -> list[SuiteIssue]:
        first: dict[str, int] = {}
        issues: list[SuiteIssue] = []
        for index, case in enumerate(self.cases):
            if case.id in first:
                issues.append(
                    SuiteIssue(
                        f"duplicate case id {case.id!r} (first: cases[{first[case.id]}])",
                        ("cases", index, "id"),
                    )
                )
            else:
                first[case.id] = index
        return issues

    def _evaluator_issues(self, registry: EvaluatorRegistry) -> list[SuiteIssue]:
        specs: list[tuple[Loc, EvaluatorSpec]] = [
            (("defaults", "evaluators", i), spec) for i, spec in enumerate(self.defaults.evaluators)
        ]
        for c, case in enumerate(self.cases):
            specs += [
                (("cases", c, "evaluators", i), spec) for i, spec in enumerate(case.evaluators)
            ]
        problems: dict[str, str | None] = {}  # spec JSON -> problem, each spec built once
        issues: list[SuiteIssue] = []
        for loc, spec in specs:
            key = spec.model_dump_json()
            if key not in problems:
                problems[key] = _build_problem(registry, spec)
            problem = problems[key]
            if problem is not None:
                issues.append(SuiteIssue(problem, loc))
        return issues


def _build_problem(registry: EvaluatorRegistry, spec: EvaluatorSpec) -> str | None:
    try:
        registry.get(spec)
    except UnknownEvaluatorError as exc:
        hint = exc.suggestions
        did_you_mean = f" (did you mean {' or '.join(repr(s) for s in hint)}?)" if hint else ""
        return f"unknown evaluator {exc.name!r}{did_you_mean}"
    except EvaluatorError as exc:
        return str(exc)
    except Exception as exc:  # a user factory failing in an unexpected way
        return f"cannot build evaluator {spec.name!r}: {type(exc).__name__}: {exc}"
    return None
