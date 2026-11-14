"""Quality gates: thresholds that turn a comparison into a pass/fail verdict (FR-S5).

A gates file maps a metric to the bounds it must respect (``docs/design.md`` §7)::

    # gates.yaml
    success_rate: {min: 0.9}           # candidate pass rate >= 0.9
    cost_usd: {max_delta_pct: 20}      # cost may grow by at most 20 % vs the baseline
    score.llm_rubric: {min_delta: -0.05}
    regressions: {max: 0}              # no metric may regress (bootstrap CI, ADR 0005)
    options: {min_cases: 20, on_missing: fail}

::

    from agentlens.compare import compare_experiments, evaluate_gates, load_gates

    report = evaluate_gates(load_gates("gates.yaml"), comparison=compare_experiments(...))
    print(report.summary())
    raise SystemExit(report.exit_code)       # 0 = every gate passed, 1 = a gate failed

**Metrics** are the comparison's (``pass_rate``, ``score.<evaluator>``,
``cost_usd``, ``latency_ms``, ``steps``) plus the aliases ``success_rate``,
``cost`` and ``latency``, and two counts that need a comparison:
``regressions`` (aggregate metrics whose CI lies entirely on the bad side) and
``case_regressions`` (cases whose pass rate dropped).

**Rules**: ``min``/``max`` bound the candidate's value; ``min_delta``/
``max_delta`` bound ``candidate - baseline``; ``min_delta_pct``/``max_delta_pct``
bound the change in percent of the baseline. Delta rules use the point
estimate; whether a change is beyond noise is what ``regressions`` checks.
"""

from __future__ import annotations

import difflib
import math
import os
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from agentlens.compare.comparator import PASS_RATE, SCORE_PREFIX, run_metrics
from agentlens.compare.model import Comparison, Status
from agentlens.schema._base import SchemaModel
from agentlens.suite.errors import Loc, SuiteError, SuiteIssue
from agentlens.suite.runner import ExperimentRun, RunResult
from agentlens.suite.source import SourceMap, collect, load_yaml

__all__ = [
    "CASE_REGRESSIONS",
    "COUNT_GATES",
    "DEFAULT_MIN_CASES",
    "EXIT_GATE_FAILED",
    "EXIT_OK",
    "METRIC_ALIASES",
    "REGRESSIONS",
    "GateReport",
    "GateResult",
    "GateRule",
    "Gates",
    "GatesError",
    "OnMissing",
    "evaluate_gates",
    "load_gates",
    "resolve_metric",
]

EXIT_OK = 0
"""Process exit code when every gate passed."""
EXIT_GATE_FAILED = 1
"""Process exit code when at least one gate failed."""

DEFAULT_MIN_CASES = 20
"""Below this many cases the report carries a (non-failing) note, after ADR 0005."""

REGRESSIONS = "regressions"
"""Gate on the number of aggregate metrics that regressed (CI on the bad side)."""
CASE_REGRESSIONS = "case_regressions"
"""Gate on the number of cases whose pass rate dropped."""
COUNT_GATES = frozenset({REGRESSIONS, CASE_REGRESSIONS})
"""Gates on counts taken from a :class:`Comparison`; only ``min``/``max`` apply."""

METRIC_ALIASES: dict[str, str] = {
    "success_rate": PASS_RATE,
    "cost": "cost_usd",
    "latency": "latency_ms",
}
"""Friendlier names accepted in a gates file, mapped to comparison metric names."""

_METRICS = (PASS_RATE, "cost_usd", "latency_ms", "steps")
_KNOWN_NAMES = sorted({*_METRICS, *METRIC_ALIASES, *COUNT_GATES})
_OPTIONS = "options"

OnMissing = Literal["fail", "skip"]
"""What a gate does when its metric has no data: fail (default) or be skipped."""

_ABSOLUTE = ("min", "max")
_DELTA = ("min_delta", "max_delta", "min_delta_pct", "max_delta_pct")
_BOUNDS = (*_ABSOLUTE, *_DELTA)
_RULE_FIELDS = (*_BOUNDS, "on_missing")
_PAIRS = (("min", "max"), ("min_delta", "max_delta"), ("min_delta_pct", "max_delta_pct"))


def resolve_metric(name: str) -> str:
    """The comparison metric a gate name refers to (aliases resolved).

    Raises:
        ValueError: for an unknown name, with a "did you mean" hint.
    """
    name = METRIC_ALIASES.get(name, name)
    if name in _METRICS or name in COUNT_GATES:
        return name
    if name.startswith(SCORE_PREFIX) and len(name) > len(SCORE_PREFIX):
        return name
    close = difflib.get_close_matches(name, _KNOWN_NAMES, n=1)
    hint = f" (did you mean {close[0]!r}?)" if close else ""
    known = ", ".join([*_KNOWN_NAMES, "score.<evaluator>"])
    raise ValueError(f"unknown metric {name!r}{hint}; known: {known}")


class GateRule(SchemaModel):
    """The bounds one metric must respect; at least one is required.

    ``min``/``max`` apply to the candidate's value, the ``*_delta`` bounds to
    ``candidate - baseline`` and the ``*_delta_pct`` bounds to that change in
    percent of the baseline. Every bound is inclusive.
    """

    min: float | None = None
    max: float | None = None
    min_delta: float | None = None
    max_delta: float | None = None
    min_delta_pct: float | None = None
    max_delta_pct: float | None = None
    on_missing: OnMissing | None = None
    """Overrides :attr:`Gates.on_missing` for this metric."""

    @field_validator(*_BOUNDS, mode="before")
    @classmethod
    def _number(cls, value: Any) -> Any:
        if isinstance(value, bool):
            raise ValueError("must be a number, not a boolean")
        return value

    @field_validator(*_BOUNDS)
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("must be a finite number")
        return value

    @model_validator(mode="after")
    def _consistent(self) -> GateRule:
        if not self.bounds():
            raise ValueError(f"a gate needs at least one of {', '.join(_BOUNDS)}")
        for low, high in _PAIRS:
            lo, hi = getattr(self, low), getattr(self, high)
            if lo is not None and hi is not None and lo > hi:
                raise ValueError(f"{low} ({lo:g}) is greater than {high} ({hi:g})")
        return self

    def bounds(self) -> list[tuple[str, float]]:
        """The bounds that are set, as ``(rule, threshold)`` in declaration order."""
        out: list[tuple[str, float]] = []
        for rule in _BOUNDS:
            value = getattr(self, rule)
            if value is not None:
                out.append((rule, value))
        return out


def _rule_problems(metric: str, rule: GateRule) -> list[tuple[str, str]]:
    """``(rule field, message)`` for bounds that do not apply to ``metric``."""
    if metric not in COUNT_GATES:
        return []
    return [
        (field, f"{metric!r} is a count of the comparison: only min/max apply, not {field!r}")
        for field, _ in rule.bounds()
        if field in _DELTA
    ]


class Gates(SchemaModel):
    """A set of gates, usually loaded with :func:`load_gates`.

    ``rules`` is keyed by the name as written (``success_rate``); see
    :func:`resolve_metric` for the metric it refers to.
    """

    rules: dict[str, GateRule] = Field(min_length=1)
    min_cases: int = Field(default=DEFAULT_MIN_CASES, ge=0)
    """Fewer cases than this adds a note to the report (never a failure); 0 disables it."""
    on_missing: OnMissing = "fail"
    """Default for gates whose metric has no data (see :class:`GateRule`)."""

    @model_validator(mode="after")
    def _known_metrics(self) -> Gates:
        seen: dict[str, str] = {}
        for name, rule in self.rules.items():
            metric = resolve_metric(name)
            if metric in seen:
                raise ValueError(f"{name!r} is the same metric as {seen[metric]!r}")
            seen[metric] = name
            for _, message in _rule_problems(metric, rule):
                raise ValueError(message)
        return self

    @classmethod
    def from_dict(cls, data: Any) -> Gates:
        """Build gates from the YAML shape (``{metric: {rule: value}, options: {...}}``).

        Raises:
            GatesError: listing every problem, each with its location.
        """
        gates, problems = _parse(data)
        if gates is None:
            raise GatesError("invalid gates", [SuiteIssue(msg, loc) for loc, msg, _ in problems])
        return gates

    def evaluate(
        self,
        comparison: Comparison | None = None,
        run: ExperimentRun | Iterable[RunResult] | None = None,
    ) -> GateReport:
        """Shortcut for :func:`evaluate_gates`."""
        return evaluate_gates(self, comparison=comparison, run=run)


class GatesError(SuiteError):
    """A gates file could not be loaded; every problem is in :attr:`issues`.

    A :class:`~agentlens.suite.SuiteError`, so tools can report suite and gates
    configuration errors the same way (``file:line:column: path: message``).
    """


# -- loading ---------------------------------------------------------------------------

_Problem = tuple[Loc, str, bool]
"""``(location, message, about the mapping key)``."""


class _Options(SchemaModel):
    min_cases: int = Field(default=DEFAULT_MIN_CASES, ge=0)
    on_missing: OnMissing = "fail"


def _model_problems(exc: ValidationError, prefix: Loc, fields: Iterable[str]) -> list[_Problem]:
    problems: list[_Problem] = []
    for err in exc.errors(include_url=False):
        loc: Loc = (*prefix, *err["loc"])
        if err["type"] == "extra_forbidden":
            field = str(err["loc"][-1])
            close = difflib.get_close_matches(field, list(fields), n=1)
            hint = f" (did you mean {close[0]!r}?)" if close else ""
            what = "option" if prefix == (_OPTIONS,) else "rule"
            problems.append((loc, f"unknown {what} {field!r}{hint}", True))
        else:
            problems.append((loc, str(err["msg"]).removeprefix("Value error, "), False))
    return problems


def _parse(data: Any) -> tuple[Gates | None, list[_Problem]]:
    if not isinstance(data, Mapping):
        message = (
            "a gates file must be a mapping of metric to rule, e.g. 'success_rate: {min: 0.9}'"
        )
        return None, [((), message, False)]
    problems: list[_Problem] = []
    options = _Options()
    raw_options = data.get(_OPTIONS)
    if raw_options is not None:
        if not isinstance(raw_options, Mapping):
            problems.append(((_OPTIONS,), "options must be a mapping, e.g. {min_cases: 20}", False))
        else:
            try:
                options = _Options.model_validate(dict(raw_options))
            except ValidationError as exc:
                problems += _model_problems(exc, (_OPTIONS,), _Options.model_fields)
    rules: dict[str, GateRule] = {}
    seen: dict[str, str] = {}
    for key, value in data.items():
        name = str(key)
        if name == _OPTIONS:
            continue
        try:
            metric = resolve_metric(name)
        except ValueError as exc:
            problems.append(((name,), str(exc), True))
            continue
        if metric in seen:
            problems.append(((name,), f"{name!r} is the same metric as {seen[metric]!r}", True))
            continue
        seen[metric] = name
        if not isinstance(value, Mapping):
            message = f"a gate is a mapping of rule to threshold, e.g. {name}: {{min: 0.9}}"
            problems.append(((name,), message, False))
            continue
        try:
            rule = GateRule.model_validate(dict(value))
        except ValidationError as exc:
            problems += _model_problems(exc, (name,), _RULE_FIELDS)
            continue
        bad = _rule_problems(metric, rule)
        problems += [((name, field), message, True) for field, message in bad]
        rules[name] = rule
    if not rules and not problems:
        message = "no gates defined: add e.g. 'success_rate: {min: 0.9}'"
        problems.append(((), message, False))
    if problems:
        return None, problems
    return Gates(rules=rules, min_cases=options.min_cases, on_missing=options.on_missing), []


def load_gates(path: str | os.PathLike[str]) -> Gates:
    """Load a gates YAML file.

    Raises:
        GatesError: if the file is missing, not valid YAML or not valid gates,
            listing every problem at its ``file:line:column``.
    """
    gates_path = Path(path)
    try:
        data, smap = load_yaml(gates_path, label="gates file")
    except SuiteError as exc:
        raise GatesError(exc.summary, exc.issues) from None
    gates, problems = _parse(data)
    if gates is None:
        raise GatesError(f"invalid gates file {gates_path}", _positioned(problems, smap))
    return gates


def _positioned(problems: list[_Problem], smap: SourceMap) -> list[SuiteIssue]:
    issues = collect(smap.issue(loc, message, key=key) for loc, message, key in problems)
    return sorted(issues, key=lambda i: (i.line or 0, i.column or 0))


# -- evaluation --------------------------------------------------------------------------


class GateResult(SchemaModel):
    """The verdict of one bound of one gate (``success_rate: {min: 0.9}``)."""

    name: str
    """The gate as written in the file (``success_rate``)."""
    metric: str
    """The metric it refers to (``pass_rate``)."""
    rule: str
    """``min``, ``max``, ``min_delta``, ``max_delta``, ``min_delta_pct`` or ``max_delta_pct``."""
    threshold: float
    actual: float | None = None
    """The value checked (candidate value, delta, percent change or count); ``None`` if unknown."""
    passed: bool
    """``True`` for skipped gates too: only a failed gate fails the report."""
    skipped: bool = False
    """The gate could not be checked and was not counted as a failure."""
    reason: str
    """One line for humans, e.g. ``pass_rate 0.85 is below min 0.9``."""

    def __str__(self) -> str:
        status = "SKIP" if self.skipped else "PASS" if self.passed else "FAIL"
        return f"{status} {self.name} {self.rule} {_fmt(self.rule, self.threshold)}: {self.reason}"


class GateReport(SchemaModel):
    """Every gate's verdict; :attr:`exit_code` is what a CI step should exit with."""

    passed: bool
    results: list[GateResult] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    """Things worth knowing that never fail the report (e.g. too few cases)."""

    @property
    def exit_code(self) -> int:
        """:data:`EXIT_OK` (0) if every gate passed, else :data:`EXIT_GATE_FAILED` (1)."""
        return EXIT_OK if self.passed else EXIT_GATE_FAILED

    @property
    def failures(self) -> list[GateResult]:
        """The gates that failed."""
        return [r for r in self.results if not r.passed]

    @property
    def skipped(self) -> list[GateResult]:
        """The gates that could not be checked and were skipped."""
        return [r for r in self.results if r.skipped]

    def summary(self) -> str:
        """Plain-text verdict, one line per gate, then the notes."""
        n_skipped, n_failed = len(self.skipped), len(self.failures)
        n_passed = len(self.results) - n_skipped - n_failed
        head = (
            f"gates {'passed' if self.passed else 'FAILED'}: "
            f"{n_passed} passed, {n_failed} failed, {n_skipped} skipped"
        )
        lines = [head, *(f"  {r}" for r in self.results)]
        lines += [f"  note: {note}" for note in self.notes]
        return "\n".join(lines)


def _fmt(rule: str, value: float) -> str:
    if rule.endswith("_pct"):
        return f"{value:+g}%"
    if rule in _DELTA:
        return f"{value:+.4g}"
    return f"{value:.4g}"


def _within(rule: str, actual: float, threshold: float) -> bool:
    if math.isclose(actual, threshold, rel_tol=1e-9, abs_tol=1e-12):
        return True  # inclusive bounds, robust to float rounding (27/30 vs 0.9)
    return actual >= threshold if rule.startswith("min") else actual <= threshold


class _Unavailable(Exception):
    def __init__(self, reason: str, *, applicable: bool = True) -> None:
        super().__init__(reason)
        self.reason = reason
        self.applicable = applicable


def _absolute(
    metric: str, comparison: Comparison | None, values: Mapping[str, tuple[float, int]]
) -> float:
    if metric in COUNT_GATES:
        if comparison is None:
            raise _Unavailable("needs a baseline comparison", applicable=False)
        found = comparison.regressions if metric == REGRESSIONS else comparison.case_regressions
        return float(len(found))
    if metric not in values:
        raise _Unavailable(f"no data for {metric!r}{_available(values)}")
    return values[metric][0]


def _available(values: Iterable[str]) -> str:
    scores = sorted(name for name in values if name.startswith(SCORE_PREFIX))
    return f" (scores available: {', '.join(scores)})" if scores else ""


def _delta(metric: str, rule: str, comparison: Comparison | None) -> float:
    if comparison is None:
        raise _Unavailable("needs a baseline comparison", applicable=False)
    try:
        delta = comparison.metric(metric)
    except KeyError:
        names = (m.name for m in comparison.metrics)
        raise _Unavailable(f"no metric {metric!r} in the comparison{_available(names)}") from None
    if delta.status is Status.NO_DATA or delta.delta is None:
        raise _Unavailable(f"no case has a {metric!r} value in both experiments")
    if not rule.endswith("_pct"):
        return delta.delta
    if delta.delta_pct is None:
        if delta.delta == 0:
            return 0.0
        raise _Unavailable(f"baseline {metric!r} is 0: percentage change undefined")
    return delta.delta_pct


def _check(
    name: str,
    metric: str,
    rule: str,
    threshold: float,
    actual: float,
) -> GateResult:
    passed = _within(rule, actual, threshold)
    shown = f"{metric}{' change' if rule in _DELTA else ''} {_fmt(rule, actual)}"
    limit = f"{rule} {_fmt(rule, threshold)}"
    if passed:
        reason = f"{shown} is within {limit}"
    else:
        side = "below" if rule.startswith("min") else "above"
        reason = f"{shown} is {side} {limit}"
    return GateResult(
        name=name,
        metric=metric,
        rule=rule,
        threshold=threshold,
        actual=actual,
        passed=passed,
        reason=reason,
    )


def evaluate_gates(
    gates: Gates,
    comparison: Comparison | None = None,
    run: ExperimentRun | Iterable[RunResult] | None = None,
) -> GateReport:
    """Check every gate and return the verdict.

    Args:
        gates: The gates (see :func:`load_gates`).
        comparison: Candidate vs baseline. Needed by delta rules and by the
            ``regressions``/``case_regressions`` gates; without it those are
            *skipped* (not failed) with a note, so a first run with no
            baseline yet can still be gated on absolute values.
        run: The candidate's runs. When given, ``min``/``max`` apply to its
            metrics over all its cases; otherwise to the comparison's
            candidate means over the paired cases.

    A gate whose metric has no data (no costs recorded, an evaluator that did
    not run) fails with the reason, unless its ``on_missing`` is ``skip``.

    Raises:
        ValueError: if neither ``comparison`` nor ``run`` is given.
    """
    if comparison is None and run is None:
        raise ValueError("evaluate_gates needs a comparison, a run, or both")
    values: dict[str, tuple[float, int]]
    if run is not None:
        values = run_metrics(run)
    else:
        assert comparison is not None
        values = {m.name: (m.candidate, m.n) for m in comparison.metrics if m.candidate is not None}

    results: list[GateResult] = []
    not_applicable = 0
    for name, gate in gates.rules.items():
        metric = resolve_metric(name)
        on_missing = gate.on_missing or gates.on_missing
        for rule, threshold in gate.bounds():
            try:
                if rule in _ABSOLUTE:
                    actual = _absolute(metric, comparison, values)
                else:
                    actual = _delta(metric, rule, comparison)
            except _Unavailable as missing:
                skip = not missing.applicable or on_missing == "skip"
                not_applicable += not missing.applicable
                results.append(
                    GateResult(
                        name=name,
                        metric=metric,
                        rule=rule,
                        threshold=threshold,
                        passed=skip,
                        skipped=skip,
                        reason=missing.reason,
                    )
                )
                continue
            results.append(_check(name, metric, rule, threshold, actual))

    notes = _notes(gates, comparison, values, not_applicable)
    return GateReport(passed=all(r.passed for r in results), results=results, notes=notes)


def _notes(
    gates: Gates,
    comparison: Comparison | None,
    values: Mapping[str, tuple[float, int]],
    not_applicable: int,
) -> list[str]:
    notes: list[str] = []
    if comparison is not None:
        n, what = len(comparison.cases), "compared (present in both experiments)"
    else:
        n, what = values.get(PASS_RATE, (0.0, 0))[1], "evaluated"
    if n < gates.min_cases:
        notes.append(
            f"only {n} case{'s' if n != 1 else ''} {what}: at least {gates.min_cases} are "
            "recommended for meaningful results (ADR 0005); not a gate failure"
        )
    if not_applicable:
        notes.append(
            f"no baseline comparison: {not_applicable} delta/regression "
            f"check{'s' if not_applicable != 1 else ''} skipped"
        )
    if comparison is not None:
        for ids, side in (
            (comparison.added_cases, "only in the candidate"),
            (comparison.removed_cases, "only in the baseline"),
        ):
            if ids:
                count = f"{len(ids)} case{'s' if len(ids) != 1 else ''}"
                listed = ", ".join(ids[:5]) + (", ..." if len(ids) > 5 else "")
                notes.append(f"{count} {side}, not part of any delta: {listed}")
    return notes
