"""LLM-as-judge evaluators: ``llm_rubric`` (score 1-5) and ``llm_pass_fail`` (FR-E2).

Both show the judge the case input, the agent's output, the case's expected
output when there is one (as a *reference*, not an exact answer) and the
``criteria``, and ask for a JSON object with ``reasoning`` first and then the
verdict. The call, parsing, retry and caching live in
:class:`~agentlens.evals.judge.LLMJudge`::

    from agentlens.evals import get_evaluator
    from agentlens.llm import MockLLM

    judge = get_evaluator("llm_rubric", criteria="Polite and confirms the refund amount",
                          model="openai:gpt-4o-mini")
    # tests: get_evaluator("llm_rubric", criteria="...", client=MockLLM([...]), cache=False)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TypeVar

from pydantic import JsonValue

from agentlens.evals.builtin._common import check_bool, params_repr
from agentlens.evals.errors import EvaluatorConfigError
from agentlens.evals.judge import (
    CacheSpec,
    JudgePrompt,
    JudgeRun,
    LLMJudge,
    check_template,
    load_template,
    parse_reasoning,
    parse_score,
    parse_verdict,
    render_template,
    value_text,
)
from agentlens.llm import LLMClient
from agentlens.schema import Case, EvalResult, Trace

__all__ = ["CRITERIA_PLACEHOLDERS", "LLMPassFail", "LLMRubric", "PassFailVerdict", "RubricVerdict"]

CRITERIA_PLACEHOLDERS = ("criteria", "input", "output", "reference")
"""Placeholders a custom ``template`` of ``llm_rubric``/``llm_pass_fail`` may use."""

RUBRIC_LOW = 1
RUBRIC_HIGH = 5

_V = TypeVar("_V")


def _check_criteria(criteria: object) -> str:
    if not isinstance(criteria, str) or not criteria.strip():
        raise EvaluatorConfigError(
            f"criteria must be a non-empty string describing what to judge, got {criteria!r}"
        )
    return criteria.strip()


class _CriteriaJudge(LLMJudge[_V]):
    """Shared prompt rendering of the criteria-based judges."""

    template_name: str = ""

    def __init__(
        self,
        criteria: str,
        *,
        include_expected: bool = True,
        template: str | None = None,
        model: str | None = None,
        client: LLMClient | None = None,
        cache: CacheSpec = None,
        retries: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            model=model, client=client, cache=cache, retries=retries, max_tokens=max_tokens
        )
        self.criteria = _check_criteria(criteria)
        self.include_expected = check_bool("include_expected", include_expected)
        self.template = (
            None if template is None else check_template(template, CRITERIA_PLACEHOLDERS)
        )

    def build_prompt(self, trace: Trace, case: Case) -> JudgePrompt:
        """Render the system prompt and the criteria template for this run."""
        agent_input = trace.input if trace.input is not None else case.input
        output = value_text(trace.output)
        reference = ""
        if self.include_expected and case.expected is not None:
            reference = render_template(
                load_template("reference"), {"expected": value_text(case.expected)}
            )
        user = render_template(
            self.template if self.template is not None else load_template(self.template_name),
            {
                "criteria": self.criteria,
                "input": value_text(agent_input),
                "output": output,
                "reference": reference,
            },
        )
        return JudgePrompt(system=load_template("judge_system"), user=user, output=output)

    def _params(self) -> dict[str, object]:
        return {
            "criteria": self.criteria,
            "include_expected": self.include_expected,
            "template": "custom" if self.template is not None else None,
            "model": self.model,
            "cache": self.cache,
            "retries": self.retries,
        }


@dataclass(frozen=True)
class RubricVerdict:
    """A parsed rubric answer: the 1-5 score and the judge's explanation."""

    score: float
    reasoning: str | None


@dataclass(frozen=True)
class PassFailVerdict:
    """A parsed pass/fail answer."""

    passed: bool
    reasoning: str | None


def _with_reasoning(head: str, reasoning: str | None) -> str:
    return f"{head}: {reasoning}" if reasoning else head


class LLMRubric(_CriteriaJudge[RubricVerdict]):
    """``llm_rubric``: an LLM judge scores the output from 1 to 5 against ``criteria``.

    The 1-5 score is normalised to ``[0, 1]`` as ``(score - 1) / 4`` (so 4 → 0.75)
    and the result passes when ``score >= min_score``. The prompt anchors each
    point of the scale (``prompts/rubric.txt``). A fractional score (``3.5``) is
    kept; a score outside 1-5 is a parse error (retried, then a failed result).

    Args:
        criteria: What a good output does (``{llm_rubric: "Polite, confirms the amount"}``).
        min_score: Lowest passing 1-5 score (default 4: "meets the criteria with
            minor issues").
        include_expected: Show ``case.expected`` to the judge as a reference.
        template: Custom prompt text with ``$criteria``, ``$input``, ``$output``,
            ``$reference`` placeholders (it must still ask for ``score``/``reasoning`` JSON).

    ``model``, ``client``, ``cache``, ``retries`` and ``max_tokens`` are as in
    :class:`~agentlens.evals.judge.LLMJudge`.
    """

    name: str = "llm_rubric"
    template_name = "rubric"

    def __init__(
        self,
        criteria: str,
        *,
        min_score: float = 4,
        include_expected: bool = True,
        template: str | None = None,
        model: str | None = None,
        client: LLMClient | None = None,
        cache: CacheSpec = None,
        retries: int = 1,
        max_tokens: int | None = None,
    ) -> None:
        super().__init__(
            criteria,
            include_expected=include_expected,
            template=template,
            model=model,
            client=client,
            cache=cache,
            retries=retries,
            max_tokens=max_tokens,
        )
        if (
            isinstance(min_score, bool)
            or not isinstance(min_score, int | float)
            or not math.isfinite(min_score)
            or not RUBRIC_LOW <= min_score <= RUBRIC_HIGH
        ):
            raise EvaluatorConfigError(
                f"min_score must be a number from {RUBRIC_LOW} to {RUBRIC_HIGH}, got {min_score!r}"
            )
        self.min_score = float(min_score)

    def parse_reply(self, data: dict[str, JsonValue]) -> RubricVerdict:
        """Read ``score`` (1-5, numeric strings accepted) and ``reasoning``."""
        return RubricVerdict(
            score=parse_score(data, low=RUBRIC_LOW, high=RUBRIC_HIGH),
            reasoning=parse_reasoning(data),
        )

    def make_result(self, verdict: RubricVerdict, run: JudgeRun[RubricVerdict]) -> EvalResult:
        """Normalise the score and compare it with ``min_score``."""
        passed = verdict.score >= self.min_score
        head = f"judge score {verdict.score:g}/5 ({'>=' if passed else '<'} {self.min_score:g})"
        details = run.details()
        details.update(
            {
                "rubric_score": verdict.score,
                "min_score": self.min_score,
                "reasoning": verdict.reasoning,
                "criteria": self.criteria,
            }
        )
        return EvalResult(
            evaluator=self.name,
            score=(verdict.score - RUBRIC_LOW) / (RUBRIC_HIGH - RUBRIC_LOW),
            passed=passed,
            reason=_with_reasoning(head, verdict.reasoning),
            details=details,
        )

    def __repr__(self) -> str:
        return params_repr("LLMRubric", {**self._params(), "min_score": self.min_score})


class LLMPassFail(_CriteriaJudge[PassFailVerdict]):
    """``llm_pass_fail``: an LLM judge decides whether the output meets *all* ``criteria``.

    The reply's ``verdict`` (``"pass"``/``"fail"``; yes/no, true/false and booleans
    are tolerated) gives score 1 or 0; ``reasoning`` becomes the result's reason.
    The prompt (``prompts/pass_fail.txt``) tells the judge to fail when unsure.

    Args:
        criteria: The conditions the output must meet.
        include_expected: Show ``case.expected`` to the judge as a reference.
        template: Custom prompt text (same placeholders as :class:`LLMRubric`; it
            must ask for ``verdict``/``reasoning`` JSON).

    ``model``, ``client``, ``cache``, ``retries`` and ``max_tokens`` are as in
    :class:`~agentlens.evals.judge.LLMJudge`.
    """

    name: str = "llm_pass_fail"
    template_name = "pass_fail"

    def parse_reply(self, data: dict[str, JsonValue]) -> PassFailVerdict:
        """Read ``verdict`` and ``reasoning``."""
        return PassFailVerdict(passed=parse_verdict(data), reasoning=parse_reasoning(data))

    def make_result(self, verdict: PassFailVerdict, run: JudgeRun[PassFailVerdict]) -> EvalResult:
        """Score 1 on pass, 0 on fail."""
        word = "pass" if verdict.passed else "fail"
        details = run.details()
        details.update({"verdict": word, "reasoning": verdict.reasoning, "criteria": self.criteria})
        return EvalResult(
            evaluator=self.name,
            score=1.0 if verdict.passed else 0.0,
            passed=verdict.passed,
            reason=_with_reasoning(f"judge verdict: {word}", verdict.reasoning),
            details=details,
        )

    def __repr__(self) -> str:
        return params_repr("LLMPassFail", self._params())
