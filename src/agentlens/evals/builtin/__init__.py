"""Built-in evaluators, registered in :data:`~agentlens.evals.default_registry` on import.

``agentlens.evals`` imports this package, so ``get_evaluator("contains", text="x")``
and suite specs like ``{contains: refund}`` work without setup. Registration uses
each class as its factory; the *primary* parameter (what a scalar shorthand
``{name: x}`` sets) is listed below.

* ``exact_match`` (:class:`ExactMatch`) — ``value``, the expected output;
* ``contains`` (:class:`Contains`) — ``text``, a substring or list of substrings;
* ``regex`` (:class:`Regex`) — ``pattern``;
* ``json_schema`` (:class:`JsonSchema`) — ``schema``;
* ``no_error`` (:class:`NoError`) — ``include_spans``;
* ``tool_sequence`` (:class:`ToolSequence`) — ``tools``, the expected tool names;
* ``max_steps`` (:class:`MaxSteps`) — ``value``, the step limit;
* ``cost_budget`` (:class:`CostBudget`) — ``max_usd``;
* ``latency_budget`` (:class:`LatencyBudget`) — ``max_ms``;
* ``llm_rubric`` (:class:`LLMRubric`) — ``criteria`` (LLM judge, score 1-5);
* ``llm_pass_fail`` (:class:`LLMPassFail`) — ``criteria`` (LLM judge, pass/fail).

The judges resolve their LLM client lazily (``model`` param, else
``$AGENTLENS_JUDGE_MODEL``/``$AGENTLENS_MODEL``), so registering them needs no setup.
"""

from __future__ import annotations

from agentlens.evals.builtin._common import (
    OutputPath,
    OutputPathError,
    output_text,
    output_value,
    to_text,
)
from agentlens.evals.builtin.judges import LLMPassFail, LLMRubric
from agentlens.evals.builtin.output import UNSET, Contains, ExactMatch, JsonSchema, Regex
from agentlens.evals.builtin.trace_checks import (
    CostBudget,
    LatencyBudget,
    MaxSteps,
    NoError,
    ToolSequence,
)
from agentlens.evals.registry import EvaluatorFactory, EvaluatorRegistry, default_registry

__all__ = [
    "BUILTIN_EVALUATORS",
    "UNSET",
    "Contains",
    "CostBudget",
    "ExactMatch",
    "JsonSchema",
    "LLMPassFail",
    "LLMRubric",
    "LatencyBudget",
    "MaxSteps",
    "NoError",
    "OutputPath",
    "OutputPathError",
    "Regex",
    "ToolSequence",
    "output_text",
    "output_value",
    "register_builtins",
    "to_text",
]

BUILTIN_EVALUATORS: dict[str, tuple[EvaluatorFactory, str]] = {
    "exact_match": (ExactMatch, "value"),
    "contains": (Contains, "text"),
    "regex": (Regex, "pattern"),
    "json_schema": (JsonSchema, "schema"),
    "no_error": (NoError, "include_spans"),
    "tool_sequence": (ToolSequence, "tools"),
    "max_steps": (MaxSteps, "value"),
    "cost_budget": (CostBudget, "max_usd"),
    "latency_budget": (LatencyBudget, "max_ms"),
    "llm_rubric": (LLMRubric, "criteria"),
    "llm_pass_fail": (LLMPassFail, "criteria"),
}
"""Name -> (factory, primary parameter) of every built-in evaluator."""


def register_builtins(registry: EvaluatorRegistry | None = None) -> None:
    """Register the built-ins in ``registry`` (default: :data:`~agentlens.evals.default_registry`).

    Called on import; call it yourself to populate a fresh :class:`EvaluatorRegistry`.
    Existing entries with the same names are replaced.
    """
    target = registry if registry is not None else default_registry
    for name, (factory, primary) in BUILTIN_EVALUATORS.items():
        target.register(name, factory, primary=primary, replace=True)


register_builtins()
