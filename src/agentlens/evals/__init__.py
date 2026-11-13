r"""Evaluators: score traces with deterministic checks or LLM judges (``docs/design.md`` §5).

An evaluator is anything with a ``name`` and ``evaluate(trace, case) -> EvalResult``
(:class:`Evaluator`). The quickest way to write one is the decorator::

    import agentlens

    @agentlens.evaluator
    def mentions_order(trace, case) -> bool:
        return "order" in str(trace.output)

Suites refer to evaluators by name through the registry, using the YAML
spellings of :class:`~agentlens.schema.EvaluatorSpec`::

    from agentlens.evals import evaluate_case, get_evaluator

    check = get_evaluator("mentions_order")
    results = evaluate_case(trace, case, ["mentions_order"])

Built-in evaluators (:mod:`agentlens.evals.builtin`: ``exact_match``,
``contains``, ``regex``, ``json_schema``, ``no_error``, ``tool_sequence``,
``max_steps``, ``cost_budget``, ``latency_budget``) register themselves in
:data:`default_registry` when this package is imported::

    results = evaluate_case(trace, case, ["no_error", {"contains": "refund"},
                                          {"regex": {"pattern": r"order \d+"}},
                                          {"tool_sequence": {"mode": "ordered"}},
                                          {"max_steps": 8}])

An evaluator that raises never crashes a run: :func:`run_evaluator` and
``@evaluator`` turn the exception into a failed :class:`~agentlens.schema.EvalResult`.
Configuration errors (unknown name, bad parameters) are raised eagerly.
"""

from agentlens.evals.base import (
    DEFAULT_THRESHOLD,
    AsyncEvaluator,
    Evaluator,
    error_result,
    to_eval_result,
)
from agentlens.evals.builtin import (
    UNSET,
    Contains,
    CostBudget,
    ExactMatch,
    JsonSchema,
    LatencyBudget,
    MaxSteps,
    NoError,
    OutputPath,
    OutputPathError,
    Regex,
    ToolSequence,
    output_text,
    output_value,
    register_builtins,
    to_text,
)
from agentlens.evals.decorator import FunctionEvaluator, evaluator
from agentlens.evals.errors import (
    DuplicateEvaluatorError,
    EvaluatorConfigError,
    EvaluatorError,
    UnknownEvaluatorError,
)
from agentlens.evals.registry import (
    EvaluatorFactory,
    EvaluatorRegistry,
    RegisteredEvaluator,
    SpecLike,
    available_evaluators,
    default_registry,
    get_evaluator,
    register,
    unregister,
)
from agentlens.evals.run import (
    aevaluate_case,
    arun_evaluator,
    evaluate_case,
    resolve_evaluators,
    run_evaluator,
)

__all__ = [
    "DEFAULT_THRESHOLD",
    "UNSET",
    "AsyncEvaluator",
    "Contains",
    "CostBudget",
    "DuplicateEvaluatorError",
    "Evaluator",
    "EvaluatorConfigError",
    "EvaluatorError",
    "EvaluatorFactory",
    "EvaluatorRegistry",
    "ExactMatch",
    "FunctionEvaluator",
    "JsonSchema",
    "LatencyBudget",
    "MaxSteps",
    "NoError",
    "OutputPath",
    "OutputPathError",
    "Regex",
    "RegisteredEvaluator",
    "SpecLike",
    "ToolSequence",
    "UnknownEvaluatorError",
    "aevaluate_case",
    "arun_evaluator",
    "available_evaluators",
    "default_registry",
    "error_result",
    "evaluate_case",
    "evaluator",
    "get_evaluator",
    "output_text",
    "output_value",
    "register",
    "register_builtins",
    "resolve_evaluators",
    "run_evaluator",
    "to_eval_result",
    "to_text",
    "unregister",
]
