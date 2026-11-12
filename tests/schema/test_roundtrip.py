"""Property tests: model -> JSON -> model is the identity for every schema type."""

from __future__ import annotations

import pytest
from hypothesis import given
from hypothesis import strategies as st
from pydantic import BaseModel

from agentlens.schema import Case, EvalResult, EvaluatorSpec, Experiment, Trace, Usage
from tests.schema import strategies as s

MODELS: list[tuple[type[BaseModel], st.SearchStrategy[BaseModel]]] = [
    (Trace, s.traces()),
    (Usage, s.usages),
    (Case, s.cases),
    (EvaluatorSpec, s.evaluator_specs),
    (EvalResult, s.eval_results),
    (Experiment, s.experiments),
]


@pytest.mark.parametrize(("model", "strategy"), MODELS, ids=lambda m: getattr(m, "__name__", ""))
def test_json_round_trip_is_identity(
    model: type[BaseModel], strategy: st.SearchStrategy[BaseModel]
) -> None:
    @given(strategy)
    def check(obj: BaseModel) -> None:
        restored = model.model_validate_json(obj.model_dump_json())
        assert restored == obj

    check()


@given(s.traces())
def test_python_dict_round_trip_is_identity(trace: Trace) -> None:
    assert Trace.model_validate(trace.model_dump(mode="json")) == trace


@given(s.traces())
def test_derived_metrics_survive_round_trip(trace: Trace) -> None:
    restored = Trace.model_validate_json(trace.model_dump_json())
    assert restored.total_tokens == trace.total_tokens
    assert restored.total_cost == trace.total_cost
    assert restored.tool_calls == trace.tool_calls
    assert restored.n_steps == trace.n_steps
    assert restored.latency_ms == trace.latency_ms
