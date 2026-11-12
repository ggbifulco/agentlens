"""Tests for EvalResult and Experiment."""

from __future__ import annotations

from datetime import UTC

import pytest
from pydantic import ValidationError

from agentlens.schema import EvalResult, Experiment, id_datetime


@pytest.mark.parametrize("score", [-0.01, 1.01])
def test_eval_result_score_must_be_normalised(score: float) -> None:
    with pytest.raises(ValidationError):
        EvalResult(evaluator="x", score=score, passed=True)


def test_eval_result_defaults() -> None:
    result = EvalResult(evaluator="exact_match", score=1.0, passed=True)
    assert result.reason is None
    assert result.details == {}


def test_experiment_defaults() -> None:
    exp = Experiment(name="pr-123", suite="support")
    assert exp.is_baseline is False
    assert exp.created_at.tzinfo is UTC
    assert exp.config == {}
    # The id embeds (roughly) the creation time.
    assert abs((id_datetime(exp.id) - exp.created_at).total_seconds()) < 1
