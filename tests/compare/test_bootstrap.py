"""Unit and property tests of the bootstrap (ADR 0005)."""

from __future__ import annotations

import math
import random
import statistics

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlens.compare import ConfidenceInterval, bootstrap_mean_ci, percentile

# Bounded, finite values: what per-case metric differences look like.
values_st = st.lists(
    st.floats(min_value=-1e6, max_value=1e6, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=40,
)
FAST = {"resamples": 300}


# -- percentile ----------------------------------------------------------------


def test_percentile_interpolates_like_numpy() -> None:
    data = [1.0, 2.0, 3.0, 4.0]
    assert percentile(data, 0.0) == 1.0
    assert percentile(data, 1.0) == 4.0
    assert percentile(data, 0.5) == pytest.approx(2.5)
    assert percentile(data, 0.25) == pytest.approx(1.75)
    assert percentile([7.0], 0.3) == 7.0


@pytest.mark.parametrize("q", [-0.1, 1.1])
def test_percentile_rejects_bad_q(q: float) -> None:
    with pytest.raises(ValueError, match="q must be"):
        percentile([1.0], q)


def test_percentile_rejects_empty() -> None:
    with pytest.raises(ValueError, match="empty"):
        percentile([], 0.5)


# -- bootstrap_mean_ci: unit tests ----------------------------------------------


def test_constant_values_give_zero_width_interval() -> None:
    ci = bootstrap_mean_ci([0.25] * 10)
    assert ci == ConfidenceInterval(0.25, 0.25, 0.95)
    assert ci.width == 0.0


def test_single_value() -> None:
    ci = bootstrap_mean_ci([-3.0])
    assert (ci.low, ci.high) == (-3.0, -3.0)


def test_defaults_and_known_case() -> None:
    data = [float(i) for i in range(20)]
    ci = bootstrap_mean_ci(data)
    assert ci.confidence == 0.95
    # Mean 9.5, sd of the mean ~1.32 -> roughly [6.9, 12.1].
    assert 6.5 < ci.low < 7.5
    assert 11.5 < ci.high < 12.5
    assert 9.5 in ci


def test_contains_and_excludes_zero() -> None:
    ci = ConfidenceInterval(-0.2, 0.1)
    assert 0 in ci
    assert 0.1 in ci
    assert 0.5 not in ci
    assert "x" not in ci
    assert not ci.excludes_zero()
    assert ConfidenceInterval(0.1, 0.3).excludes_zero()
    assert ConfidenceInterval(-0.3, -0.1).excludes_zero()
    assert not ConfidenceInterval(0.1, 0.3).excludes_zero(tolerance=0.2)


@pytest.mark.parametrize(
    ("kwargs", "match"),
    [
        ({"resamples": 0}, "resamples"),
        ({"confidence": 0.0}, "confidence"),
        ({"confidence": 1.0}, "confidence"),
    ],
)
def test_rejects_bad_arguments(kwargs: dict[str, float], match: str) -> None:
    with pytest.raises(ValueError, match=match):
        bootstrap_mean_ci([1.0, 2.0], **kwargs)  # type: ignore[arg-type]


def test_rejects_empty_and_non_finite() -> None:
    with pytest.raises(ValueError, match="empty"):
        bootstrap_mean_ci([])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_ci([1.0, math.nan])
    with pytest.raises(ValueError, match="finite"):
        bootstrap_mean_ci([1.0, math.inf])


def test_wider_confidence_gives_wider_interval() -> None:
    rng = random.Random(1)
    data = [rng.gauss(0, 1) for _ in range(30)]
    narrow = bootstrap_mean_ci(data, confidence=0.8)
    wide = bootstrap_mean_ci(data, confidence=0.99)
    assert wide.low <= narrow.low
    assert wide.high >= narrow.high
    assert wide.width > narrow.width


def test_different_seeds_differ_slightly() -> None:
    rng = random.Random(2)
    data = [rng.gauss(0, 1) for _ in range(30)]
    a = bootstrap_mean_ci(data, seed=1)
    b = bootstrap_mean_ci(data, seed=2)
    assert a != b
    assert a.low == pytest.approx(b.low, abs=0.05)


def test_coverage_is_close_to_nominal() -> None:
    """Statistical check: the 95% interval covers the true mean ~95% of the time."""
    rng = random.Random(42)
    trials, covered = 300, 0
    for trial in range(trials):
        sample = [rng.gauss(0.3, 1.0) for _ in range(40)]
        ci = bootstrap_mean_ci(sample, resamples=1000, seed=trial)
        covered += 0.3 in ci
    # Percentile bootstrap undercovers a little at n=40; allow a generous band.
    assert 0.88 <= covered / trials <= 0.99


def test_detects_a_real_shift_and_ignores_noise() -> None:
    rng = random.Random(7)
    shifted = [rng.gauss(-0.2, 0.1) for _ in range(30)]
    assert bootstrap_mean_ci(shifted).excludes_zero()
    noise = [rng.gauss(0.0, 0.1) for _ in range(30)]
    noise = [x - statistics.fmean(noise) for x in noise]  # exactly zero mean
    assert not bootstrap_mean_ci(noise).excludes_zero()


# -- bootstrap_mean_ci: properties ------------------------------------------------


@settings(max_examples=60, deadline=None)
@given(values_st, st.integers(min_value=0, max_value=2**32))
def test_property_reproducible_with_seed(values: list[float], seed: int) -> None:
    assert bootstrap_mean_ci(values, seed=seed, **FAST) == bootstrap_mean_ci(
        values, seed=seed, **FAST
    )


@settings(max_examples=60, deadline=None)
@given(values_st)
def test_property_bounds_within_data_range(values: list[float]) -> None:
    ci = bootstrap_mean_ci(values, **FAST)
    assert min(values) <= ci.low <= ci.high <= max(values)


@settings(max_examples=60, deadline=None)
@given(values_st)
def test_property_interval_contains_the_sample_mean(values: list[float]) -> None:
    """The bootstrap distribution is centred on the sample mean."""
    ci = bootstrap_mean_ci(values, resamples=2000)
    mean = statistics.fmean(values)
    slack = 1e-9 * max(1.0, *(abs(v) for v in values))
    assert ci.low - slack <= mean <= ci.high + slack


@settings(max_examples=60, deadline=None)
@given(values_st, st.floats(min_value=-1e3, max_value=1e3, allow_nan=False))
def test_property_shift_equivariant(values: list[float], shift: float) -> None:
    base = bootstrap_mean_ci(values, **FAST)
    moved = bootstrap_mean_ci([v + shift for v in values], **FAST)
    scale = max(1.0, *(abs(v) for v in values), abs(shift))
    assert moved.low == pytest.approx(base.low + shift, abs=1e-9 * scale)
    assert moved.high == pytest.approx(base.high + shift, abs=1e-9 * scale)


@settings(max_examples=60, deadline=None)
@given(values_st)
def test_property_sign_flip_mirrors_interval(values: list[float]) -> None:
    base = bootstrap_mean_ci(values, **FAST)
    flipped = bootstrap_mean_ci([-v for v in values], **FAST)
    scale = max(1.0, *(abs(v) for v in values))
    assert flipped.low == pytest.approx(-base.high, abs=1e-9 * scale)
    assert flipped.high == pytest.approx(-base.low, abs=1e-9 * scale)


@settings(max_examples=60, deadline=None)
@given(values_st, st.floats(min_value=0.01, max_value=100.0))
def test_property_scale_equivariant(values: list[float], factor: float) -> None:
    """Resampled positions do not depend on the values: scaling scales the interval."""
    base = bootstrap_mean_ci(values, **FAST)
    scaled = bootstrap_mean_ci([v * factor for v in values], **FAST)
    scale = max(1.0, *(abs(v) * factor for v in values))
    assert scaled.low == pytest.approx(base.low * factor, abs=1e-9 * scale)
    assert scaled.high == pytest.approx(base.high * factor, abs=1e-9 * scale)
