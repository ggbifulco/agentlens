from __future__ import annotations

import math
import random

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from agentlens.compare import DEFAULT_RESAMPLES, paired_bootstrap, percentile

FAST = 200  # resamples in property tests: the properties hold for any count

diffs_st = st.lists(
    st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    min_size=1,
    max_size=40,
)
seeds = st.integers(min_value=0, max_value=2**32)


def mean(values: list[float]) -> float:
    return math.fsum(values) / len(values)


# -- percentile ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("values", "q", "expected"),
    [
        ([5.0], 0.3, 5.0),
        ([1.0, 2.0, 3.0, 4.0], 0.0, 1.0),
        ([1.0, 2.0, 3.0, 4.0], 1.0, 4.0),
        ([1.0, 2.0, 3.0, 4.0], 0.5, 2.5),  # numpy.percentile(..., 50) == 2.5
        ([0.0, 10.0], 0.25, 2.5),
        ([1.0, 1.0, 3.0], 0.25, 1.0),
    ],
)
def test_percentile_interpolates_like_numpy(values: list[float], q: float, expected: float) -> None:
    assert percentile(values, q) == pytest.approx(expected)


@pytest.mark.parametrize(("values", "q"), [([], 0.5), ([1.0], -0.1), ([1.0], 1.5)])
def test_percentile_rejects_bad_input(values: list[float], q: float) -> None:
    with pytest.raises(ValueError, match=r"empty|q must"):
        percentile(values, q)


# -- paired_bootstrap: examples -----------------------------------------------------


def test_clear_negative_shift_has_ci_below_zero() -> None:
    diffs = [-1.0, -0.5, -1.0, -0.8, -1.2, -0.9, -1.1, -0.7] * 3
    low, high = paired_bootstrap(diffs)
    assert low <= mean(diffs) <= high < 0


def test_noise_around_zero_has_ci_containing_zero() -> None:
    rng = random.Random(7)
    diffs = [rng.gauss(0.0, 1.0) for _ in range(50)]
    low, high = paired_bootstrap(diffs)
    assert low < 0 < high


def test_default_resamples_is_ten_thousand() -> None:
    assert DEFAULT_RESAMPLES == 10_000


def test_higher_confidence_gives_wider_interval() -> None:
    diffs = [0.1, -0.3, 0.5, 0.2, -0.1, 0.4, 0.0, 0.3]
    low90, high90 = paired_bootstrap(diffs, confidence=0.90)
    low99, high99 = paired_bootstrap(diffs, confidence=0.99)
    assert low99 <= low90
    assert high90 <= high99


def test_seed_none_still_returns_a_valid_interval() -> None:
    diffs = [0.0, 1.0, 2.0, 3.0]
    low, high = paired_bootstrap(diffs, seed=None, resamples=500)
    assert 0.0 <= low <= 1.5 <= high <= 3.0


def test_single_difference_is_a_point_interval() -> None:
    assert paired_bootstrap([0.25]) == (0.25, 0.25)


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"diffs": []}, "at least one"),
        ({"diffs": [1.0], "resamples": 0}, "resamples"),
        ({"diffs": [1.0], "confidence": 1.0}, "confidence"),
        ({"diffs": [1.0], "confidence": 0.0}, "confidence"),
    ],
)
def test_rejects_bad_arguments(kwargs: dict[str, object], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        paired_bootstrap(**kwargs)  # type: ignore[arg-type]


# -- paired_bootstrap: properties ---------------------------------------------------


@given(diffs=diffs_st, seed=seeds)
@settings(max_examples=150, deadline=None)
def test_ci_contains_observed_mean(diffs: list[float], seed: int) -> None:
    low, high = paired_bootstrap(diffs, resamples=FAST, seed=seed)
    assert low <= mean(diffs) <= high


@given(n=st.integers(min_value=1, max_value=50), seed=seeds)
@settings(max_examples=50, deadline=None)
def test_zero_diffs_give_zero_interval(n: int, seed: int) -> None:
    assert paired_bootstrap([0.0] * n, resamples=FAST, seed=seed) == (0.0, 0.0)


@given(diffs=diffs_st, shift=st.floats(min_value=-1e3, max_value=1e3), seed=seeds)
@settings(max_examples=150, deadline=None)
def test_shifting_every_diff_shifts_the_interval(
    diffs: list[float], shift: float, seed: int
) -> None:
    low, high = paired_bootstrap(diffs, resamples=FAST, seed=seed)
    low2, high2 = paired_bootstrap([d + shift for d in diffs], resamples=FAST, seed=seed)
    assert low2 == pytest.approx(low + shift, abs=1e-6)
    assert high2 == pytest.approx(high + shift, abs=1e-6)


@given(diffs=diffs_st, seed=seeds)
@settings(max_examples=100, deadline=None)
def test_same_seed_same_interval(diffs: list[float], seed: int) -> None:
    first = paired_bootstrap(diffs, resamples=FAST, seed=seed)
    assert paired_bootstrap(diffs, resamples=FAST, seed=seed) == first


@given(diffs=diffs_st, seed=seeds)
@settings(max_examples=100, deadline=None)
def test_interval_within_data_range(diffs: list[float], seed: int) -> None:
    low, high = paired_bootstrap(diffs, resamples=FAST, seed=seed)
    assert min(diffs) - 1e-9 <= low <= high <= max(diffs) + 1e-9
