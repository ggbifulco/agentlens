"""Paired bootstrap confidence intervals (ADR 0005), pure standard library.

The comparator reduces a metric to one *paired* difference per case
(``candidate - baseline``) and asks: how sure are we that the mean difference is
not zero? The bootstrap answers without assuming a distribution: resample the
cases with replacement many times, take the mean of each resample, and read the
interval from the percentiles of those means (the *percentile* bootstrap).

Resampling *cases* (not runs) keeps each case's baseline and candidate together,
which removes the case-to-case difficulty variance from the comparison: that is
what makes it *paired*.

Determinism: the resampled positions depend only on ``seed``, ``resamples`` and
the number of values, never on the values themselves. Two metrics compared over
the same cases with the same seed are therefore resampled with exactly the same
cases, and a report re-computed tomorrow shows the same interval.
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from dataclasses import dataclass

__all__ = [
    "DEFAULT_CONFIDENCE",
    "DEFAULT_RESAMPLES",
    "DEFAULT_SEED",
    "ConfidenceInterval",
    "bootstrap_mean_ci",
    "percentile",
]

DEFAULT_RESAMPLES = 10_000
"""Resamples per interval (ADR 0005): the percentile error is well below a
rounding step of the reported numbers."""
DEFAULT_CONFIDENCE = 0.95
DEFAULT_SEED = 0


@dataclass(frozen=True)
class ConfidenceInterval:
    """A two-sided interval ``[low, high]`` around the mean at ``confidence``."""

    low: float
    high: float
    confidence: float = DEFAULT_CONFIDENCE

    def __contains__(self, value: object) -> bool:
        return isinstance(value, int | float) and self.low <= value <= self.high

    @property
    def width(self) -> float:
        """``high - low``."""
        return self.high - self.low

    def excludes_zero(self, tolerance: float = 0.0) -> bool:
        """Whether the whole interval is on one side of zero (beyond ``tolerance``)."""
        return self.low > tolerance or self.high < -tolerance


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-quantile (``0 <= q <= 1``) of already sorted values.

    Linear interpolation between the closest ranks (the default of NumPy's
    ``percentile``), so ``q=0`` is the minimum and ``q=1`` the maximum.

    Raises:
        ValueError: if ``sorted_values`` is empty or ``q`` is outside ``[0, 1]``.
    """
    if not sorted_values:
        raise ValueError("percentile of an empty sequence")
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"q must be in [0, 1], got {q}")
    position = q * (len(sorted_values) - 1)
    lower = math.floor(position)
    upper = min(lower + 1, len(sorted_values) - 1)
    fraction = position - lower
    low_value, high_value = sorted_values[lower], sorted_values[upper]
    return low_value + (high_value - low_value) * fraction


def bootstrap_mean_ci(
    values: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int = DEFAULT_SEED,
) -> ConfidenceInterval:
    """Percentile-bootstrap confidence interval of the mean of ``values``.

    For a paired comparison pass the per-case differences ``candidate -
    baseline``; the interval is then the interval of the mean difference.

    The bounds always lie within ``[min(values), max(values)]``; identical
    values give the zero-width interval ``[v, v]``.

    Args:
        values: One number per case.
        resamples: How many bootstrap resamples to draw.
        confidence: Coverage of the two-sided interval, in ``(0, 1)``.
        seed: Seed of the private random generator (reproducible results).

    Raises:
        ValueError: on empty or non-finite ``values``, ``resamples < 1`` or
            ``confidence`` outside ``(0, 1)``.
    """
    if not values:
        raise ValueError("cannot bootstrap an empty sequence")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    data = [float(v) for v in values]
    if not all(math.isfinite(v) for v in data):
        raise ValueError("values must be finite numbers")
    smallest, largest = min(data), max(data)
    if smallest == largest:
        return ConfidenceInterval(smallest, largest, confidence)

    rng = random.Random(seed)
    n = len(data)
    # ``choices`` draws positions from random() only, so the resampled cases do
    # not depend on the values (see the module docstring).
    means = sorted(sum(rng.choices(data, k=n)) / n for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0

    def clamp(x: float) -> float:
        # Floating-point summation can step a hair outside the data range.
        return min(max(x, smallest), largest)

    return ConfidenceInterval(
        clamp(percentile(means, alpha)), clamp(percentile(means, 1.0 - alpha)), confidence
    )
