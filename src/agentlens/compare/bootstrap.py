"""Paired bootstrap confidence intervals (ADR 0005), standard library only.

Given one difference per case (``candidate - baseline``), the bootstrap
estimates how much the *mean* difference could move if we had drawn a
different set of cases: resample the cases with replacement many times,
take the mean of each resample, and read the confidence interval off the
percentiles of those means. Because each case contributes its own paired
difference, case-to-case variation (some cases are just harder) cancels out,
which makes the test far more sensitive than comparing two unpaired means.

::

    from agentlens.compare import paired_bootstrap

    low, high = paired_bootstrap([-0.5, 0.0, -1.0, -0.5], seed=0)
    # high < 0: the candidate is worse on these cases beyond resampling noise
"""

from __future__ import annotations

import math
import random
from collections.abc import Sequence

__all__ = ["DEFAULT_CONFIDENCE", "DEFAULT_RESAMPLES", "paired_bootstrap", "percentile"]

DEFAULT_RESAMPLES = 10_000
"""Bootstrap resamples used by default (ADR 0005)."""

DEFAULT_CONFIDENCE = 0.95
"""Default confidence level of the interval."""


def percentile(sorted_values: Sequence[float], q: float) -> float:
    """The ``q``-quantile (``0 <= q <= 1``) of ``sorted_values``, linearly interpolated.

    Same definition as NumPy's default (``method="linear"``): position
    ``q * (n - 1)`` between the two nearest order statistics.

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
    if fraction == 0.0 or low_value == high_value:
        return low_value
    return low_value + (high_value - low_value) * fraction


def paired_bootstrap(
    diffs: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    confidence: float = DEFAULT_CONFIDENCE,
    seed: int | None = 0,
) -> tuple[float, float]:
    """Percentile bootstrap confidence interval of the mean of paired differences.

    Draws ``resamples`` resamples of ``len(diffs)`` differences with
    replacement (``random.Random(seed)``), takes each resample's mean and
    returns the ``(1 - confidence) / 2`` and ``(1 + confidence) / 2``
    percentiles of those means.

    Which cases are drawn depends only on ``seed``, ``len(diffs)`` and
    ``resamples`` — never on the values — so the same seed gives the same
    interval, and adding a constant to every difference shifts the interval
    by that constant. The interval is widened, if needed, to contain the
    observed mean (this only matters for floating-point rounding, e.g. when
    every difference is the same).

    Args:
        diffs: One difference per case (``candidate - baseline``).
        resamples: Number of bootstrap resamples (>= 1).
        confidence: Confidence level, strictly between 0 and 1.
        seed: Seed of the random generator; ``None`` for a non-reproducible run.

    Returns:
        ``(low, high)``, with ``low <= mean(diffs) <= high``.

    Raises:
        ValueError: if ``diffs`` is empty, ``resamples < 1`` or ``confidence``
            is not in ``(0, 1)``.
    """
    if not diffs:
        raise ValueError("paired_bootstrap needs at least one difference")
    if resamples < 1:
        raise ValueError(f"resamples must be >= 1, got {resamples}")
    if not 0.0 < confidence < 1.0:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")
    values = [float(d) for d in diffs]
    n = len(values)
    mean = math.fsum(values) / n
    if all(v == values[0] for v in values):
        return mean, mean  # every resample has the same mean: nothing to estimate
    rng = random.Random(seed)
    choices = rng.choices
    means = sorted(sum(choices(values, k=n)) / n for _ in range(resamples))
    alpha = (1.0 - confidence) / 2.0
    low = percentile(means, alpha)
    high = percentile(means, 1.0 - alpha)
    return min(low, mean), max(high, mean)
