"""Statistics for judge scores.

Pure functions: no LLM calls and no file I/O. Wilson score intervals
quantify uncertainty in a rate (for example, the share of traces a
judge marked relevant); Cohen's kappa measures agreement between a
judge's labels and human labels.
"""

from __future__ import annotations

import math
from collections import Counter
from statistics import NormalDist
from typing import Hashable, Sequence


def wilson_interval(
    successes: int, trials: int, confidence: float = 0.95
) -> tuple[float, float]:
    """Return the (lower, upper) bounds of the Wilson score interval.

    ``successes`` out of ``trials`` observations at the given
    ``confidence`` level (between 0 and 1, exclusive). With zero trials
    there is no information, so the interval is the full [0, 1] range.
    """
    if trials < 0:
        raise ValueError("trials must not be negative")
    if successes < 0 or successes > trials:
        raise ValueError("successes must be between 0 and trials")
    if not 0 < confidence < 1:
        raise ValueError("confidence must be between 0 and 1")

    if trials == 0:
        return (0.0, 1.0)

    z = NormalDist().inv_cdf(1 - (1 - confidence) / 2)
    p_hat = successes / trials
    denominator = 1 + z * z / trials
    centre = p_hat + z * z / (2 * trials)
    spread = z * math.sqrt(p_hat * (1 - p_hat) / trials + z * z / (4 * trials * trials))
    lower = (centre - spread) / denominator
    upper = (centre + spread) / denominator
    return (max(0.0, lower), min(1.0, upper))


def cohens_kappa(
    labels_a: Sequence[Hashable], labels_b: Sequence[Hashable]
) -> float:
    """Return Cohen's kappa for two lists of labels.

    1.0 is perfect agreement, 0.0 is agreement equal to chance, and
    negative values mean agreement worse than chance. The two lists
    must have the same length and must not be empty.
    """
    if len(labels_a) != len(labels_b):
        raise ValueError("label lists must have the same length")
    if not labels_a:
        raise ValueError("label lists must not be empty")

    n = len(labels_a)
    observed = sum(1 for a, b in zip(labels_a, labels_b) if a == b) / n

    counts_a = Counter(labels_a)
    counts_b = Counter(labels_b)
    expected = sum(
        (counts_a[label] / n) * (counts_b[label] / n)
        for label in set(counts_a) | set(counts_b)
    )

    if expected == 1.0:
        return 1.0 if observed == 1.0 else 0.0
    return (observed - expected) / (1 - expected)
