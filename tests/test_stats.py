"""Tests for the statistics module: Wilson intervals and Cohen's kappa."""

from __future__ import annotations

import pytest

from judgecheck.stats import cohens_kappa, wilson_interval


def test_wilson_interval_bounds_inside_unit_interval() -> None:
    lower, upper = wilson_interval(8, 10)
    assert 0.0 <= lower <= upper <= 1.0


def test_wilson_interval_narrows_as_trials_grow() -> None:
    small_lower, small_upper = wilson_interval(50, 100)
    large_lower, large_upper = wilson_interval(500, 1000)
    assert (large_upper - large_lower) < (small_upper - small_lower)


def test_wilson_interval_zero_trials_is_full_width() -> None:
    assert wilson_interval(0, 0) == (0.0, 1.0)


def test_cohens_kappa_perfect_agreement_is_one() -> None:
    assert cohens_kappa(["yes", "no", "yes"], ["yes", "no", "yes"]) == 1.0


def test_cohens_kappa_chance_agreement_is_zero() -> None:
    # Observed agreement (2 of 4) equals chance agreement (0.5).
    assert cohens_kappa([0, 0, 1, 1], [0, 1, 0, 1]) == pytest.approx(0.0)


def test_cohens_kappa_mismatched_lengths_raises() -> None:
    with pytest.raises(ValueError):
        cohens_kappa(["yes", "no"], ["yes"])
