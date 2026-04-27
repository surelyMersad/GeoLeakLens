import math

import numpy as np
import pytest

from geoleaklens.scoring.bootstrap import (
    bootstrap_median,
    bootstrap_paired_difference,
)


def test_bootstrap_median_recovers_point_estimate():
    rng = np.random.default_rng(0)
    values = rng.normal(loc=10.0, scale=2.0, size=200)
    res = bootstrap_median(values, n_bootstrap=500, seed=42)
    assert res.point_estimate == pytest.approx(np.median(values))
    # CI should bracket the true population median (10) for a sample this size.
    assert res.ci_low <= 10.0 <= res.ci_high
    # And it should be tight for n=200.
    assert (res.ci_high - res.ci_low) < 1.0


def test_bootstrap_median_empty_input_returns_nan():
    res = bootstrap_median([])
    assert math.isnan(res.point_estimate)
    assert math.isnan(res.se)
    assert res.n_samples == 0


def test_bootstrap_median_se_reflects_dispersion():
    """Higher-variance input should yield a larger bootstrap SE."""
    rng = np.random.default_rng(1)
    low_var = rng.normal(loc=0.0, scale=1.0, size=100)
    high_var = rng.normal(loc=0.0, scale=10.0, size=100)
    r_low = bootstrap_median(low_var, n_bootstrap=300, seed=7)
    r_high = bootstrap_median(high_var, n_bootstrap=300, seed=7)
    assert r_high.se > r_low.se


def test_bootstrap_stratifies_only_when_all_strata_meet_min():
    values = list(range(100))
    # Two equal strata of 50 each — exceeds default min=30 → stratifies.
    strata_ok = [0] * 50 + [1] * 50
    res_ok = bootstrap_median(values, strata=strata_ok, n_bootstrap=200, seed=0)
    assert res_ok.stratified is True

    # Skewed: one stratum has only 10 samples → no stratification.
    strata_skewed = [0] * 90 + [1] * 10
    res_skewed = bootstrap_median(values, strata=strata_skewed, n_bootstrap=200, seed=0)
    assert res_skewed.stratified is False


def test_bootstrap_strata_length_mismatch_raises():
    with pytest.raises(ValueError):
        bootstrap_median([1.0, 2.0, 3.0], strata=[0, 1])


def test_paired_difference_sign_preserved():
    """If method A is consistently higher than B per image, the paired
    median difference should be positive and CI should not include zero."""
    rng = np.random.default_rng(3)
    n = 100
    # Per-image baseline error, plus a fixed +5 km penalty for method A.
    base = rng.exponential(scale=5.0, size=n)
    a = base + 5.0 + rng.normal(scale=0.5, size=n)
    b = base + rng.normal(scale=0.5, size=n)
    res = bootstrap_paired_difference(a, b, n_bootstrap=500, seed=11)
    assert res.point_estimate > 4.0
    assert res.point_estimate < 6.0
    assert res.ci_low > 0.0   # CI excludes zero → significant difference
    assert res.ci_high > res.ci_low


def test_paired_difference_unequal_length_raises():
    with pytest.raises(ValueError):
        bootstrap_paired_difference([1.0, 2.0], [1.0])


def test_paired_difference_zero_when_methods_identical():
    n = 50
    rng = np.random.default_rng(5)
    same = rng.normal(size=n)
    res = bootstrap_paired_difference(same, same, n_bootstrap=200, seed=5)
    assert res.point_estimate == pytest.approx(0.0)
    assert res.ci_low == pytest.approx(0.0)
    assert res.ci_high == pytest.approx(0.0)


# ----- §11.6 mean-statistic variant for 0/1 indicator data ------------------

def test_bootstrap_mean_recovers_proportion():
    """A 0/1 indicator (Acc@25km) needs mean, not median (median collapses
    to 0 or 1)."""
    from geoleaklens.scoring.bootstrap import bootstrap_mean
    values = [0, 0, 0, 1, 1, 1, 1, 0, 1, 1]   # 60% rate
    res = bootstrap_mean(values, n_bootstrap=500, seed=0)
    assert res.point_estimate == pytest.approx(0.6)
    # CI is wide at n=10 but should bracket the true rate.
    assert res.ci_low <= 0.6 <= res.ci_high


def test_bootstrap_paired_difference_with_mean_statistic():
    """Pair where method A beats method B by ~10pp on average."""
    rng = np.random.default_rng(11)
    n = 200
    base = rng.binomial(1, 0.5, size=n)
    # method_a flips 10% of zeros to ones (so it's strictly better-or-equal)
    a = base.copy()
    flip = rng.random(size=n) < 0.20
    a[(base == 0) & flip] = 1
    b = base
    res = bootstrap_paired_difference(
        a, b, statistic="mean", n_bootstrap=500, seed=42
    )
    # Per-image diff is 0 or 1; mean ≈ 0.10. CI should exclude zero.
    assert res.point_estimate > 0.05
    assert res.ci_low > 0.0


def test_bootstrap_statistic_rejects_unknown_statistic():
    from geoleaklens.scoring.bootstrap import bootstrap_statistic
    with pytest.raises(ValueError, match="unknown statistic"):
        bootstrap_statistic([1.0, 2.0, 3.0], statistic="trimmed_mean")  # type: ignore[arg-type]
