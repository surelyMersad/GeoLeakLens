import math

import pytest

from geoleaklens.scoring.geolocation_metrics import (
    DEFAULT_THRESHOLDS_KM,
    aggregate_errors,
    per_image_errors,
)


def test_aggregate_errors_basic_counts_and_median():
    errs = [0.5, 10.0, 100.0, 1000.0, float("inf")]
    parses = [True, True, True, True, False]
    out = aggregate_errors(errs, parses)
    assert out["n"] == 5
    assert out["n_parsed"] == 4
    assert out["parse_success_rate"] == pytest.approx(0.8)
    assert out["median_error_km"] == pytest.approx(100.0)
    # Acc@25km: only the first two (0.5, 10) succeed → 2/5.
    assert out["acc_25km"] == pytest.approx(2 / 5)
    # Acc@2500km: 0.5, 10, 100, 1000 succeed → 4/5.
    assert out["acc_2500km"] == pytest.approx(4 / 5)


def test_aggregate_errors_treats_inf_and_none_as_failures():
    errs = [None, float("inf"), float("nan"), 5.0]
    parses = [True, True, True, True]
    out = aggregate_errors(errs, parses, thresholds_km=[25.0])
    # Only 5.0 is within 25km; the rest are clipped to MAX_ERROR_KM and fail.
    assert out["acc_25km"] == pytest.approx(0.25)
    # Median of clipped values: [5, MAX, MAX, MAX] → MAX at index 2.
    assert out["median_error_km"] > 1000


def test_aggregate_errors_failed_parse_disqualifies_threshold_success():
    """A successful parse with very small error counts at threshold; a parse
    failure with the same numeric error should not."""
    errs = [1.0, 1.0]
    out_both_parsed = aggregate_errors(errs, [True, True], thresholds_km=[25.0])
    assert out_both_parsed["acc_25km"] == pytest.approx(1.0)
    out_one_failed = aggregate_errors(errs, [True, False], thresholds_km=[25.0])
    assert out_one_failed["acc_25km"] == pytest.approx(0.5)


def test_aggregate_errors_default_thresholds_present():
    errs = [10.0]
    parses = [True]
    out = aggregate_errors(errs, parses)
    for tau in DEFAULT_THRESHOLDS_KM:
        assert f"acc_{int(tau)}km" in out


def test_aggregate_errors_empty_input():
    out = aggregate_errors([], [])
    assert out == {"n": 0}


def test_aggregate_errors_mismatched_lengths_raises():
    with pytest.raises(ValueError):
        aggregate_errors([1.0, 2.0], [True])


def test_per_image_errors_zero_for_exact_match():
    errs = per_image_errors([10.0], [20.0], [10.0], [20.0])
    assert errs == [pytest.approx(0.0, abs=1e-6)]


def test_per_image_errors_inf_for_missing_prediction():
    errs = per_image_errors([None, 10.0], [None, 20.0], [10.0, 10.0], [20.0, 20.0])
    assert math.isinf(errs[0])
    assert errs[1] == pytest.approx(0.0, abs=1e-6)
