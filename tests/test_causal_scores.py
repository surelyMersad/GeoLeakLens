import math

import pytest

from geoleaklens.scoring.causal_scores import (
    assign_ranks_within_image,
    single_region_attribution,
)

# A reference triple: ground truth in San Francisco, original prediction near
# SF (within 25km), edited prediction in NYC.
TRUE = (37.7749, -122.4194)
ORIG_NEAR = (37.78, -122.40)
EDIT_FAR = (40.7128, -74.0060)


def _row(**overrides):
    base = dict(
        image_id="img_1",
        region_id="img_1__sam_000",
        model_name="geoclip",
        intervention_type="mean_mask",
        threshold_km=25.0,
        true_lat=TRUE[0],
        true_lon=TRUE[1],
        orig_lat=ORIG_NEAR[0],
        orig_lon=ORIG_NEAR[1],
        edit_lat=EDIT_FAR[0],
        edit_lon=EDIT_FAR[1],
        area_frac=0.05,
    )
    base.update(overrides)
    return single_region_attribution(**base)


def test_binary_attribution_when_edit_breaks_success():
    r = _row()
    assert r["original_success"] is True
    assert r["edited_success"] is False
    assert r["binary_attribution"] == 1.0
    assert r["continuous_attribution"] > 0
    assert r["score_per_area"] > 0


def test_continuous_attribution_log_diff_matches_spec():
    r = _row()
    expected = math.log1p(r["edited_error_km"]) - math.log1p(r["original_error_km"])
    assert r["continuous_attribution"] == pytest.approx(expected)


def test_no_change_gives_zero_attribution():
    r = _row(edit_lat=ORIG_NEAR[0], edit_lon=ORIG_NEAR[1])
    assert r["binary_attribution"] == 0.0
    assert r["continuous_attribution"] == pytest.approx(0.0, abs=1e-9)
    assert r["score_per_area"] == pytest.approx(0.0, abs=1e-9)


def test_negative_attribution_when_edit_helps():
    # Edit lands closer than the original — attribution should be negative.
    r = _row(orig_lat=EDIT_FAR[0], orig_lon=EDIT_FAR[1],
             edit_lat=ORIG_NEAR[0], edit_lon=ORIG_NEAR[1])
    assert r["binary_attribution"] == -1.0
    assert r["continuous_attribution"] < 0


def test_missing_edit_prediction_clips_to_max():
    r = _row(edit_lat=None, edit_lon=None)
    assert r["edited_success"] is False
    # edited_error_km clips to MAX_ERROR_KM.
    assert r["edited_error_km"] > 20000
    assert r["binary_attribution"] == 1.0


def test_score_per_area_floors_tiny_regions():
    # A 1px-equivalent region (area_frac well below floor) shouldn't blow up;
    # divisor is floored at 0.005.
    r = _row(area_frac=0.0001)
    expected = r["continuous_attribution"] / 0.005
    assert r["score_per_area"] == pytest.approx(expected)


def test_utility_cost_drives_score_per_utility_when_provided():
    r = _row(utility_cost=0.2)
    expected = r["continuous_attribution"] / 0.2
    assert r["score_per_utility"] == pytest.approx(expected)


def test_score_per_utility_is_none_without_utility_cost():
    r = _row()
    assert r["score_per_utility"] is None


def test_joint_causal_fields_are_none_for_single_region_method():
    r = _row()
    assert r["binary_joint_causal"] is None
    assert r["continuous_joint_causal"] is None


def test_assign_ranks_orders_within_image_by_score_per_area():
    rows = [
        _row(region_id="r0", area_frac=0.05),                     # high attribution, normal area
        _row(region_id="r1", edit_lat=ORIG_NEAR[0],               # zero attribution
             edit_lon=ORIG_NEAR[1], area_frac=0.05),
        _row(region_id="r2", area_frac=0.5),                      # high attribution but big area
    ]
    out = assign_ranks_within_image(rows)
    by_id = {r["region_id"]: r for r in out}
    # r0 (small area, big effect) > r2 (big area, big effect) > r1 (no effect).
    assert by_id["r0"]["rank_within_image"] == 0
    assert by_id["r2"]["rank_within_image"] == 1
    assert by_id["r1"]["rank_within_image"] == 2
