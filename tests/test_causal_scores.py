import math

import pandas as pd
import pytest

from geoleaklens.scoring.causal_scores import (
    DEFAULT_INTERVENTION_SET,
    assign_ranks_within_image,
    min_across_interventions,
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


# ----- §10.7 min_across_interventions ----------------------------------------

def _attribution_row(*, image_id, region_id, intervention_type,
                     continuous_attribution, binary_attribution=None,
                     area_frac=0.05, edited_error_km=None,
                     utility_cost=None, method="static_greedy"):
    """Compact factory for fake §7.5 rows in the tests below."""
    if binary_attribution is None:
        binary_attribution = 1.0 if continuous_attribution > 0 else 0.0
    if edited_error_km is None:
        # Pretend the haversine-clip error correlates with the attribution.
        edited_error_km = max(0.1, continuous_attribution * 10)
    return {
        "image_id": image_id,
        "region_id": region_id,
        "model_name": "geoclip",
        "threshold_km": 25.0,
        "original_error_km": 1.0,
        "edited_error_km": edited_error_km,
        "original_success": True,
        "edited_success": edited_error_km <= 25.0,
        "binary_attribution": binary_attribution,
        "continuous_attribution": continuous_attribution,
        "binary_joint_causal": None,
        "continuous_joint_causal": None,
        "method": method,
        "intervention_type": intervention_type,
        "area_frac": area_frac,
        "utility_cost": utility_cost,
        "score_raw": continuous_attribution,
        "score_per_area": continuous_attribution / max(area_frac, 0.005),
        "score_per_utility": None,
        "rank_within_image": -1,
    }


def test_min_across_picks_smallest_continuous_attribution():
    """The conservative reading: a region only counts as leaky if the
    *least* eager intervention agrees."""
    rows = [
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="mean_mask",
                         continuous_attribution=1.5),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="blur",
                         continuous_attribution=0.5),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="inpaint",
                         continuous_attribution=0.1),
    ]
    out = min_across_interventions(pd.DataFrame(rows))
    assert len(out) == 1
    row = out.iloc[0]
    assert row["intervention_type"] == "min_across"
    assert row["continuous_attribution"] == pytest.approx(0.1)
    assert row["region_id"] == "r1"
    # score_per_area is recomputed from the min, not averaged from the rows.
    assert row["score_per_area"] == pytest.approx(0.1 / 0.05)


def test_min_across_skips_groups_with_partial_coverage():
    """If only 2 of 3 interventions are present, drop the region —
    silent undercounting is worse than a missing row."""
    rows = [
        _attribution_row(image_id="img", region_id="full",
                         intervention_type="mean_mask",
                         continuous_attribution=0.3),
        _attribution_row(image_id="img", region_id="full",
                         intervention_type="blur",
                         continuous_attribution=0.4),
        _attribution_row(image_id="img", region_id="full",
                         intervention_type="inpaint",
                         continuous_attribution=0.2),
        _attribution_row(image_id="img", region_id="partial",
                         intervention_type="mean_mask",
                         continuous_attribution=0.9),
        _attribution_row(image_id="img", region_id="partial",
                         intervention_type="blur",
                         continuous_attribution=0.7),
        # missing: inpaint for region "partial"
    ]
    out = min_across_interventions(pd.DataFrame(rows))
    region_ids = set(out["region_id"])
    assert "full" in region_ids
    assert "partial" not in region_ids


def test_min_across_aggregates_per_region():
    """Multiple regions and multiple images each get their own min."""
    rows = []
    for img in ("img1", "img2"):
        for region, scores in [("r1", (0.5, 0.3, 0.1)),
                               ("r2", (0.05, 0.2, 0.4))]:
            for intervention, cont in zip(("mean_mask", "blur", "inpaint"), scores):
                rows.append(_attribution_row(
                    image_id=img, region_id=region,
                    intervention_type=intervention,
                    continuous_attribution=cont,
                ))
    out = min_across_interventions(pd.DataFrame(rows))
    assert len(out) == 4
    by_key = {(r["image_id"], r["region_id"]): r["continuous_attribution"]
              for _, r in out.iterrows()}
    assert by_key[("img1", "r1")] == pytest.approx(0.1)
    assert by_key[("img1", "r2")] == pytest.approx(0.05)
    assert by_key[("img2", "r1")] == pytest.approx(0.1)
    assert by_key[("img2", "r2")] == pytest.approx(0.05)


def test_min_across_picks_negative_when_one_intervention_helps():
    """If one intervention actually pulls the prediction *closer* (negative
    continuous_attribution), the conservative min reflects that."""
    rows = [
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="mean_mask",
                         continuous_attribution=1.0),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="blur",
                         continuous_attribution=0.4),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="inpaint",
                         continuous_attribution=-0.05),
    ]
    out = min_across_interventions(pd.DataFrame(rows))
    assert out.iloc[0]["continuous_attribution"] == pytest.approx(-0.05)


def test_min_across_binary_attribution_is_min():
    """Binary min: a region only counts as flipping success if EVERY
    intervention flipped it."""
    rows = [
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="mean_mask",
                         continuous_attribution=1.0, binary_attribution=1.0),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="blur",
                         continuous_attribution=0.5, binary_attribution=1.0),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="inpaint",
                         continuous_attribution=0.1, binary_attribution=0.0),
    ]
    out = min_across_interventions(pd.DataFrame(rows))
    assert out.iloc[0]["binary_attribution"] == 0.0


def test_min_across_carries_edit_fields_from_argmin_row():
    """The edited_error_km / edited_success on the min_across row come from
    the most-conservative intervention, not aggregated across all three."""
    rows = [
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="mean_mask",
                         continuous_attribution=1.5, edited_error_km=100.0),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="blur",
                         continuous_attribution=0.5, edited_error_km=10.0),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="inpaint",
                         continuous_attribution=0.1, edited_error_km=2.0),
    ]
    out = min_across_interventions(pd.DataFrame(rows))
    row = out.iloc[0]
    # The min Δ came from inpaint; that row's edited_error_km should ride along.
    assert row["edited_error_km"] == pytest.approx(2.0)


def test_min_across_empty_input_returns_empty():
    out = min_across_interventions(pd.DataFrame())
    assert out.empty


def test_min_across_no_matching_interventions_returns_empty():
    """If the input has only `replace` rows but we ask for the default set,
    the output is empty (no overlap)."""
    rows = [
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="replace",
                         continuous_attribution=0.5),
    ]
    out = min_across_interventions(pd.DataFrame(rows))
    assert out.empty


def test_min_across_custom_intervention_set():
    """Caller can pin to a 2-intervention set (e.g., when inpaint is
    too expensive). Only groups covering both contribute."""
    rows = [
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="mean_mask",
                         continuous_attribution=0.7),
        _attribution_row(image_id="img", region_id="r1",
                         intervention_type="blur",
                         continuous_attribution=0.3),
    ]
    out = min_across_interventions(
        pd.DataFrame(rows), intervention_types=("mean_mask", "blur")
    )
    assert len(out) == 1
    assert out.iloc[0]["continuous_attribution"] == pytest.approx(0.3)


def test_default_intervention_set_constant():
    """The default set is the §10.7 trio, in the documented order."""
    assert DEFAULT_INTERVENTION_SET == ("mean_mask", "blur", "inpaint")
