import pandas as pd

from geoleaklens.redaction.methods import StaticGreedy, selection_summary


def _make_data(image_id: str = "img_a"):
    """Three regions of varying score density and area."""
    regions = pd.DataFrame(
        [
            {"image_id": image_id, "region_id": "r_high_small", "area_frac": 0.02},
            {"image_id": image_id, "region_id": "r_mid_med", "area_frac": 0.05},
            {"image_id": image_id, "region_id": "r_low_big", "area_frac": 0.30},
            {"image_id": image_id, "region_id": "r_tiny", "area_frac": 0.005},
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "image_id": image_id,
                "region_id": "r_high_small",
                "score_per_area": 50.0,
                "intervention_type": "mean_mask",
            },
            {
                "image_id": image_id,
                "region_id": "r_mid_med",
                "score_per_area": 5.0,
                "intervention_type": "mean_mask",
            },
            {
                "image_id": image_id,
                "region_id": "r_low_big",
                "score_per_area": 1.0,
                "intervention_type": "mean_mask",
            },
            {
                "image_id": image_id,
                "region_id": "r_tiny",
                "score_per_area": 100.0,
                "intervention_type": "mean_mask",
            },
        ]
    )
    return regions, scores


def test_picks_highest_score_per_area_first():
    regions, scores = _make_data()
    method = StaticGreedy()
    selected = method.select_regions(
        "img_a", regions, scores, budget={"area_frac": 0.10}
    )
    # r_tiny (score=100) and r_high_small (score=50) both fit; r_mid_med
    # (score=5, area 0.05) also fits within 0.10 once tiny+high are taken.
    # Total area: 0.005 + 0.02 + 0.05 = 0.075 < 0.10. r_low_big (0.30) skipped.
    assert selected == ["r_tiny", "r_high_small", "r_mid_med"]


def test_skips_oversized_regions():
    regions, scores = _make_data()
    method = StaticGreedy()
    selected = method.select_regions(
        "img_a", regions, scores, budget={"area_frac": 0.025}
    )
    # Only r_tiny (0.005) and r_high_small (0.02) fit at all.
    assert selected == ["r_tiny", "r_high_small"]


def test_zero_budget_returns_empty():
    regions, scores = _make_data()
    method = StaticGreedy()
    assert method.select_regions("img_a", regions, scores, {"area_frac": 0.0}) == []
    assert method.select_regions("img_a", regions, scores, {"area_frac": -0.1}) == []


def test_filters_to_image_id_and_intervention_type():
    regions = pd.DataFrame(
        [
            {"image_id": "img_a", "region_id": "r1", "area_frac": 0.05},
            {"image_id": "img_b", "region_id": "r2", "area_frac": 0.05},
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "image_id": "img_a",
                "region_id": "r1",
                "score_per_area": 1.0,
                "intervention_type": "mean_mask",
            },
            {
                "image_id": "img_a",
                "region_id": "r1",
                "score_per_area": 99.0,
                "intervention_type": "blur",
            },
            {
                "image_id": "img_b",
                "region_id": "r2",
                "score_per_area": 50.0,
                "intervention_type": "mean_mask",
            },
        ]
    )
    method = StaticGreedy(intervention_type="mean_mask")
    assert method.select_regions("img_a", regions, scores, {"area_frac": 0.10}) == ["r1"]
    assert method.select_regions("img_b", regions, scores, {"area_frac": 0.10}) == ["r2"]


def test_no_regions_for_image_returns_empty():
    regions, scores = _make_data()
    method = StaticGreedy()
    assert (
        method.select_regions("img_missing", regions, scores, {"area_frac": 0.10})
        == []
    )


def test_stops_at_99pct_of_target():
    """Once cumulative >= 99% of target, no further regions are added."""
    regions = pd.DataFrame(
        [
            {"image_id": "img", "region_id": f"r{i}", "area_frac": 0.05}
            for i in range(10)
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "image_id": "img",
                "region_id": f"r{i}",
                "score_per_area": 100.0 - i,
                "intervention_type": "mean_mask",
            }
            for i in range(10)
        ]
    )
    method = StaticGreedy()
    selected = method.select_regions(
        "img", regions, scores, budget={"area_frac": 0.10}
    )
    # 0.05 + 0.05 = 0.10 = 99%+ of 0.10 → stop after 2.
    assert selected == ["r0", "r1"]


def test_selection_summary_tracks_actual_vs_target_area():
    regions, _ = _make_data()
    summary = selection_summary(
        selected_ids=["r_tiny", "r_high_small"],
        regions=regions,
        image_id="img_a",
        target_area_frac=0.10,
    )
    assert summary["n_selected"] == 2
    assert summary["selected_region_ids"] == ["r_tiny", "r_high_small"]
    assert abs(summary["actual_area_frac"] - 0.025) < 1e-9
    assert abs(summary["area_gap_pp"] - (-7.5)) < 1e-9


def test_tie_break_prefers_smaller_region():
    """Two regions with the same score_per_area: pick the smaller one first."""
    regions = pd.DataFrame(
        [
            {"image_id": "img", "region_id": "big", "area_frac": 0.08},
            {"image_id": "img", "region_id": "small", "area_frac": 0.02},
        ]
    )
    scores = pd.DataFrame(
        [
            {
                "image_id": "img",
                "region_id": "big",
                "score_per_area": 10.0,
                "intervention_type": "mean_mask",
            },
            {
                "image_id": "img",
                "region_id": "small",
                "score_per_area": 10.0,
                "intervention_type": "mean_mask",
            },
        ]
    )
    method = StaticGreedy()
    selected = method.select_regions(
        "img", regions, scores, budget={"area_frac": 0.05}
    )
    # Budget 0.05 only fits 'small' (0.02); 'big' (0.08) too big.
    assert selected == ["small"]
