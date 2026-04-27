"""§12.11 Shapley spot-check unit tests with mocked scorer."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from geoleaklens.redaction.shapley import (
    ShapleyImageResult,
    shapley_spot_check_image,
    static_vs_shapley_agreement,
)


def _img(h=4, w=4) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")


def _mask(h: int, w: int, idx: int) -> np.ndarray:
    """Disjoint single-pixel masks for region indexing."""
    m = np.zeros((h, w), dtype=bool)
    m[idx // w, idx % w] = True
    return m


# Per-test queue used by mocks
_mock_state: dict = {"queued_masks": []}


def _apply_intervention(image: Image.Image, mask: np.ndarray) -> Image.Image:
    _mock_state["queued_masks"].append(mask.copy())
    return image


def _make_score_image(error_for_mask):
    def score(images):
        masks = _mock_state["queued_masks"]
        assert len(masks) == len(images), f"{len(masks)} masks, {len(images)} imgs"
        out = []
        for m in masks:
            err = error_for_mask(m)
            # Synthetic lat/lon: 1° lat ≈ 111 km, true location is origin.
            out.append((err / 111.0, 0.0))
        _mock_state["queued_masks"] = []
        return out
    return score


def _error_for_pred(lat, lon):
    if lat is None or lon is None:
        return 1e9
    return float(lat) * 111.0  # invert the lat/error encoding


@pytest.fixture(autouse=True)
def _reset():
    _mock_state["queued_masks"] = []
    yield
    _mock_state["queued_masks"] = []


def test_shapley_picks_only_leaky_region_when_others_are_inert():
    """Only region 'a' is leaky. Shapley(a) should be ~50; others ~0."""
    masks = {f"r{i}": _mask(4, 4, i) for i in range(3)}
    region_ids = list(masks.keys())

    def err_for_mask(m):
        a_in = bool(m[0, 0])  # region r0 is leaky
        return 50.0 if a_in else 1.0

    res = shapley_spot_check_image(
        image_id="img",
        region_ids=region_ids,
        region_masks=masks,
        original_image=_img(),
        original_error_km=1.0,
        apply_intervention=_apply_intervention,
        score_image=_make_score_image(err_for_mask),
        error_for_pred=_error_for_pred,
        n_orderings=50,
        seed=0,
    )
    # Shapley(r0) ≈ 49 (leaky); Shapley(r1), Shapley(r2) ≈ 0.
    s = res.shapley_values
    assert s["r0"] > 40
    assert abs(s["r1"]) < 5
    assert abs(s["r2"]) < 5


def test_shapley_distributes_credit_fairly_for_two_jointly_leaky_regions():
    """Regions a and b together cause error = 100; alone they cause 1.
    Shapley should split the joint effect ~50/50 between them."""
    masks = {"a": _mask(4, 4, 0), "b": _mask(4, 4, 1)}
    region_ids = ["a", "b"]

    def err_for_mask(m):
        a_in = bool(m[0, 0]); b_in = bool(m[0, 1])
        if a_in and b_in: return 100.0
        return 1.0

    res = shapley_spot_check_image(
        image_id="img",
        region_ids=region_ids,
        region_masks=masks,
        original_image=_img(),
        original_error_km=1.0,
        apply_intervention=_apply_intervention,
        score_image=_make_score_image(err_for_mask),
        error_for_pred=_error_for_pred,
        n_orderings=100,
        seed=0,
    )
    # Sum of Shapley values ≈ total error increase from {} → {a, b} = 99.
    s = res.shapley_values
    total = s["a"] + s["b"]
    assert abs(total - 99) < 5
    # Symmetry: |S(a) - S(b)| should be small. Loose bound because at
    # K=2 we only have 2! = 2 unique orderings and 100 samples easily
    # come out 55/45 instead of 50/50, producing |Δ| ≈ 10.
    assert abs(s["a"] - s["b"]) < 15


def test_unique_set_count_caches_aggressively():
    """With K=4 regions and 50 orderings, unique sets are bounded by
    2^4 = 16 even though we'd otherwise have 50*4=200 prefix sets."""
    masks = {f"r{i}": _mask(4, 4, i) for i in range(4)}
    region_ids = list(masks.keys())

    res = shapley_spot_check_image(
        image_id="img",
        region_ids=region_ids,
        region_masks=masks,
        original_image=_img(),
        original_error_km=1.0,
        apply_intervention=_apply_intervention,
        score_image=_make_score_image(lambda m: 1.0),
        error_for_pred=_error_for_pred,
        n_orderings=50,
        seed=0,
    )
    # Powerset of 4 elements has 16 sets; we score every non-empty one
    # (caller has the empty case for free), so n_unique_sets_scored
    # is between 1 and 16.
    assert res.n_unique_sets_scored <= 16
    assert res.n_orderings == 50


def test_empty_region_list_returns_empty_result():
    res = shapley_spot_check_image(
        image_id="img",
        region_ids=[],
        region_masks={},
        original_image=_img(),
        original_error_km=1.0,
        apply_intervention=_apply_intervention,
        score_image=lambda imgs: [],
        error_for_pred=_error_for_pred,
        n_orderings=10,
        seed=0,
    )
    assert res.shapley_values == {}
    assert res.region_ids == []
    assert res.n_orderings == 0


def test_static_vs_shapley_agreement_full_match():
    """If static and Shapley both rank r0 > r1 > r2, top-1 + Jaccard = 1."""
    static_scores = pd.DataFrame([
        {"image_id": "img", "region_id": "r0",
         "intervention_type": "min_across", "score_per_area": 100.0},
        {"image_id": "img", "region_id": "r1",
         "intervention_type": "min_across", "score_per_area": 50.0},
        {"image_id": "img", "region_id": "r2",
         "intervention_type": "min_across", "score_per_area": 10.0},
    ])
    shapley_res = ShapleyImageResult(
        image_id="img",
        region_ids=["r0", "r1", "r2"],
        shapley_values={"r0": 90.0, "r1": 40.0, "r2": 5.0},
        n_orderings=50,
        n_unique_sets_scored=8,
    )
    out = static_vs_shapley_agreement(
        static_scores, [shapley_res], top_k=2
    )
    assert out["top1_agreement"] == 1.0
    assert out["top2_jaccard"] == 1.0
    assert out["mean_spearman"] == pytest.approx(1.0)


def test_static_vs_shapley_agreement_disagreement():
    """Static says r0 is leaky; Shapley says r2 is. top-1 should disagree."""
    static_scores = pd.DataFrame([
        {"image_id": "img", "region_id": "r0",
         "intervention_type": "min_across", "score_per_area": 100.0},
        {"image_id": "img", "region_id": "r1",
         "intervention_type": "min_across", "score_per_area": 50.0},
        {"image_id": "img", "region_id": "r2",
         "intervention_type": "min_across", "score_per_area": 10.0},
    ])
    shapley_res = ShapleyImageResult(
        image_id="img",
        region_ids=["r0", "r1", "r2"],
        shapley_values={"r0": 5.0, "r1": 30.0, "r2": 90.0},
        n_orderings=50,
        n_unique_sets_scored=8,
    )
    out = static_vs_shapley_agreement(
        static_scores, [shapley_res], top_k=1
    )
    assert out["top1_agreement"] == 0.0


def test_static_vs_shapley_agreement_empty_inputs():
    out = static_vs_shapley_agreement(pd.DataFrame(), [], top_k=5)
    assert out["n_images"] == 0
