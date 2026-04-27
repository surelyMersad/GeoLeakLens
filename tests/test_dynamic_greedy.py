"""§12.2 dynamic greedy unit tests with mocked model + intervention.

The class is decoupled from Modal entirely — `apply_intervention` and
`score_image` are passed in as callables. Tests can therefore run without
GPU and assert the algorithmic properties (joint-effect ordering, top-K,
budget recording, stop-early) directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
from PIL import Image

from geoleaklens.redaction.optimize import (
    BudgetSnapshot,
    DynamicGreedy,
    DynamicGreedyResult,
)


# ----- Test fixtures and helpers --------------------------------------------

def _img(h=4, w=4) -> Image.Image:
    return Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")


def _mask(h: int, w: int, region: str) -> np.ndarray:
    """Disjoint per-region masks at fixed positions in a 4x4 grid."""
    m = np.zeros((h, w), dtype=bool)
    if region == "a":
        m[0, 0] = True
    elif region == "b":
        m[1, 1] = True
    elif region == "c":
        m[2, 2] = True
    elif region == "d":
        m[3, 3] = True
    return m


def _make_inputs(scores_per_area: dict[str, float]):
    region_ids = list(scores_per_area.keys())
    regions = pd.DataFrame([
        {"image_id": "img", "region_id": r, "area_frac": 0.05,
         "mask_path": f"/fake/{r}"}
        for r in region_ids
    ])
    scores = pd.DataFrame([
        {"image_id": "img", "region_id": r,
         "intervention_type": "mean_mask",
         "score_per_area": scores_per_area[r]}
        for r in region_ids
    ])
    return regions, scores


def _mock_load_mask(h=4, w=4):
    """Inverse of mask_path → which region; returns the right mask."""
    def loader(path: str) -> np.ndarray:
        rid = path.rsplit("/", 1)[-1]
        return _mask(h, w, rid)
    return loader


def _mock_apply_intervention():
    """Identity: return the original image. The score_image fixture
    inspects the mask separately via state, so we don't actually edit
    pixels here. Real interventions are unit-tested in test_blur.py /
    test_inpaint.py / test_mean_mask.py."""
    def apply(image: Image.Image, mask: np.ndarray) -> Image.Image:
        # Stash the mask on the image so the mock score_image can read it.
        # Pillow images are mutable enough; we use a side-channel dict.
        image_id_state["last_mask"] = mask.copy()
        return image
    return apply


image_id_state: dict = {}


def _scoring_factory(error_for_mask):
    """Build a score_image that returns (lat, lon) where the implied error
    matches `error_for_mask(mask) -> error_km`. We synthesize lat/lon from
    error so haversine_km recovers the right value."""
    true_lat, true_lon = 0.0, 0.0  # simple — origin is "true location"

    def score(image: Image.Image):
        m = image_id_state.get("last_mask", np.zeros((4, 4), dtype=bool))
        err = error_for_mask(m)
        # Approximate: 1 degree of lat at the equator ≈ 111 km.
        lat_off = err / 111.0
        return (lat_off, 0.0)

    return score, (true_lat, true_lon)


# ----- Tests ----------------------------------------------------------------

def test_dynamic_greedy_picks_in_joint_effect_order():
    """Region 'b' has low single-region score but high JOINT score with 'a'.
    Static greedy would pick c (higher score_per_area). Dynamic greedy picks
    b because the joint effect of {a, b} > {a, c}."""
    regions, scores = _make_inputs({"a": 100.0, "b": 5.0, "c": 50.0, "d": 1.0})

    def error_for_mask(m: np.ndarray) -> float:
        # Mask pixel 'a' alone: 50 km. 'a' + 'b': 100 km. 'a' + 'c': 70 km.
        # 'a' + 'b' + 'c': 110. 'a' + 'b' + 'c' + 'd': 110.
        a, b, c, d = m[0, 0], m[1, 1], m[2, 2], m[3, 3]
        if a and b and c and d: return 110.0
        if a and b and c: return 110.0
        if a and b: return 100.0
        if a and c: return 70.0
        if a: return 50.0
        if b: return 5.0
        if c: return 30.0
        return 1.0

    score_image, (true_lat, true_lon) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(top_k=4, early_stop_consecutive=99).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.05, 0.10, 0.15],
        original_image=_img(),
        true_lat=true_lat, true_lon=true_lon,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    # First pick = 'a' (biggest standalone effect): ~50 km gain.
    # Second pick = 'b' (joint with a → 100 km > a+c=70): wins despite low score.
    assert result.snapshots[0.05].selected_region_ids[0] == "a"
    assert result.snapshots[0.10].selected_region_ids[:2] == ["a", "b"]
    assert result.n_picks >= 2


def test_top_k_caps_initial_candidate_pool():
    """Region 'd' has highest score but is excluded by top_k=2."""
    regions, scores = _make_inputs({"a": 90.0, "b": 80.0, "c": 70.0, "d": 100.0})

    def error_for_mask(m: np.ndarray) -> float:
        # Each region contributes +20km; d contributes +50km when included.
        e = 1.0
        if m[0, 0]: e += 20  # a
        if m[1, 1]: e += 20  # b
        if m[2, 2]: e += 20  # c
        if m[3, 3]: e += 50  # d
        return e

    score_image, (true_lat, true_lon) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(top_k=2, early_stop_consecutive=99).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.10, 0.20],
        original_image=_img(),
        true_lat=true_lat, true_lon=true_lon,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    # top_k=2 admits only {d, a} (highest scores). Despite c being in the
    # full set, dynamic greedy can never see it.
    selected = result.snapshots[0.20].selected_region_ids
    assert "c" not in selected
    assert "d" in selected
    assert "a" in selected


def test_records_state_at_each_budget_threshold():
    """Multiple budgets share one run; each gets the snapshot at first crossing."""
    regions, scores = _make_inputs({"a": 100.0, "b": 90.0, "c": 80.0})

    def error_for_mask(m):
        e = 1.0
        if m[0, 0]: e += 30
        if m[1, 1]: e += 20
        if m[2, 2]: e += 10
        return e

    score_image, (tl, ton) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(top_k=10, early_stop_consecutive=99).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.05, 0.10, 0.15],
        original_image=_img(),
        true_lat=tl, true_lon=ton,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    # Each region is 0.05 area; one pick per budget threshold.
    assert len(result.snapshots[0.05].selected_region_ids) == 1
    assert len(result.snapshots[0.10].selected_region_ids) == 2
    assert len(result.snapshots[0.15].selected_region_ids) == 3


def test_stops_when_no_candidate_increases_error():
    """If every remaining candidate has zero or negative joint effect, stop
    rather than spend budget on noise."""
    regions, scores = _make_inputs({"a": 100.0, "b": 50.0})

    def error_for_mask(m):
        # 'a' raises error to 50. 'b' alone → 1. 'a' + 'b' → 49 (b *helps*).
        a, b = m[0, 0], m[1, 1]
        if a and b: return 49.0
        if a: return 50.0
        if b: return 1.0
        return 1.0

    score_image, (tl, ton) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(top_k=10, early_stop_consecutive=99).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.10],
        original_image=_img(),
        true_lat=tl, true_lon=ton,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    # Pick 'a' (49 → 50 = +49). Then b would make it WORSE for redaction
    # purposes (50 → 49 = -1), so stop rather than add.
    assert result.snapshots[0.10].selected_region_ids == ["a"]


def test_early_stop_triggers_when_marginal_drops():
    """Two consecutive small marginal log-error increments → early stop."""
    regions, scores = _make_inputs({"a": 100.0, "b": 50.0, "c": 10.0, "d": 1.0})

    # Big jump on first pick, then tiny jumps. Δlog1p < 0.05 from pick 2 onward.
    def error_for_mask(m):
        e = 1.0
        if m[0, 0]: e += 100   # big jump
        if m[1, 1]: e += 0.5   # tiny
        if m[2, 2]: e += 0.5
        if m[3, 3]: e += 0.5
        return e

    score_image, (tl, ton) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(
        top_k=10, early_stop_delta=0.05, early_stop_consecutive=2,
    ).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.20],
        original_image=_img(),
        true_lat=tl, true_lon=ton,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    assert result.stopped_early is True
    assert result.n_picks <= 3   # stopped before consuming all 4


def test_skips_oversized_candidates_to_stay_within_budget():
    """A region whose area alone exceeds max_budget+tolerance is skipped
    even if it would dominate the joint effect."""
    regions, scores = _make_inputs({})  # rebuild with custom areas
    regions = pd.DataFrame([
        {"image_id": "img", "region_id": "huge",
         "area_frac": 0.40, "mask_path": "/fake/a"},
        {"image_id": "img", "region_id": "small",
         "area_frac": 0.05, "mask_path": "/fake/b"},
    ])
    scores = pd.DataFrame([
        {"image_id": "img", "region_id": "huge",
         "intervention_type": "mean_mask", "score_per_area": 1000.0},
        {"image_id": "img", "region_id": "small",
         "intervention_type": "mean_mask", "score_per_area": 1.0},
    ])

    def error_for_mask(m):
        a = m[0, 0]; b = m[1, 1]
        if a: return 100.0
        if b: return 5.0
        return 1.0

    score_image, (tl, ton) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(top_k=10, early_stop_consecutive=99).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.10],
        original_image=_img(),
        true_lat=tl, true_lon=ton,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    assert "huge" not in result.snapshots[0.10].selected_region_ids
    assert "small" in result.snapshots[0.10].selected_region_ids


def test_empty_scores_returns_empty_snapshots():
    regions = pd.DataFrame(columns=["image_id", "region_id", "area_frac", "mask_path"])
    scores = pd.DataFrame(columns=["image_id", "region_id",
                                   "intervention_type", "score_per_area"])
    result = DynamicGreedy().run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.10],
        original_image=_img(),
        true_lat=0.0, true_lon=0.0,
        original_error_km=42.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=lambda _img: (None, None),
    )
    assert result.n_picks == 0
    assert result.snapshots[0.10].selected_region_ids == []
    assert result.snapshots[0.10].edited_error_km == 42.0


def test_all_budgets_covered_in_snapshots():
    """Every requested budget appears in `snapshots`, even if no pick crossed it."""
    regions, scores = _make_inputs({"a": 100.0})

    def error_for_mask(m):
        return 100.0 if m[0, 0] else 1.0

    score_image, (tl, ton) = _scoring_factory(error_for_mask)
    result = DynamicGreedy(top_k=10, early_stop_consecutive=99).run_image(
        image_id="img",
        regions=regions, scores=scores,
        budgets=[0.05, 0.10, 0.20],
        original_image=_img(),
        true_lat=tl, true_lon=ton,
        original_error_km=1.0,
        load_mask=_mock_load_mask(),
        apply_intervention=_mock_apply_intervention(),
        score_image=score_image,
    )
    # Only one region (5% area). All three budgets see it.
    for b in (0.05, 0.10, 0.20):
        assert result.snapshots[b].selected_region_ids == ["a"]


def test_default_intervention_type_is_mean_mask():
    """Lock in the default so a runner that constructs DynamicGreedy()
    without args gets the §12.2 default."""
    g = DynamicGreedy()
    assert g.intervention_type == "mean_mask"
    assert g.top_k == 30
    assert g.early_stop_delta == 0.05
    assert g.early_stop_consecutive == 2
    assert g.name == "dynamic_greedy"
