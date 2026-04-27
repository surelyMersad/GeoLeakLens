"""§13.E6 ablation analyzer tests."""
from __future__ import annotations

import math

import pandas as pd
import pytest

from geoleaklens.scoring.intervention_ablation import (
    pairwise_agreement_summary,
    per_image_agreement,
)


def _rows(image_id, scores_per_intervention):
    """Compact factory: scores_per_intervention is dict {intv: [(rid, spa)]}."""
    rows = []
    for intv, pairs in scores_per_intervention.items():
        for rid, spa in pairs:
            rows.append({
                "image_id": image_id,
                "region_id": rid,
                "intervention_type": intv,
                "score_per_area": spa,
                # Carry through the rest of the §7.5 schema with placeholders
                # so the merge on image_id, region_id works:
                "continuous_attribution": spa,
                "area_frac": 0.05,
            })
    return rows


def test_top1_match_when_same_region_ranks_first():
    rows = _rows("img", {
        "mean_mask": [("r1", 1.0), ("r2", 0.5), ("r3", 0.1)],
        "blur":      [("r1", 0.9), ("r2", 0.6), ("r3", 0.2)],
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur", top_k=2)
    assert len(out) == 1
    assert bool(out.iloc[0]["top1_match"]) is True


def test_top1_match_false_when_different_top_regions():
    rows = _rows("img", {
        "mean_mask": [("r1", 1.0), ("r2", 0.5)],
        "blur":      [("r1", 0.5), ("r2", 1.0)],   # ranks reversed
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur", top_k=2)
    assert bool(out.iloc[0]["top1_match"]) is False


def test_topk_jaccard_full_overlap():
    rows = _rows("img", {
        "mean_mask": [("r1", 3), ("r2", 2), ("r3", 1), ("r4", 0)],
        "blur":      [("r1", 2), ("r2", 3), ("r3", 1), ("r4", 0)],
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur", top_k=3)
    # Top-3 of both: {r1, r2, r3} → Jaccard 1.0
    assert out.iloc[0]["top3_jaccard"] == pytest.approx(1.0)


def test_topk_jaccard_partial_overlap():
    rows = _rows("img", {
        "mean_mask": [("r1", 4), ("r2", 3), ("r3", 2), ("r4", 1)],  # top-2: r1, r2
        "blur":      [("r1", 4), ("r3", 3), ("r2", 2), ("r4", 1)],  # top-2: r1, r3
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur", top_k=2)
    # Intersection {r1}, union {r1, r2, r3} → 1/3
    assert out.iloc[0]["top2_jaccard"] == pytest.approx(1 / 3)


def test_spearman_perfect_correlation():
    rows = _rows("img", {
        "mean_mask": [("r1", 4), ("r2", 3), ("r3", 2), ("r4", 1)],
        "blur":      [("r1", 4), ("r2", 3), ("r3", 2), ("r4", 1)],
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur")
    assert out.iloc[0]["spearman"] == pytest.approx(1.0)


def test_spearman_perfect_anti_correlation():
    rows = _rows("img", {
        "mean_mask": [("r1", 4), ("r2", 3), ("r3", 2), ("r4", 1)],
        "blur":      [("r1", 1), ("r2", 2), ("r3", 3), ("r4", 4)],
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur")
    assert out.iloc[0]["spearman"] == pytest.approx(-1.0)


def test_spearman_constant_values_returns_nan():
    rows = _rows("img", {
        "mean_mask": [("r1", 1), ("r2", 1), ("r3", 1)],
        "blur":      [("r1", 5), ("r2", 5), ("r3", 5)],
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur")
    assert math.isnan(out.iloc[0]["spearman"])


def test_skips_images_missing_one_intervention():
    rows = _rows("a", {
        "mean_mask": [("r1", 1), ("r2", 0.5)],
        "blur":      [("r1", 1), ("r2", 0.5)],
    }) + _rows("b", {
        "mean_mask": [("r1", 1), ("r2", 0.5)],
        # no blur for image b
    })
    out = per_image_agreement(pd.DataFrame(rows), "mean_mask", "blur")
    # Only image 'a' fully covered.
    assert set(out["image_id"]) == {"a"}


def test_pairwise_summary_aggregates_across_images():
    """Two images with identical perfect agreement → top1_match_rate=1.0."""
    rows = []
    for img in ("a", "b"):
        rows.extend(_rows(img, {
            "mean_mask": [("r1", 1.0), ("r2", 0.5)],
            "blur":      [("r1", 0.9), ("r2", 0.4)],
            "inpaint":   [("r1", 1.1), ("r2", 0.3)],
        }))
    out = pairwise_agreement_summary(
        pd.DataFrame(rows),
        ["mean_mask", "blur", "inpaint"],
        top_k=2,
    )
    # 3 unordered pairs from 3 interventions.
    assert len(out) == 3
    for _, row in out.iterrows():
        assert row["n_images"] == 2
        assert row["top1_match_rate"] == 1.0


def test_pairwise_summary_mixed_agreement():
    """Image a: agree on top-1 (r1). Image b: disagree (r1 vs r2)."""
    rows = []
    rows.extend(_rows("a", {
        "mean_mask": [("r1", 1.0), ("r2", 0.5)],
        "blur":      [("r1", 0.9), ("r2", 0.4)],
    }))
    rows.extend(_rows("b", {
        "mean_mask": [("r1", 1.0), ("r2", 0.5)],
        "blur":      [("r1", 0.5), ("r2", 1.0)],
    }))
    out = pairwise_agreement_summary(
        pd.DataFrame(rows), ["mean_mask", "blur"], top_k=2,
    )
    assert len(out) == 1
    assert out.iloc[0]["top1_match_rate"] == pytest.approx(0.5)


def test_pairwise_summary_handles_empty_intervention_type():
    """If one of the requested interventions has no rows, the pair is
    skipped quietly (no exception)."""
    rows = _rows("a", {
        "mean_mask": [("r1", 1.0), ("r2", 0.5)],
    })
    out = pairwise_agreement_summary(
        pd.DataFrame(rows), ["mean_mask", "blur"], top_k=2,
    )
    assert out.empty
