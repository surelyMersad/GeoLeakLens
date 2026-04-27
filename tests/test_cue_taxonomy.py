"""§13.E3 cue-taxonomy bucket-mapping + aggregation tests."""
from __future__ import annotations

import pandas as pd
import pytest

from geoleaklens.scoring.cue_taxonomy import (
    DOMINANT_LABEL_PRIORITY,
    SEMANTIC_BUCKETS,
    aggregate_by_bucket,
    assign_dominant_label,
    attach_buckets,
    label_to_bucket,
)


# ----- label_to_bucket ------------------------------------------------------

def test_label_to_bucket_handles_ocr_text():
    assert label_to_bucket("ocr_text") == "explicit_text_signage"


def test_label_to_bucket_normalizes_articles_and_case():
    """'A street sign', 'a street sign', 'street sign', 'Street Sign'
    all hit the same bucket. Important because OCR/CLIP/object detector
    emit varying capitalizations."""
    for variant in ["A street sign", "a street sign", "street sign",
                    "Street Sign", "  street sign  "]:
        assert label_to_bucket(variant) == "explicit_text_signage", variant


def test_label_to_bucket_clip_prompt_with_leading_article():
    assert label_to_bucket("a building facade") == "building_architecture"
    assert label_to_bucket("the sky") == "sky_skyline"


def test_label_to_bucket_unknown_label_falls_to_other():
    assert label_to_bucket("some_random_label") == "other"
    assert label_to_bucket("") == "other"
    assert label_to_bucket(None) == "other"


def test_label_to_bucket_explicit_unknown_source():
    """The §9.5 fallback `sam_unknown` / `grid_patch` go to `unknown`,
    which is distinct from `other` — `unknown` means the labeler ran
    but produced nothing; `other` means it produced something we
    don't have a bucket for."""
    assert label_to_bucket("sam_unknown") == "unknown"
    assert label_to_bucket("grid_patch") == "unknown"


def test_label_to_bucket_covers_all_spec_buckets():
    """Every spec bucket must have at least one label that maps to it."""
    bucketed = set()
    for label in [
        "ocr_text", "person", "license plate", "building",
        "storefront", "road marking", "sidewalk",
        "tree", "mountain", "sky", "utility pole", "bus stop",
    ]:
        bucketed.add(label_to_bucket(label))
    # All 12 leakage-relevant buckets covered (excluding `other` / `unknown`).
    expected = set(SEMANTIC_BUCKETS) - {"other", "unknown"}
    assert expected.issubset(bucketed), \
        f"missing buckets: {expected - bucketed}"


# ----- assign_dominant_label ------------------------------------------------

def test_dominant_label_picks_highest_priority_source():
    out, secondary = assign_dominant_label({
        "ocr_text": ["XYZ"],
        "object": ["building"],
        "clip_zero_shot": ["a building facade"],
    })
    assert out == "XYZ"
    # Other sources flow into secondaries, deduped, in priority-then-source order.
    assert "building" in secondary
    assert "a building facade" in secondary


def test_dominant_label_falls_through_to_clip_when_higher_empty():
    out, secondary = assign_dominant_label({
        "ocr_text": [],
        "object": [],
        "clip_zero_shot": ["a storefront"],
    })
    assert out == "a storefront"
    assert secondary == []


def test_dominant_label_object_first_in_list_wins_within_source():
    out, _ = assign_dominant_label({
        "object": ["building", "window"],  # caller sorts by IoU desc
    })
    assert out == "building"


def test_dominant_label_empty_input():
    out, secondary = assign_dominant_label({})
    assert out is None
    assert secondary == []

    out, secondary = assign_dominant_label({"object": []})
    assert out is None
    assert secondary == []


def test_dominant_label_dedupes_secondaries():
    out, secondary = assign_dominant_label({
        "ocr_text": ["sign"],
        "object": ["sign", "storefront"],  # 'sign' duplicates dominant
        "clip_zero_shot": ["sign"],         # also duplicates
    })
    assert out == "sign"
    assert secondary.count("sign") == 0
    assert "storefront" in secondary


def test_dominant_label_priority_order_constant():
    """Lock in the §9.5 priority for downstream code."""
    assert DOMINANT_LABEL_PRIORITY == (
        "ocr_text", "object", "clip_zero_shot", "fallback"
    )


# ----- attach_buckets -------------------------------------------------------

def test_attach_buckets_adds_dominant_and_secondary_columns():
    df = pd.DataFrame([
        {"image_id": "img", "region_id": "r1",
         "dominant_label": "ocr_text",
         "secondary_labels": ["building"]},
        {"image_id": "img", "region_id": "r2",
         "dominant_label": "a storefront",
         "secondary_labels": ["a sign", "the sky"]},
    ])
    out = attach_buckets(df)
    assert out.loc[0, "dominant_bucket"] == "explicit_text_signage"
    assert out.loc[0, "secondary_buckets"] == ["building_architecture"]
    assert out.loc[1, "dominant_bucket"] == "storefront_commercial"
    assert out.loc[1, "secondary_buckets"] == [
        "explicit_text_signage", "sky_skyline"
    ]


def test_attach_buckets_dedupes_secondary_buckets():
    df = pd.DataFrame([
        {"image_id": "img", "region_id": "r",
         "dominant_label": "ocr_text",
         "secondary_labels": ["sign", "street sign", "a shop sign"]},
    ])
    out = attach_buckets(df)
    # All three secondaries map to explicit_text_signage; dedupe keeps one.
    assert out.loc[0, "secondary_buckets"] == ["explicit_text_signage"]


def test_attach_buckets_handles_none_secondary_labels():
    df = pd.DataFrame([
        {"image_id": "img", "region_id": "r",
         "dominant_label": "building",
         "secondary_labels": None},
    ])
    out = attach_buckets(df)
    assert out.loc[0, "secondary_buckets"] == []


# ----- aggregate_by_bucket --------------------------------------------------

def _aggregator_inputs(rows):
    """Build the (regions_with_labels, scores) pair from compact fixtures."""
    regions = pd.DataFrame([
        {"image_id": r["image_id"], "region_id": r["region_id"],
         "dominant_label": r["dominant"],
         "secondary_labels": r.get("secondaries", []),
         "dominant_bucket": label_to_bucket(r["dominant"]),
         "secondary_buckets": [
             label_to_bucket(x) for x in r.get("secondaries", [])
         ]}
        for r in rows
    ])
    scores = pd.DataFrame([
        {"image_id": r["image_id"], "region_id": r["region_id"],
         "intervention_type": "min_across",
         "continuous_attribution": r["attr"],
         "score_per_area": r["spa"],
         "area_frac": r.get("area", 0.05)}
        for r in rows
    ])
    return regions, scores


def test_aggregate_counts_regions_per_bucket():
    rows = [
        {"image_id": "a", "region_id": "r1",
         "dominant": "ocr_text", "attr": 1.0, "spa": 100},
        {"image_id": "a", "region_id": "r2",
         "dominant": "building", "attr": 0.5, "spa": 5},
        {"image_id": "b", "region_id": "r1",
         "dominant": "ocr_text", "attr": 0.7, "spa": 50},
    ]
    regions, scores = _aggregator_inputs(rows)
    out = aggregate_by_bucket(regions, scores, top_k=5)
    by_bucket = out.set_index("bucket")["n_regions"]
    assert by_bucket["explicit_text_signage"] == 2
    assert by_bucket["building_architecture"] == 1
    # Other buckets are present (with n=0) so the figure layout is stable.
    assert by_bucket["vegetation"] == 0


def test_aggregate_mean_continuous_attribution_is_unnormalized():
    """§13.E3 PRIMARY: mean continuous_attribution per bucket — DON'T
    normalize by area or utility cost when describing where leakage lives."""
    rows = [
        {"image_id": "a", "region_id": "r1",
         "dominant": "ocr_text", "attr": 2.0, "spa": 1.0},
        {"image_id": "b", "region_id": "r1",
         "dominant": "ocr_text", "attr": 4.0, "spa": 1.0},
        {"image_id": "c", "region_id": "r1",
         "dominant": "building", "attr": 0.5, "spa": 1.0},
    ]
    regions, scores = _aggregator_inputs(rows)
    out = aggregate_by_bucket(regions, scores).set_index("bucket")
    assert out.loc["explicit_text_signage", "mean_continuous_attribution"] == 3.0
    assert out.loc["explicit_text_signage", "median_continuous_attribution"] == 3.0
    assert out.loc["building_architecture", "mean_continuous_attribution"] == 0.5


def test_aggregate_top1_share_uses_per_image_ranking():
    """`frac_top1_dominant` is the share of IMAGES whose top-1 region is
    in this bucket, not the share of all regions."""
    rows = [
        # img_a: top-1 (highest spa) is text; building is rank 2.
        {"image_id": "a", "region_id": "r1",
         "dominant": "ocr_text", "attr": 1.0, "spa": 100},
        {"image_id": "a", "region_id": "r2",
         "dominant": "building", "attr": 0.8, "spa": 50},
        # img_b: top-1 is building; nothing else.
        {"image_id": "b", "region_id": "r1",
         "dominant": "building", "attr": 0.5, "spa": 30},
        # img_c: top-1 is building; text rank 2.
        {"image_id": "c", "region_id": "r1",
         "dominant": "building", "attr": 0.6, "spa": 40},
        {"image_id": "c", "region_id": "r2",
         "dominant": "ocr_text", "attr": 0.2, "spa": 10},
    ]
    regions, scores = _aggregator_inputs(rows)
    out = aggregate_by_bucket(regions, scores, top_k=5).set_index("bucket")
    # 1 of 3 images has text as top-1 → 33%.
    assert out.loc["explicit_text_signage", "frac_top1_dominant"] == pytest.approx(1 / 3)
    # 2 of 3 have building as top-1 → 67%.
    assert out.loc["building_architecture", "frac_top1_dominant"] == pytest.approx(2 / 3)


def test_aggregate_secondary_overlap_per_image():
    """`frac_topk_secondary_overlap` is the share of images that have ANY
    top-k region whose secondary buckets contain the target — surfaces
    overlap (e.g. text on a building façade)."""
    rows = [
        # img_a: top-1 is a building, secondaries include text.
        {"image_id": "a", "region_id": "r1",
         "dominant": "building",
         "secondaries": ["a street sign"],
         "attr": 1.0, "spa": 100},
        # img_b: top-1 is a building, no text overlap.
        {"image_id": "b", "region_id": "r1",
         "dominant": "building",
         "secondaries": ["the sky"],
         "attr": 0.5, "spa": 30},
    ]
    regions, scores = _aggregator_inputs(rows)
    out = aggregate_by_bucket(regions, scores, top_k=5).set_index("bucket")
    # 1 of 2 images has top-region with secondary text overlap.
    assert out.loc["explicit_text_signage", "frac_top5_secondary_overlap"] == pytest.approx(0.5)
    # No image has top-region with secondary vegetation overlap.
    assert out.loc["vegetation", "frac_top5_secondary_overlap"] == 0.0


def test_aggregate_filters_by_intervention_type():
    """Only `min_across` rows (default) feed the aggregation; per-intervention
    rows from the same scores parquet must be excluded."""
    regions = pd.DataFrame([
        {"image_id": "a", "region_id": "r",
         "dominant_label": "building", "secondary_labels": [],
         "dominant_bucket": "building_architecture",
         "secondary_buckets": []},
    ])
    scores = pd.DataFrame([
        {"image_id": "a", "region_id": "r",
         "intervention_type": "mean_mask",
         "continuous_attribution": 99.0, "score_per_area": 1.0,
         "area_frac": 0.05},
        {"image_id": "a", "region_id": "r",
         "intervention_type": "min_across",
         "continuous_attribution": 0.5, "score_per_area": 1.0,
         "area_frac": 0.05},
    ])
    out = aggregate_by_bucket(regions, scores).set_index("bucket")
    # Should pick up the min_across=0.5, NOT the mean_mask=99.0.
    assert out.loc["building_architecture", "mean_continuous_attribution"] == 0.5


def test_aggregate_empty_inputs_return_empty_with_correct_columns():
    out_a = aggregate_by_bucket(pd.DataFrame(), pd.DataFrame())
    assert out_a.empty
    assert "mean_continuous_attribution" in out_a.columns

    regions, _ = _aggregator_inputs([
        {"image_id": "a", "region_id": "r",
         "dominant": "ocr_text", "attr": 1.0, "spa": 1.0},
    ])
    out_b = aggregate_by_bucket(regions, pd.DataFrame())
    assert out_b.empty


def test_aggregate_buckets_in_spec_order():
    """The §13.E3 figure plots buckets left-to-right in `SEMANTIC_BUCKETS`
    order. The output preserves that even when only some buckets have data."""
    rows = [
        {"image_id": "a", "region_id": "r1",
         "dominant": "vegetation", "attr": 0.3, "spa": 5},
        {"image_id": "b", "region_id": "r1",
         "dominant": "ocr_text", "attr": 1.0, "spa": 100},
    ]
    regions, scores = _aggregator_inputs(rows)
    out = aggregate_by_bucket(regions, scores)
    bucket_order = list(out["bucket"])
    spec_order = list(SEMANTIC_BUCKETS)
    # Spec buckets appear in spec order in the output (with 0-rows for
    # absent ones).
    spec_positions = [bucket_order.index(b) for b in spec_order if b in bucket_order]
    assert spec_positions == sorted(spec_positions)
