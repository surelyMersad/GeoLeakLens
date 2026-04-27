"""§13.E3 cue-taxonomy aggregation: map raw region labels into one of the 14
spec buckets, then aggregate per-bucket leakage statistics.

This module is the *shape* of the §13.E3 deliverable, decoupled from how
labels were obtained. Whether a region's labels come from EasyOCR
(`ocr_text`), an object detector (`storefront sign`), or CLIP zero-shot
(`a building facade`), the aggregation here treats them uniformly.

Three primary capabilities, all pure Python (no Modal, no models):

1. `label_to_bucket(label)` — collapses the long tail of detector class
   names + CLIP prompts into one of the 14 §13.E3 semantic buckets.
2. `assign_dominant_label(labels)` — applies the §9.5 priority order
   (ocr_text > object > clip_zero_shot > fallback) to a region's full
   label set.
3. `aggregate_by_bucket(scored_labeled_regions)` — produces the per-bucket
   table the §13.E3 figure plots: mean/median continuous_attribution,
   top-1 / top-5 share, secondary-label overlap, n_regions, mean area.

All §9.5 schema fields (`labels`, `dominant_label`, `secondary_labels`)
are accepted as DataFrame columns. The aggregator joins this with the
§7.5 leakage-score parquet and emits §13.E3's `tables/cue_taxonomy.csv`.
"""
from __future__ import annotations

from typing import Iterable, Optional, Sequence

import pandas as pd


# §13.E3 specifies these 14 buckets (in this order) for the headline figure.
SEMANTIC_BUCKETS: tuple[str, ...] = (
    "explicit_text_signage",
    "face_person",
    "license_plate_vehicle",
    "building_architecture",
    "storefront_commercial",
    "road_marking",
    "sidewalk_curb",
    "vegetation",
    "terrain_mountain_water",
    "sky_skyline",
    "utility_pole_wires",
    "transit_infrastructure",
    "other",
    "unknown",
)


# §9.5 dominant-label priority order. Sources earlier in the tuple win when
# multiple labels are present on a region.
DOMINANT_LABEL_PRIORITY: tuple[str, ...] = (
    "ocr_text",
    "object",
    "clip_zero_shot",
    "fallback",
)


# Map the §9.4 detector prompt list, the §9.5 CLIP zero-shot prompt list,
# and the special `ocr_text` source label into the 14 buckets.
#
# Keys are the lowercased token string returned by the labeler — case is
# normalized inside `label_to_bucket`. Both detector class names and CLIP
# prompts (with leading "a/the") map to the same buckets.
_LABEL_BUCKET_MAP: dict[str, str] = {
    # Explicit text / signage
    "ocr_text": "explicit_text_signage",
    "street sign": "explicit_text_signage",
    "storefront sign": "explicit_text_signage",
    "road sign": "explicit_text_signage",
    "shop sign": "explicit_text_signage",
    "sign": "explicit_text_signage",
    "transit sign": "explicit_text_signage",
    "subway sign": "explicit_text_signage",
    "school sign": "explicit_text_signage",
    "hospital sign": "explicit_text_signage",
    "flag": "explicit_text_signage",  # text/symbol carrier

    # Faces / people
    "face": "face_person",
    "person": "face_person",
    "people": "face_person",

    # Vehicles / license plates
    "license plate": "license_plate_vehicle",
    "car": "license_plate_vehicle",
    "vehicle": "license_plate_vehicle",
    "bus": "license_plate_vehicle",
    "taxi": "license_plate_vehicle",
    "truck": "license_plate_vehicle",

    # Buildings / architecture
    "building": "building_architecture",
    "building facade": "building_architecture",
    "facade": "building_architecture",
    "religious building": "building_architecture",
    "window": "building_architecture",
    "bridge": "building_architecture",

    # Storefronts / commercial
    "storefront": "storefront_commercial",

    # Road / sidewalk
    "road marking": "road_marking",
    "traffic light": "road_marking",
    "sidewalk": "sidewalk_curb",
    "curb": "sidewalk_curb",

    # Vegetation
    "tree": "vegetation",
    "palm tree": "vegetation",
    "vegetation": "vegetation",

    # Terrain / mountain / water
    "mountain": "terrain_mountain_water",
    "mountains": "terrain_mountain_water",
    "water": "terrain_mountain_water",

    # Sky
    "sky": "sky_skyline",
    "the sky": "sky_skyline",
    "skyline": "sky_skyline",

    # Utilities
    "utility pole": "utility_pole_wires",
    "power line": "utility_pole_wires",

    # Transit infrastructure (stations, stops)
    "bus stop": "transit_infrastructure",
    "train station": "transit_infrastructure",

    # Background fallbacks
    "sam_unknown": "unknown",
    "grid_patch": "unknown",
}


def _normalize_label(label: str) -> str:
    """Lowercase and drop leading articles so 'A Street Sign' and
    'a street sign' and 'street sign' all hit the same map entry."""
    s = (label or "").strip().lower()
    for prefix in ("the ", "a ", "an "):
        if s.startswith(prefix):
            s = s[len(prefix):]
            break
    return s


def label_to_bucket(label: Optional[str]) -> str:
    """Collapse a single raw label into one of the 14 §13.E3 buckets.

    Unknown / empty labels go to the `other` bucket so they show up
    distinctly from the explicit `unknown` (which is the §9.5 fallback
    when a region matched nothing).
    """
    if not label:
        return "other"
    normalized = _normalize_label(label)
    return _LABEL_BUCKET_MAP.get(normalized, "other")


def assign_dominant_label(
    labels_by_source: dict[str, Sequence[str]]
) -> tuple[Optional[str], list[str]]:
    """Apply §9.5 priority order to choose the dominant label.

    `labels_by_source` is a mapping like:

        {
          "ocr_text": ["XYZ STREET"],          # one OCR hit
          "object":   ["building", "window"],  # multiple detector classes
          "clip_zero_shot": ["a building facade"],
          "fallback": [],
        }

    Returns `(dominant_label, secondary_labels)`. The dominant label comes
    from the highest-priority non-empty source. Within `object`, when
    multiple detector classes overlap the region, the *first* one in the
    list is taken — caller is responsible for sorting by IoU desc.

    Empty input -> (None, []).
    """
    if not labels_by_source:
        return None, []

    dominant: Optional[str] = None
    flat_secondaries: list[str] = []
    for source in DOMINANT_LABEL_PRIORITY:
        labels = labels_by_source.get(source, [])
        if not labels:
            continue
        if dominant is None:
            dominant = labels[0]
            flat_secondaries.extend(labels[1:])
        else:
            flat_secondaries.extend(labels)
    # Also include any sources we didn't list in the priority order
    # (forward-compat for new label sources).
    for source, labels in labels_by_source.items():
        if source in DOMINANT_LABEL_PRIORITY:
            continue
        flat_secondaries.extend(labels)
    # Dedupe while preserving order.
    seen: set[str] = set([dominant]) if dominant else set()
    deduped = []
    for s in flat_secondaries:
        if s in seen:
            continue
        seen.add(s)
        deduped.append(s)
    return dominant, deduped


def attach_buckets(regions_with_labels: pd.DataFrame) -> pd.DataFrame:
    """Add `dominant_bucket` + `secondary_buckets` columns.

    `regions_with_labels` is expected to carry the §9.5 schema fields
    `dominant_label: str` and `secondary_labels: list[str]`. A copy is
    returned; the input is left untouched so callers can rerun with
    different bucket maps if desired.
    """
    out = regions_with_labels.copy()
    out["dominant_bucket"] = out["dominant_label"].map(label_to_bucket)

    def _secondary_buckets(labels):
        if labels is None:
            return []
        seen: set[str] = set()
        result: list[str] = []
        for label in labels:
            b = label_to_bucket(label)
            if b in seen:
                continue
            seen.add(b)
            result.append(b)
        return result

    out["secondary_buckets"] = out["secondary_labels"].map(_secondary_buckets)
    return out


def aggregate_by_bucket(
    regions_with_labels: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    intervention_type: str = "min_across",
    top_k: int = 5,
) -> pd.DataFrame:
    """Per-bucket rollup that becomes one row of `tables/cue_taxonomy.csv`.

    Joins regions on (image_id, region_id) with scores filtered to the
    headline `intervention_type` (default §10.7 min_across). Computes:

      - mean / median `continuous_attribution`  (PRIMARY per §13.E3)
      - mean `area_frac`                        (informational)
      - n_regions in bucket
      - share of top-1 regions per image whose dominant bucket is this
      - share of top-`top_k` regions whose dominant bucket is this
      - share of top-`top_k` regions whose secondary_buckets contain this

    Returns one row per bucket present in `regions_with_labels`, ordered by
    the `SEMANTIC_BUCKETS` constant (so callers can plot in spec order).
    """
    if regions_with_labels.empty or scores.empty:
        return pd.DataFrame(columns=[
            "bucket", "n_regions",
            "mean_continuous_attribution", "median_continuous_attribution",
            "mean_area_frac",
            "frac_top1_dominant", f"frac_top{top_k}_dominant",
            f"frac_top{top_k}_secondary_overlap",
        ])

    s = scores.loc[
        scores["intervention_type"] == intervention_type,
        ["image_id", "region_id", "continuous_attribution",
         "score_per_area", "area_frac"],
    ]
    df = regions_with_labels.merge(
        s, on=["image_id", "region_id"], how="inner",
        suffixes=("", "_score"),
    )
    if df.empty:
        return pd.DataFrame(columns=[
            "bucket", "n_regions",
            "mean_continuous_attribution", "median_continuous_attribution",
            "mean_area_frac",
            "frac_top1_dominant", f"frac_top{top_k}_dominant",
            f"frac_top{top_k}_secondary_overlap",
        ])

    # Use the score-side area_frac (matches what the static-greedy ranking
    # was computed against).
    if "area_frac_score" in df.columns:
        df = df.rename(columns={
            "area_frac_score": "area_frac_used",
        })
    elif "area_frac" in df.columns:
        df = df.rename(columns={"area_frac": "area_frac_used"})

    # Per-image rank by score_per_area desc, so we can ask "top-1" etc.
    df["rank_within_image"] = (
        df.groupby("image_id")["score_per_area"]
        .rank(ascending=False, method="first")
        .astype(int)
    )

    by_image_top1 = df[df["rank_within_image"] == 1]
    by_image_topk = df[df["rank_within_image"] <= top_k]

    n_images = df["image_id"].nunique()

    rows: list[dict] = []
    bucket_to_count_top1 = (
        by_image_top1["dominant_bucket"].value_counts().to_dict()
    )
    bucket_to_count_topk = (
        by_image_topk["dominant_bucket"].value_counts().to_dict()
    )

    # Secondary-overlap counts per bucket, restricted to top_k rows.
    secondary_overlap_count: dict[str, set[str]] = {b: set() for b in SEMANTIC_BUCKETS}
    secondary_overlap_count["other"] = set()
    secondary_overlap_count["unknown"] = set()
    for _, row in by_image_topk.iterrows():
        for b in (row.get("secondary_buckets") or []):
            secondary_overlap_count.setdefault(b, set()).add(row["image_id"])

    buckets_present = set(df["dominant_bucket"].unique())
    # Always include the spec buckets in spec order; emit 0-row entries for
    # buckets that didn't appear at all so the figure layout stays consistent.
    bucket_order = list(SEMANTIC_BUCKETS) + sorted(
        b for b in buckets_present if b not in SEMANTIC_BUCKETS
    )
    for bucket in bucket_order:
        sub = df[df["dominant_bucket"] == bucket]
        n_regions = len(sub)
        if n_regions == 0 and bucket not in buckets_present:
            mean_attr = 0.0
            median_attr = 0.0
            mean_area = 0.0
        else:
            mean_attr = float(sub["continuous_attribution"].mean()) if n_regions else 0.0
            median_attr = float(sub["continuous_attribution"].median()) if n_regions else 0.0
            mean_area = float(sub["area_frac_used"].mean()) if n_regions else 0.0
        rows.append({
            "bucket": bucket,
            "n_regions": int(n_regions),
            "mean_continuous_attribution": mean_attr,
            "median_continuous_attribution": median_attr,
            "mean_area_frac": mean_area,
            "frac_top1_dominant": (
                bucket_to_count_top1.get(bucket, 0) / n_images
                if n_images else 0.0
            ),
            f"frac_top{top_k}_dominant": (
                bucket_to_count_topk.get(bucket, 0) /
                max(by_image_topk["image_id"].nunique() * top_k, 1)
            ),
            f"frac_top{top_k}_secondary_overlap": (
                len(secondary_overlap_count.get(bucket, set())) / n_images
                if n_images else 0.0
            ),
        })
    return pd.DataFrame(rows)
