"""§9.5 region labeling — combine multi-source labels into the §7.2 schema.

This module is the *plumbing* that turns:
  - SAM regions (no labels yet)
  - OCR detections (per-image text boxes)
  - Object detector boxes (later, via GroundingDINO/YOLO-World)
  - CLIP zero-shot labels (later, via a CLIP @app.cls)

into the §9.5 schema:
  labels:           list[str]   # all sources combined
  dominant_label:   str | None  # priority-ordered top
  secondary_labels: list[str]   # rest, deduped

Per §9.5 the priority order is: ocr_text > object > clip_zero_shot > fallback.
The `assign_dominant_label` helper in `scoring/cue_taxonomy.py` already
implements that policy; this module does the matching step (which OCR
boxes overlap which SAM region by IoU > 0.3, etc.) and packages the
results.

Pure-Python — accepts plain numpy masks and label lists, no Modal calls.
"""
from __future__ import annotations

from typing import Iterable, Optional

import numpy as np
import pandas as pd

from geoleaklens.scoring.cue_taxonomy import assign_dominant_label


# §9.5 minimum IoU for cross-source label assignment. The spec uses 0.3.
DEFAULT_OVERLAP_IOU = 0.3


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    """Intersection-over-union of two same-shape bool masks.

    Returns 0.0 when either mask is empty (avoids 0/0). Inputs are coerced
    to bool so callers can pass uint8 masks too.
    """
    if a.shape != b.shape:
        raise ValueError(f"mask shape mismatch: {a.shape} vs {b.shape}")
    a_b = a.astype(bool, copy=False)
    b_b = b.astype(bool, copy=False)
    intersection = int(np.logical_and(a_b, b_b).sum())
    union = int(np.logical_or(a_b, b_b).sum())
    if union == 0:
        return 0.0
    return intersection / union


def label_regions_for_image(
    *,
    sam_masks: dict[str, np.ndarray],
    ocr_detections: Optional[list[dict]] = None,
    ocr_masks: Optional[dict[str, np.ndarray]] = None,
    object_detections: Optional[list[dict]] = None,
    object_masks: Optional[dict[str, np.ndarray]] = None,
    clip_zero_shot_labels: Optional[dict[str, str]] = None,
    overlap_iou: float = DEFAULT_OVERLAP_IOU,
) -> pd.DataFrame:
    """Assign §9.5 multi-source labels to a per-image set of SAM regions.

    Args:
        sam_masks: {region_id -> bool mask} — every SAM region for this image.
        ocr_detections: list of EasyOCR-style dicts (must include 'text'
            and a corresponding mask in `ocr_masks` keyed by detection id).
        ocr_masks: {detection_id -> bool mask} so we can compute IoU.
        object_detections: list of detector dicts with 'class' and an id
            referenced in `object_masks`. (Hook for the next E3 chunk —
            unused for now.)
        object_masks: {detection_id -> bool mask}.
        clip_zero_shot_labels: {region_id -> top-1 CLIP label} for SAM regions.
            (Hook for the next E3 chunk — unused for now.)
        overlap_iou: minimum IoU for cross-source assignment (§9.5: 0.3).

    Returns a DataFrame with one row per SAM region, keyed by `region_id`,
    carrying `labels`, `dominant_label`, `secondary_labels`, and the
    diagnostic field `overlapping_region_ids` (other SAM regions whose mask
    IoU > 0.3 with this one — useful for §9.5's "label overlap" panel).
    """
    rows: list[dict] = []
    sam_ids = list(sam_masks.keys())

    # Pre-compute SAM↔SAM IoUs so the diagnostic `overlapping_region_ids`
    # field is filled. O(N²) but N ≤ 80 per spec, so this is cheap.
    sam_overlaps: dict[str, list[str]] = {rid: [] for rid in sam_ids}
    for i, ri in enumerate(sam_ids):
        for rj in sam_ids[i + 1:]:
            iou = mask_iou(sam_masks[ri], sam_masks[rj])
            if iou > overlap_iou:
                sam_overlaps[ri].append(rj)
                sam_overlaps[rj].append(ri)

    # Build per-source label assignments per region.
    ocr_dets = ocr_detections or []
    ocr_msks = ocr_masks or {}
    obj_dets = object_detections or []
    obj_msks = object_masks or {}
    clip_labels = clip_zero_shot_labels or {}

    for region_id, sam_mask in sam_masks.items():
        labels_by_source: dict[str, list[str]] = {
            "ocr_text": [],
            "object": [],
            "clip_zero_shot": [],
            "fallback": [],
        }

        # OCR: any OCR detection whose mask IoU > threshold contributes
        # the source label `ocr_text` (we use the constant rather than the
        # actual extracted text — taxonomy buckets care about source, not
        # which sign was read).
        for det in ocr_dets:
            det_id = det.get("id") or det.get("region_id")
            if det_id is None or det_id not in ocr_msks:
                continue
            if mask_iou(sam_mask, ocr_msks[det_id]) > overlap_iou:
                labels_by_source["ocr_text"].append("ocr_text")
                break  # one ocr_text source label per region is enough

        # Object detector: each detector class with IoU > threshold
        # contributes the class name. Sort by IoU desc so the highest-
        # overlap class wins the dominant slot when this source is picked.
        obj_hits: list[tuple[float, str]] = []
        for det in obj_dets:
            det_id = det.get("id") or det.get("region_id")
            if det_id is None or det_id not in obj_msks:
                continue
            iou = mask_iou(sam_mask, obj_msks[det_id])
            if iou > overlap_iou:
                cls = str(det.get("class") or det.get("label") or "object")
                obj_hits.append((iou, cls))
        obj_hits.sort(reverse=True, key=lambda x: x[0])
        labels_by_source["object"] = [cls for _iou, cls in obj_hits]

        # CLIP zero-shot: top-1 label for this region (single string).
        clip_top = clip_labels.get(region_id)
        if clip_top:
            labels_by_source["clip_zero_shot"] = [clip_top]

        # Fallback: nothing from any source -> sam_unknown.
        any_label = any(labels_by_source[k] for k in labels_by_source)
        if not any_label:
            labels_by_source["fallback"] = ["sam_unknown"]

        dominant, secondaries = assign_dominant_label(labels_by_source)
        all_labels = [dominant] + secondaries if dominant else []

        rows.append({
            "region_id": region_id,
            "labels": all_labels,
            "dominant_label": dominant,
            "secondary_labels": secondaries,
            "overlapping_region_ids": sam_overlaps[region_id],
        })

    return pd.DataFrame(rows)
