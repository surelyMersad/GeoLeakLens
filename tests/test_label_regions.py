"""§9.5 region-labeling tests with synthetic masks. No model dependencies."""
from __future__ import annotations

import numpy as np
import pytest

from geoleaklens.segmentation.label_regions import (
    DEFAULT_OVERLAP_IOU,
    label_regions_for_image,
    mask_iou,
)


def _box_mask(h: int, w: int, *, x: int, y: int, dx: int, dy: int) -> np.ndarray:
    m = np.zeros((h, w), dtype=bool)
    m[y:y + dy, x:x + dx] = True
    return m


# ----- mask_iou -------------------------------------------------------------

def test_mask_iou_identical_is_one():
    a = _box_mask(20, 20, x=5, y=5, dx=10, dy=10)
    assert mask_iou(a, a) == 1.0


def test_mask_iou_disjoint_is_zero():
    a = _box_mask(20, 20, x=0, y=0, dx=5, dy=5)
    b = _box_mask(20, 20, x=10, y=10, dx=5, dy=5)
    assert mask_iou(a, b) == 0.0


def test_mask_iou_partial_overlap():
    """Two 10×10 boxes overlapping in a 5×5 region: IoU = 25 / 175."""
    a = _box_mask(40, 40, x=0, y=0, dx=10, dy=10)
    b = _box_mask(40, 40, x=5, y=5, dx=10, dy=10)
    expected = 25 / (100 + 100 - 25)
    assert mask_iou(a, b) == pytest.approx(expected)


def test_mask_iou_empty_masks_return_zero():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((10, 10), dtype=bool)
    assert mask_iou(a, b) == 0.0


def test_mask_iou_shape_mismatch_raises():
    a = np.zeros((10, 10), dtype=bool)
    b = np.zeros((20, 20), dtype=bool)
    with pytest.raises(ValueError):
        mask_iou(a, b)


def test_mask_iou_accepts_uint8_input():
    a = _box_mask(10, 10, x=0, y=0, dx=5, dy=5).astype(np.uint8)
    b = _box_mask(10, 10, x=0, y=0, dx=5, dy=5).astype(np.uint8)
    assert mask_iou(a, b) == 1.0


# ----- label_regions_for_image ---------------------------------------------

def test_ocr_overlap_sets_ocr_text_dominant_label():
    """A SAM region overlapping an OCR text box gets ocr_text as dominant."""
    sam = {"r1": _box_mask(40, 40, x=0, y=0, dx=20, dy=20)}
    ocr_dets = [{"id": "ocr_0", "text": "STREET 5"}]
    ocr_masks = {"ocr_0": _box_mask(40, 40, x=5, y=5, dx=15, dy=15)}
    out = label_regions_for_image(
        sam_masks=sam, ocr_detections=ocr_dets, ocr_masks=ocr_masks,
    )
    row = out[out.region_id == "r1"].iloc[0]
    assert row["dominant_label"] == "ocr_text"


def test_object_label_when_no_ocr():
    """No OCR overlap → object detector label takes priority."""
    sam = {"r1": _box_mask(40, 40, x=0, y=0, dx=20, dy=20)}
    obj = [{"id": "obj_0", "class": "building"}]
    obj_m = {"obj_0": _box_mask(40, 40, x=2, y=2, dx=15, dy=15)}
    out = label_regions_for_image(
        sam_masks=sam, object_detections=obj, object_masks=obj_m,
    )
    row = out[out.region_id == "r1"].iloc[0]
    assert row["dominant_label"] == "building"


def test_clip_label_when_no_ocr_no_object():
    """OCR + object empty → CLIP zero-shot is dominant."""
    sam = {"r1": _box_mask(40, 40, x=0, y=0, dx=20, dy=20)}
    out = label_regions_for_image(
        sam_masks=sam,
        clip_zero_shot_labels={"r1": "a building facade"},
    )
    row = out[out.region_id == "r1"].iloc[0]
    assert row["dominant_label"] == "a building facade"


def test_no_labels_falls_through_to_sam_unknown():
    """No source contributes anything → fallback `sam_unknown`."""
    sam = {"r1": _box_mask(40, 40, x=0, y=0, dx=20, dy=20)}
    out = label_regions_for_image(sam_masks=sam)
    row = out[out.region_id == "r1"].iloc[0]
    assert row["dominant_label"] == "sam_unknown"
    assert row["labels"] == ["sam_unknown"]


def test_low_overlap_does_not_assign():
    """OCR box with IoU below the 0.3 threshold should NOT contribute."""
    sam = {"r1": _box_mask(40, 40, x=0, y=0, dx=10, dy=10)}
    ocr = [{"id": "ocr_0"}]
    # 1×1 overlap with 10×10 box → IoU = 1/(100+1-1) = 1/100 — way under 0.3.
    ocr_m = {"ocr_0": _box_mask(40, 40, x=9, y=9, dx=2, dy=2)}
    out = label_regions_for_image(
        sam_masks=sam, ocr_detections=ocr, ocr_masks=ocr_m,
        # Add a CLIP label so we can confirm OCR didn't fire.
        clip_zero_shot_labels={"r1": "a window"},
    )
    row = out[out.region_id == "r1"].iloc[0]
    assert row["dominant_label"] == "a window"


def test_object_detector_sorts_by_iou_desc():
    """When multiple object classes overlap a region above the 0.3
    threshold, the highest-IoU one is dominant; lower-IoU ones flow into
    secondaries. Both must clear the threshold to be assigned at all."""
    sam = {"r1": _box_mask(40, 40, x=0, y=0, dx=20, dy=20)}
    obj = [
        {"id": "obj_low",  "class": "window"},
        {"id": "obj_high", "class": "building"},
    ]
    # Both above 0.3 IoU with sam (400 px):
    #   building 18×18 = 324 inside → IoU 324/400 = 0.81
    #   window   14×14 = 196 inside → IoU 196/(400+196-196) = 0.49
    obj_m = {
        "obj_low":  _box_mask(40, 40, x=6,  y=6,  dx=14, dy=14),
        "obj_high": _box_mask(40, 40, x=1,  y=1,  dx=18, dy=18),
    }
    out = label_regions_for_image(
        sam_masks=sam, object_detections=obj, object_masks=obj_m,
    )
    row = out[out.region_id == "r1"].iloc[0]
    assert row["dominant_label"] == "building"
    assert "window" in row["secondary_labels"]
    # And building appears BEFORE window in `labels` overall.
    assert row["labels"].index("building") < row["labels"].index("window")


def test_overlapping_region_ids_finds_sam_sam_overlap():
    """If two SAM regions overlap above threshold, each should reference
    the other in `overlapping_region_ids` (the §9.5 diagnostic field)."""
    sam = {
        "r1": _box_mask(40, 40, x=0,  y=0, dx=20, dy=20),
        "r2": _box_mask(40, 40, x=5,  y=5, dx=20, dy=20),  # high overlap
        "r3": _box_mask(40, 40, x=30, y=30, dx=5, dy=5),   # disjoint
    }
    out = label_regions_for_image(sam_masks=sam).set_index("region_id")
    assert "r2" in out.loc["r1", "overlapping_region_ids"]
    assert "r1" in out.loc["r2", "overlapping_region_ids"]
    assert out.loc["r3", "overlapping_region_ids"] == []


def test_default_overlap_iou_constant():
    """Lock in the §9.5 threshold."""
    assert DEFAULT_OVERLAP_IOU == 0.3


def test_empty_sam_input_returns_empty_dataframe():
    out = label_regions_for_image(sam_masks={})
    assert out.empty
