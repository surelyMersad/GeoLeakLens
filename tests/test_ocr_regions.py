"""Unit tests for the OCR mask-conversion logic. Real EasyOCR runs via
`EasyOCRModal` on Modal — these tests use synthetic detection inputs."""
from __future__ import annotations

import numpy as np

from geoleaklens.segmentation.ocr_regions import (
    detections_to_regions,
    quad_to_mask,
)


# ----- quad_to_mask --------------------------------------------------------

def test_quad_to_mask_axis_aligned_rect():
    quad = [[10, 5], [50, 5], [50, 25], [10, 25]]
    mask = quad_to_mask(quad, (40, 60))
    assert mask.shape == (40, 60)
    assert mask.dtype == bool
    # Inside the rectangle.
    assert bool(mask[15, 30]) is True
    # Outside.
    assert bool(mask[0, 0]) is False
    # Pillow's polygon fill uses INCLUSIVE endpoints, so a quad spanning
    # x∈[10, 50] and y∈[5, 25] rasterizes 41 wide × 21 tall = 861 pixels.
    # Accept Pillow's actual rasterization with a small slack for any
    # rounding-mode quirk across versions.
    assert 850 <= int(mask.sum()) <= 870


def test_quad_to_mask_rotated_quadrilateral_keeps_area_below_bbox():
    """A diamond inscribed in a 20×20 bbox covers about half the bbox
    area — the whole point of using polygon masks over axis-aligned bboxes.
    Pillow's actual rasterization comes out at ~221 with inclusive
    endpoints; expected mathematical area is 200."""
    quad = [[10, 0], [20, 10], [10, 20], [0, 10]]
    mask = quad_to_mask(quad, (40, 40))
    rot_area = int(mask.sum())
    # Pillow rasterization: ~221. Bbox would be 400.
    assert 200 <= rot_area <= 240
    # Substantially less than the 400-pixel bbox.
    assert rot_area < 400


def test_quad_to_mask_clips_to_image_bounds():
    """A quad partially outside the image should not error and should
    only fill the in-bounds portion."""
    quad = [[-5, -5], [5, -5], [5, 5], [-5, 5]]  # 10×10 starting off-image
    mask = quad_to_mask(quad, (10, 10))
    assert mask.shape == (10, 10)
    # Only the (0..5, 0..5) quadrant is in-bounds.
    assert mask[0, 0]
    assert mask[3, 3]
    assert not mask[8, 8]


def test_quad_to_mask_degenerate_returns_empty():
    """Fewer than 3 points -> no polygon to fill."""
    assert int(quad_to_mask([[5, 5], [5, 5]], (20, 20)).sum()) == 0
    assert int(quad_to_mask([], (20, 20)).sum()) == 0


def test_quad_to_mask_zero_size_image():
    """Empty shape -> empty mask, no error."""
    mask = quad_to_mask([[1, 1], [2, 1], [2, 2], [1, 2]], (0, 10))
    assert mask.shape == (0, 10)
    assert mask.size == 0


# ----- detections_to_regions ----------------------------------------------

def test_detections_to_regions_basic():
    detections = [
        {"quad": [[0, 0], [10, 0], [10, 10], [0, 10]],
         "text": "HELLO", "confidence": 0.9},
    ]
    regions = detections_to_regions(detections, "img_x", (100, 100))
    assert len(regions) == 1
    r = regions[0]
    assert r["image_id"] == "img_x"
    assert r["region_id"].startswith("img_x__ocr_")
    assert r["source"] == "ocr"
    assert r["label"] == "ocr_text"
    assert r["ocr_text"] == "HELLO"
    assert r["object_confidence"] == 0.9
    # 10×10 rectangle (inclusive endpoints) → 11×11 = 121 pixels at 100×100.
    assert 115 <= int(r["area_px"]) <= 130
    assert r["area_frac"] == r["area_px"] / 10000.0
    # Mask is included for downstream serialization.
    assert isinstance(r["mask"], np.ndarray)
    assert r["mask"].dtype == bool


def test_detections_to_regions_filters_zero_area():
    """Degenerate quads (collinear / single-point) should drop out."""
    detections = [
        {"quad": [[10, 10], [10, 10], [10, 10], [10, 10]],
         "text": "X", "confidence": 0.99},   # single-point degenerate
        {"quad": [[0, 0], [0, 5], [0, 5], [0, 0]],   # zero-width
         "text": "Y", "confidence": 0.95},
    ]
    regions = detections_to_regions(detections, "img", (50, 50))
    assert regions == []


def test_detections_to_regions_assigns_unique_region_ids():
    detections = [
        {"quad": [[0, 0], [5, 0], [5, 5], [0, 5]], "text": "A", "confidence": 0.9},
        {"quad": [[10, 10], [15, 10], [15, 15], [10, 15]], "text": "B", "confidence": 0.9},
        {"quad": [[20, 20], [25, 20], [25, 25], [20, 25]], "text": "C", "confidence": 0.9},
    ]
    regions = detections_to_regions(detections, "img", (50, 50))
    ids = [r["region_id"] for r in regions]
    assert len(set(ids)) == 3
    # Region ids are zero-padded so sorted-string order matches detection order.
    assert ids == sorted(ids)


def test_detections_to_regions_preserves_ocr_metadata_columns():
    """§7.2 schema padding fields (stability_score, predicted_iou, etc.)
    are populated as None on OCR rows so the unioned regions parquet
    has consistent columns regardless of source."""
    det = [{"quad": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "text": "X", "confidence": 0.9}]
    regions = detections_to_regions(det, "img", (50, 50))
    r = regions[0]
    for key in ("stability_score", "predicted_iou",
                "parent_region_id", "dedup_group"):
        assert key in r
        assert r[key] is None


def test_detections_to_regions_empty_input():
    assert detections_to_regions([], "img", (50, 50)) == []


def test_detections_to_regions_zero_area_image_does_not_divide_by_zero():
    """If image_shape is (0, 0) we shouldn't crash; everything just drops."""
    det = [{"quad": [[0, 0], [5, 0], [5, 5], [0, 5]],
            "text": "X", "confidence": 0.9}]
    regions = detections_to_regions(det, "img", (0, 0))
    assert regions == []
