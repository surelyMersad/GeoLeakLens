"""§9.3 OCR region utilities — local logic, no Modal.

Turns the §7.2 OCR detection records produced by `EasyOCRModal.detect`
into region rows compatible with the rest of the pipeline. Two pieces:

1. `quad_to_mask(quad, shape)` — convert a quadrilateral to a binary mask
   via Pillow's polygon fill. EasyOCR returns rotated quads (text isn't
   axis-aligned), so the bounding box is loose and the polygon is the
   honest mask.
2. `detections_to_regions(detections, image_id, shape)` — flatten the
   detector output into §7.2 region rows. Source is `"ocr"`, label is
   `"ocr_text"`, the OCR string lands in the `ocr_text` column, and the
   per-detection confidence rides as `object_confidence` for now.

The full §9.6 region merge (OCR ∪ SAM ∪ object detector ∪ grid, deduped
by IoU) lives in a separate module — this one is just the OCR side.
"""
from __future__ import annotations

from typing import Iterable

import numpy as np
from PIL import Image, ImageDraw


def quad_to_mask(
    quad: Iterable[Iterable[int]],
    shape: tuple[int, int],
) -> np.ndarray:
    """Rasterize a 4-point quadrilateral into a (H, W) bool mask.

    `shape = (H, W)` matches `numpy.shape` convention. Pillow's
    `ImageDraw.polygon(... fill=1)` uses inclusive endpoints — a quad
    spanning x∈[10, 50] rasterizes 41 pixels wide (10 through 50), not 40.

    Degenerate quads — fewer than 3 *distinct* points (a single point or a
    zero-width line) — return an empty mask. Real EasyOCR doesn't produce
    these but the synthetic test fixtures do, and silently rasterizing a
    1-pixel line as a "region" would noise up downstream stats.
    """
    h, w = shape
    if h <= 0 or w <= 0:
        return np.zeros((max(h, 0), max(w, 0)), dtype=bool)
    points = [(int(x), int(y)) for x, y in quad]
    distinct = list(dict.fromkeys(points))  # preserve order, dedup
    if len(distinct) < 3:
        return np.zeros((h, w), dtype=bool)
    img = Image.new("L", (w, h), 0)
    ImageDraw.Draw(img).polygon(points, outline=1, fill=1)
    return np.asarray(img, dtype=np.uint8).astype(bool)


def detections_to_regions(
    detections: list[dict],
    image_id: str,
    shape: tuple[int, int],
) -> list[dict]:
    """Convert a list of EasyOCR detections into §7.2 region row dicts.

    Each detection should have at least `quad`, `text`, `confidence` keys.
    Returns rows that include the rasterized `mask` (a NumPy bool array)
    so callers can save it via the shared `_save_mask_npz` helper without
    re-rasterizing. Rows whose mask has zero area (degenerate quads,
    out-of-bounds points) are filtered out.
    """
    h, w = shape
    img_area = float(h * w) if h * w > 0 else 1.0
    out: list[dict] = []
    for i, det in enumerate(detections):
        mask = quad_to_mask(det.get("quad", []), shape)
        if mask.sum() == 0:
            continue
        ys, xs = np.where(mask)
        x1, y1 = int(xs.min()), int(ys.min())
        x2, y2 = int(xs.max()) + 1, int(ys.max()) + 1
        area_px = int(mask.sum())
        out.append({
            "image_id": image_id,
            "region_id": f"{image_id}__ocr_{i:03d}",
            "source": "ocr",
            "label": "ocr_text",
            "bbox_x1": x1,
            "bbox_y1": y1,
            "bbox_x2": x2,
            "bbox_y2": y2,
            "area_px": area_px,
            "area_frac": area_px / img_area,
            "ocr_text": det.get("text"),
            "object_confidence": float(det.get("confidence", 0.0)),
            "mask": mask,
            # Schema-completeness padding — these are populated by §9.5
            # labeling later in the pipeline:
            "stability_score": None,
            "predicted_iou": None,
            "parent_region_id": None,
            "dedup_group": None,
        })
    return out
