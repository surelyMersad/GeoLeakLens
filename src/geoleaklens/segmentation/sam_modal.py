"""§9.2 SAM v1 automatic mask generation as a Modal class.

Why a separate module from `modal_app.py`
-----------------------------------------
GeoCLIP and SAM live on the same `modal.App` (so a single `with app.run():`
brings up both), but their images, volumes, and lifetimes diverge. Splitting
SAM into its own file keeps `modal_app.py` from becoming a junk drawer as we
add LaMa, CLIP-utility, and VLM wrappers.

Importing this module is enough to register `SAMModal` against the shared
`app` — Modal's app object is mutated by `@app.cls(...)` at import time.

What it returns
---------------
SAM's `SamAutomaticMaskGenerator` produces dense bool masks of shape (H, W).
Sending 80 of those raw across the Modal boundary is ~80 MB per 1024² image.
We pack them once per image as a single (N, H, W) bool tensor → `np.packbits`
→ `zlib` → base64. Local-side decode is the inverse. See `decode_masks()`.

Filtering, dedup, cap (§9.2 + §9.6)
-----------------------------------
1. Drop masks whose `area_frac` is outside `[min_area_frac, max_area_frac]`.
2. Sort remaining by `predicted_iou` desc.
3. Greedy-dedup at IoU > `dedup_iou_threshold` (skip a mask if it overlaps a
   kept one above that threshold).
4. Cap to `max_regions`.

The returned masks are **not** dilated — `interventions/mask.py` applies its
own dilation per `interventions.mask_dilation_px` (§10.2). Storing raw masks
keeps the segmentation step independent of intervention parameters.
"""
from __future__ import annotations

import base64
import io
import zlib
from typing import Any

import modal

from geoleaklens.modal_app import (
    SAM_CKPT_MOUNT,
    app,
    sam_checkpoints,
    sam_image,
)

# Public Meta SAM v1 ViT-B checkpoint, ~375 MB. Matches the size called out in
# `modal_app.py`'s docstring for the `geoleaklens-sam-checkpoints` volume.
_SAM_VIT_B_URL = (
    "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth"
)
_SAM_VIT_B_FILENAME = "sam_vit_b_01ec64.pth"
_SAM_MODEL_TYPE = "vit_b"


def _encode_masks(masks_nhw_bool) -> str:
    """packbits → zlib → base64 over a (N, H, W) bool array. Empty-safe."""
    import numpy as np

    arr = np.asarray(masks_nhw_bool, dtype=bool)
    if arr.size == 0:
        return ""
    packed = np.packbits(arr.reshape(-1).astype(np.uint8))
    return base64.b64encode(zlib.compress(packed.tobytes(), level=6)).decode("ascii")


def decode_masks(b64: str, n: int, h: int, w: int):
    """Inverse of `_encode_masks`. Returns a (N, H, W) bool ndarray.

    Local-side helper (not Modal-only) — this is what the orchestrator calls
    after a `generate_masks.remote(...)` round-trip.
    """
    import numpy as np

    if n == 0 or not b64:
        return np.zeros((0, h, w), dtype=bool)
    raw = zlib.decompress(base64.b64decode(b64.encode("ascii")))
    packed = np.frombuffer(raw, dtype=np.uint8)
    bits = np.unpackbits(packed)[: n * h * w]
    return bits.astype(bool).reshape(n, h, w)


@app.cls(
    image=sam_image,
    gpu="L4",
    timeout=900,
    volumes={SAM_CKPT_MOUNT: sam_checkpoints},
)
class SAMModal:
    """Loads SAM v1 (ViT-B) once per container and serves automatic masks.

    The checkpoint is fetched into the persistent `geoleaklens-sam-checkpoints`
    volume on first cold-start; subsequent containers find it there and skip
    the download.
    """

    @modal.enter()
    def load(self) -> None:
        import os
        import urllib.request

        import torch
        from segment_anything import SamAutomaticMaskGenerator, sam_model_registry

        ckpt_path = os.path.join(SAM_CKPT_MOUNT, _SAM_VIT_B_FILENAME)
        if not os.path.exists(ckpt_path):
            os.makedirs(SAM_CKPT_MOUNT, exist_ok=True)
            urllib.request.urlretrieve(_SAM_VIT_B_URL, ckpt_path)
            sam_checkpoints.commit()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        sam = sam_model_registry[_SAM_MODEL_TYPE](checkpoint=ckpt_path)
        sam.to(self.device)
        sam.eval()

        # Defaults match the SAM paper's automatic-mask-generator settings;
        # filtering/dedup happens after the call so we don't bake §9.2 thresholds
        # into model state.
        self.mask_generator = SamAutomaticMaskGenerator(model=sam)

    @modal.method()
    def generate_masks(
        self,
        image_bytes: bytes,
        *,
        min_area_frac: float = 0.001,
        max_area_frac: float = 0.45,
        dedup_iou_threshold: float = 0.85,
        max_regions: int = 80,
    ) -> dict[str, Any]:
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        img_np = np.array(img)
        h, w = img_np.shape[:2]
        image_area = float(h * w)

        raw_masks = self.mask_generator.generate(img_np)
        # Each entry: {"segmentation": HxW bool, "area": int, "bbox": [x,y,w,h],
        #              "predicted_iou": float, "stability_score": float, ...}

        # 1. Area filter.
        kept: list[dict] = []
        for m in raw_masks:
            af = float(m["area"]) / image_area
            if af < min_area_frac or af > max_area_frac:
                continue
            m["_area_frac"] = af
            kept.append(m)

        # 2. Sort by predicted_iou desc — best-quality masks get to claim
        #    territory first under greedy dedup.
        kept.sort(key=lambda m: float(m.get("predicted_iou", 0.0)), reverse=True)

        # 3. Greedy IoU dedup. Brute-force is fine for N≤200, H*W≤1.5M (smoke).
        deduped: list[dict] = []
        deduped_segs: list[np.ndarray] = []
        for m in kept:
            seg = m["segmentation"].astype(bool)
            seg_sum = int(seg.sum())
            drop = False
            for k in deduped_segs:
                inter = int(np.logical_and(seg, k).sum())
                union = seg_sum + int(k.sum()) - inter
                if union == 0:
                    continue
                if inter / union > dedup_iou_threshold:
                    drop = True
                    break
            if not drop:
                deduped.append(m)
                deduped_segs.append(seg)

        # 4. Cap.
        deduped = deduped[: int(max_regions)]
        deduped_segs = deduped_segs[: int(max_regions)]

        if deduped_segs:
            stack = np.stack(deduped_segs, axis=0)  # (N, H, W) bool
        else:
            stack = np.zeros((0, h, w), dtype=bool)

        masks_meta = []
        for m in deduped:
            x, y, bw, bh = (int(v) for v in m["bbox"])
            masks_meta.append(
                {
                    "area_px": int(m["area"]),
                    "area_frac": float(m["_area_frac"]),
                    "bbox_x1": x,
                    "bbox_y1": y,
                    "bbox_x2": x + bw,
                    "bbox_y2": y + bh,
                    "predicted_iou": float(m.get("predicted_iou", 0.0)),
                    "stability_score": float(m.get("stability_score", 0.0)),
                }
            )

        return {
            "image_height": int(h),
            "image_width": int(w),
            "n_masks": int(stack.shape[0]),
            "masks_packed_b64": _encode_masks(stack),
            "masks_meta": masks_meta,
        }
