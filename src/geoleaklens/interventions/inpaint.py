"""§10.4 inpaint intervention — local wrapper around the LaMa Modal cls.

Why this lives in `interventions/` and not `models/`
-----------------------------------------------------
`mean_mask` and `blur` are pure NumPy/Pillow functions — no Modal needed.
Inpaint is GPU-bound, so the heavy lifting runs in `models/inpaint_modal`.
This local wrapper preserves the same signature shape as the other two
interventions:

    edited_image, applied_mask = intervention(image, mask, ...)

so the §13.E2 / §10.7 sweep can iterate over all three intervention types
with one call site. The only oddity: `inpaint(...)` requires a live
`LaMaModal` reference, which the orchestrator provides while inside
`with app.run():`.

§10.4 spec rules baked in here:
- Mask is binary-dilated by `dilation_px` (default 7) before LaMa sees it.
- Output dimensions verified equal to input.
- An `artifact_flag` is *not* yet emitted on failure — exceptions propagate.
  TODO when E6 ablation lands: catch, log, and return the flag per §10.4.
- EXIF stripping is a non-issue here because we re-encode through Pillow.
"""
from __future__ import annotations

import io
from typing import Any, Tuple

import numpy as np
from PIL import Image

from .mask import dilate_mask


def inpaint(
    image: Image.Image,
    mask: np.ndarray,
    *,
    lama_modal: Any,  # geoleaklens.models.inpaint_modal.LaMaModal instance
    dilation_px: int = 7,
    jpeg_quality: int = 92,
) -> Tuple[Image.Image, np.ndarray]:
    """Inpaint the masked region using a live LaMa Modal cls.

    Args:
        image: RGB Pillow image.
        mask: 2-D bool/0-1 array, shape (H, W) matching the image.
        lama_modal: A `LaMaModal()` instance obtained inside `with app.run():`.
        dilation_px: §10.2 / §10.4 — pixels of binary dilation before LaMa
            sees the mask. Default 7 matches `interventions.mask_dilation_px`.
        jpeg_quality: Quality used to encode the input for transport. Higher
            = larger payload but less compression artifact bleed-through.

    Returns:
        (edited_image, applied_mask). `applied_mask` is the post-dilation
        bool mask actually painted, useful for area_frac bookkeeping.

    Raises:
        ValueError: if mask shape differs from image shape, or if LaMa
            returns a different-size image (§10.4 dimensions invariant).
    """
    if image.mode != "RGB":
        image = image.convert("RGB")
    arr = np.asarray(image, dtype=np.uint8)
    h, w = arr.shape[:2]
    if mask.shape != (h, w):
        raise ValueError(
            f"mask shape {mask.shape} does not match image shape {(h, w)}"
        )

    applied = dilate_mask(mask.astype(bool), dilation_px)
    if not applied.any():
        # No-op: nothing to inpaint. Return a copy.
        return Image.fromarray(arr.copy(), mode="RGB"), applied

    # Encode for transport.
    img_buf = io.BytesIO()
    image.save(img_buf, format="JPEG", quality=int(jpeg_quality))

    mask_pil = Image.fromarray((applied.astype(np.uint8) * 255), mode="L")
    mask_buf = io.BytesIO()
    mask_pil.save(mask_buf, format="PNG")

    inpainted_bytes = lama_modal.inpaint.remote(
        img_buf.getvalue(), mask_buf.getvalue()
    )

    edited = Image.open(io.BytesIO(inpainted_bytes)).convert("RGB")
    if edited.size != image.size:
        raise ValueError(
            f"inpaint dimensions changed: input {image.size}, output {edited.size}"
        )
    return edited, applied
