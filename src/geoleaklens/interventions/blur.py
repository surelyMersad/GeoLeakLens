"""§10.2 Gaussian blur intervention.

Replace pixels inside a region mask with a Gaussian-blurred version of
themselves. Mask is binary-dilated by `dilation_px` (default 7 per
`interventions.mask_dilation_px`) before the blend so the soft boundary
of the region is fully covered — otherwise a 1-pixel halo of original
pixels survives along the perimeter and the model can latch onto edge
cues.

Why blur is on the §10.7 menu (not just mean_mask)
---------------------------------------------------
Per §10.4, mean_mask, blur, and inpaint each have characteristic failure
modes:
  - mean_mask leaves a flat-color rectangle that GeoCLIP can detect as
    "this looks redacted" rather than missing the original content.
  - blur preserves texture but loses identity; the model sees "something
    is here, but I can't read it."
  - inpaint hallucinates plausible content; the model's response then
    depends on what the inpainter produced.

For the §13.E2 sparsity headline (`min_across_interventions`, §10.7)
we need all three scores so a region only counts as leaky if every
intervention agrees.

Pure NumPy/Pillow. No OpenCV needed locally.
"""
from __future__ import annotations

from typing import Tuple

import numpy as np
from PIL import Image, ImageFilter

from .mask import dilate_mask


def blur(
    image: Image.Image,
    mask: np.ndarray,
    *,
    blur_sigma_frac: float = 0.035,
    dilation_px: int = 7,
) -> Tuple[Image.Image, np.ndarray]:
    """Blur the masked region with Gaussian sigma scaled to image size.

    Args:
        image: RGB Pillow image.
        mask: 2-D bool/0-1 array, shape (H, W).
        blur_sigma_frac: §10.2 — sigma = blur_sigma_frac * max(W, H). 0.035
            is the spec default.
        dilation_px: pixels of binary dilation applied before blending.

    Returns:
        (edited, applied_mask). `applied_mask` is the post-dilation bool
        mask actually painted, useful for area_frac bookkeeping.
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
        return Image.fromarray(arr.copy(), mode="RGB"), applied

    sigma = float(blur_sigma_frac) * float(max(w, h))
    if sigma <= 0:
        # Degenerate — no blur means no intervention. Return a copy.
        return Image.fromarray(arr.copy(), mode="RGB"), applied

    blurred_full = np.asarray(
        image.filter(ImageFilter.GaussianBlur(radius=sigma)), dtype=np.uint8
    )
    out = arr.copy()
    out[applied] = blurred_full[applied]
    return Image.fromarray(out, mode="RGB"), applied
