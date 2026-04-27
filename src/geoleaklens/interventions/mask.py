"""§10.3 mean_mask intervention.

Replace pixels inside a region mask with either:
  - the region's own per-channel mean color (local; default per
    `interventions.mean_mask_use_local_color: true`)
  - the whole-image per-channel mean color (global)

We dilate the mask by `dilation_px` (default 7, per §10.2 / `interventions.mask_dilation_px`)
before fill so that the intervention covers the soft boundary of the region —
otherwise a one-pixel-thin halo of original pixels survives around every edit
and the model can still latch onto edge cues.

Pure NumPy / Pillow. No GPU, no OpenCV — keeps this module testable on the
local laptop without the `seg` extras installed.
"""
from __future__ import annotations

from typing import Literal

import numpy as np
from PIL import Image

MeanMode = Literal["local", "global"]


def dilate_mask(mask: np.ndarray, px: int) -> np.ndarray:
    """Binary dilation by a square structuring element of radius `px`.

    `mask` is a 2-D bool/0-1 array. Returns a bool array of the same shape.

    A square SE (Chebyshev ball) is good enough for redaction halos; we don't
    need disk-shaped dilation. Implemented as iterated 1-pixel shifts so the
    module stays NumPy-only — `scipy.ndimage` is not a hard dep here.
    """
    if px <= 0:
        return mask.astype(bool, copy=True)
    out = mask.astype(bool, copy=True)
    for _ in range(int(px)):
        shifted = out.copy()
        shifted[1:, :] |= out[:-1, :]
        shifted[:-1, :] |= out[1:, :]
        shifted[:, 1:] |= out[:, :-1]
        shifted[:, :-1] |= out[:, 1:]
        out = shifted
    return out


def _channel_mean(img_rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-channel mean of `img_rgb` over the True pixels of `mask`.

    Falls back to the whole-image mean when `mask` is empty so callers don't
    have to special-case it.
    """
    if mask.any():
        sel = img_rgb[mask]  # (K, 3)
        return sel.mean(axis=0)
    return img_rgb.reshape(-1, 3).mean(axis=0)


def mean_mask(
    image: Image.Image,
    mask: np.ndarray,
    *,
    mode: MeanMode = "local",
    dilation_px: int = 7,
) -> tuple[Image.Image, np.ndarray]:
    """Apply mean-color fill inside the (dilated) region mask.

    Args:
        image: RGB Pillow image. Converted to RGB if not already.
        mask: 2-D bool/0-1 array, shape (H, W) matching the image.
        mode: "local" replaces with region mean; "global" with image mean.
        dilation_px: Pixels of binary dilation applied to `mask` before fill.
            §10.2 / `interventions.mask_dilation_px` default = 7.

    Returns:
        (edited_image, applied_mask) — `applied_mask` is the post-dilation bool
        mask actually painted, useful for area_frac bookkeeping in §11 scoring.
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
        # Nothing to paint — return a copy so callers can mutate freely.
        return Image.fromarray(arr.copy(), mode="RGB"), applied

    if mode == "local":
        fill = _channel_mean(arr, applied)
    elif mode == "global":
        fill = arr.reshape(-1, 3).mean(axis=0)
    else:
        raise ValueError(f"unknown mode: {mode!r}")

    fill_uint8 = np.clip(np.round(fill), 0, 255).astype(np.uint8)
    out = arr.copy()
    out[applied] = fill_uint8
    return Image.fromarray(out, mode="RGB"), applied
