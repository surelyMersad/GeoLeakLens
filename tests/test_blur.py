import numpy as np
import pytest
from PIL import Image

from geoleaklens.interventions.blur import blur


def test_blur_changes_only_inside_mask():
    """Outside the mask, pixels must be exactly preserved."""
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:, :20] = (200, 0, 0)
    arr[:, 20:] = (0, 0, 200)
    img = Image.fromarray(arr, mode="RGB")
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True

    out, applied = blur(img, mask, blur_sigma_frac=0.05, dilation_px=0)
    out_arr = np.asarray(out)

    # Every pixel where the mask is False must equal the original.
    assert np.array_equal(out_arr[~applied], arr[~applied])


def test_blur_actually_blurs_inside_mask():
    """A high-contrast checkerboard inside the mask should lose contrast."""
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[::2, :, :] = 255  # alternating black/white rows
    img = Image.fromarray(arr, mode="RGB")
    mask = np.ones((40, 40), dtype=bool)

    out, _ = blur(img, mask, blur_sigma_frac=0.1, dilation_px=0)
    out_arr = np.asarray(out)

    orig_std = arr.astype(np.float64).std()
    blurred_std = out_arr.astype(np.float64).std()
    assert blurred_std < orig_std * 0.5  # large drop in pixel-value spread


def test_blur_empty_mask_returns_unchanged():
    arr = np.full((20, 20, 3), 100, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    mask = np.zeros((20, 20), dtype=bool)
    out, applied = blur(img, mask, dilation_px=3)
    assert applied.sum() == 0
    assert np.array_equal(np.asarray(out), arr)


def test_blur_dilation_grows_painted_region():
    arr = np.full((40, 40, 3), 50, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True
    _, applied = blur(img, mask, blur_sigma_frac=0.05, dilation_px=2)
    assert applied.sum() == 13  # 4-connected diamond


def test_blur_shape_mismatch_raises():
    arr = np.zeros((10, 10, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    bad = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        blur(img, bad)


def test_blur_zero_sigma_is_noop_inside_mask():
    """sigma_frac=0 → no blur applied; pixels unchanged even inside the mask."""
    arr = np.zeros((30, 30, 3), dtype=np.uint8)
    arr[:, :15] = (150, 0, 0)
    arr[:, 15:] = (0, 150, 0)
    img = Image.fromarray(arr, mode="RGB")
    mask = np.ones((30, 30), dtype=bool)
    out, _ = blur(img, mask, blur_sigma_frac=0.0, dilation_px=0)
    assert np.array_equal(np.asarray(out), arr)
