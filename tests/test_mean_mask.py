import numpy as np
import pytest
from PIL import Image

from geoleaklens.interventions.mask import dilate_mask, mean_mask


def _solid(rgb: tuple[int, int, int], h: int = 64, w: int = 64) -> Image.Image:
    return Image.fromarray(np.full((h, w, 3), rgb, dtype=np.uint8), mode="RGB")


def test_dilate_zero_is_noop():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    out = dilate_mask(m, 0)
    assert np.array_equal(out, m)


def test_dilate_one_pixel():
    m = np.zeros((5, 5), dtype=bool)
    m[2, 2] = True
    out = dilate_mask(m, 1)
    expected = np.zeros((5, 5), dtype=bool)
    expected[1:4, 2] = True
    expected[2, 1:4] = True
    assert np.array_equal(out, expected)


def test_dilate_grows_monotonically():
    m = np.zeros((11, 11), dtype=bool)
    m[5, 5] = True
    sizes = [int(dilate_mask(m, k).sum()) for k in range(0, 4)]
    assert sizes == sorted(sizes)
    assert sizes[0] == 1
    assert sizes[-1] > sizes[0]


def test_mean_mask_global_replaces_with_image_mean():
    # Half-red, half-blue image. Mask covers the red half. Global mean is purple.
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:, :20] = (200, 0, 0)
    arr[:, 20:] = (0, 0, 200)
    img = Image.fromarray(arr, mode="RGB")

    mask = np.zeros((40, 40), dtype=bool)
    mask[:, :20] = True

    out, applied = mean_mask(img, mask, mode="global", dilation_px=0)
    out_arr = np.asarray(out)

    # Outside mask: untouched blue.
    assert np.all(out_arr[:, 20:] == (0, 0, 200))
    # Inside mask: per-channel image mean = (100, 0, 100).
    assert np.all(out_arr[:, :20] == (100, 0, 100))
    assert applied.sum() == 40 * 20


def test_mean_mask_local_uses_region_mean():
    # Image: red + blue halves. Mask spans only the red half.
    # Local mean over the masked (red) region is (200, 0, 0) -> red stays red.
    arr = np.zeros((40, 40, 3), dtype=np.uint8)
    arr[:, :20] = (200, 0, 0)
    arr[:, 20:] = (0, 0, 200)
    img = Image.fromarray(arr, mode="RGB")

    mask = np.zeros((40, 40), dtype=bool)
    mask[:, :20] = True

    out, _ = mean_mask(img, mask, mode="local", dilation_px=0)
    out_arr = np.asarray(out)
    assert np.all(out_arr[:, :20] == (200, 0, 0))
    assert np.all(out_arr[:, 20:] == (0, 0, 200))


def test_mean_mask_dilation_expands_painted_region():
    img = _solid((10, 20, 30), h=40, w=40)
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True

    out, applied = mean_mask(img, mask, mode="local", dilation_px=2)
    # Dilating a single pixel by 2 in 4-connected steps gives a diamond of 13 px.
    assert applied.sum() == 13
    out_arr = np.asarray(out)
    # All applied pixels share the same fill color (region mean of a constant
    # image is itself).
    assert np.all(out_arr[applied] == (10, 20, 30))


def test_mean_mask_empty_mask_returns_unchanged():
    img = _solid((50, 50, 50), h=20, w=20)
    mask = np.zeros((20, 20), dtype=bool)
    out, applied = mean_mask(img, mask, mode="local", dilation_px=3)
    assert applied.sum() == 0
    assert np.array_equal(np.asarray(out), np.asarray(img))


def test_mean_mask_shape_mismatch_raises():
    img = _solid((0, 0, 0), h=10, w=10)
    bad = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError):
        mean_mask(img, bad)


def test_mean_mask_converts_non_rgb_input():
    # Grayscale input should be converted to RGB without raising.
    arr = np.full((20, 20), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")
    mask = np.ones((20, 20), dtype=bool)
    out, _ = mean_mask(img, mask, mode="global", dilation_px=0)
    assert out.mode == "RGB"
    assert np.asarray(out).shape == (20, 20, 3)
