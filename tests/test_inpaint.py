"""Unit tests for the local inpaint wrapper. Real Modal call is mocked."""
import io
from dataclasses import dataclass
from typing import Callable

import numpy as np
import pytest
from PIL import Image

from geoleaklens.interventions.inpaint import inpaint


@dataclass
class _FakeRemote:
    """Stand-in for `LaMaModal.inpaint` — has `.remote(image_bytes, mask_bytes)`."""

    fn: Callable[[bytes, bytes], bytes]
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []

    def remote(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        self.calls.append((image_bytes, mask_bytes))
        return self.fn(image_bytes, mask_bytes)


@dataclass
class _FakeLaMaModal:
    """Match the LaMaModal interface: `.inpaint.remote(...)`."""

    inpaint: _FakeRemote


def _solid_jpeg_bytes(rgb: tuple[int, int, int], h: int = 40, w: int = 40) -> bytes:
    img = Image.fromarray(np.full((h, w, 3), rgb, dtype=np.uint8), mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=95)
    return buf.getvalue()


def test_inpaint_calls_modal_with_correct_size_image():
    """Wrapper sends image + mask bytes to LaMa; result is decoded back."""
    h, w = 40, 40
    expected_out = _solid_jpeg_bytes((50, 100, 150), h=h, w=w)

    captured = {}

    def fake_inpaint(image_bytes, mask_bytes):
        # Decode locally to verify roundtrip shapes.
        img = Image.open(io.BytesIO(image_bytes))
        mask = Image.open(io.BytesIO(mask_bytes))
        captured["img_size"] = img.size
        captured["mask_size"] = mask.size
        captured["mask_mode"] = mask.mode
        return expected_out

    lama = _FakeLaMaModal(inpaint=_FakeRemote(fn=fake_inpaint))

    in_img = Image.fromarray(np.zeros((h, w, 3), dtype=np.uint8), mode="RGB")
    mask = np.zeros((h, w), dtype=bool)
    mask[10:30, 10:30] = True

    out, applied = inpaint(in_img, mask, lama_modal=lama, dilation_px=0)
    assert out.size == in_img.size
    # Exactly one Modal call.
    assert len(lama.inpaint.calls) == 1
    # Mask sent over the wire is single-channel and same size as image.
    assert captured["mask_mode"] == "L"
    assert captured["mask_size"] == (w, h)
    assert captured["img_size"] == (w, h)
    # Output is the bytes we returned, decoded back to RGB.
    assert np.asarray(out).shape == (h, w, 3)
    # Applied mask is the post-dilation mask actually painted.
    assert applied.sum() == 20 * 20  # dilation_px=0


def test_inpaint_dilation_grows_mask_before_send():
    captured_mask = {}

    def fake_inpaint(image_bytes, mask_bytes):
        m = np.asarray(Image.open(io.BytesIO(mask_bytes)).convert("L"))
        captured_mask["nonzero"] = int((m > 0).sum())
        return _solid_jpeg_bytes((0, 0, 0), h=m.shape[0], w=m.shape[1])

    lama = _FakeLaMaModal(inpaint=_FakeRemote(fn=fake_inpaint))
    img = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8), mode="RGB")
    mask = np.zeros((40, 40), dtype=bool)
    mask[20, 20] = True  # single pixel

    _, applied = inpaint(img, mask, lama_modal=lama, dilation_px=2)
    # 4-connected dilation by 2 → 13-pixel diamond.
    assert applied.sum() == 13
    assert captured_mask["nonzero"] == 13


def test_inpaint_empty_mask_skips_modal_call():
    """No Modal call when nothing to inpaint."""
    lama = _FakeLaMaModal(
        inpaint=_FakeRemote(fn=lambda *_: pytest.fail("modal called on empty mask"))
    )
    img = Image.fromarray(np.full((20, 20, 3), 50, dtype=np.uint8), mode="RGB")
    mask = np.zeros((20, 20), dtype=bool)
    out, applied = inpaint(img, mask, lama_modal=lama, dilation_px=3)
    assert applied.sum() == 0
    # Output is a copy of the input.
    assert np.array_equal(np.asarray(out), np.asarray(img))
    assert len(lama.inpaint.calls) == 0


def test_inpaint_rejects_size_mismatch_from_lama():
    """If LaMa returns a different-sized image, raise (§10.4 invariant)."""
    def fake_inpaint(image_bytes, mask_bytes):
        # Return a JPEG with WRONG dimensions.
        return _solid_jpeg_bytes((0, 0, 0), h=20, w=20)

    lama = _FakeLaMaModal(inpaint=_FakeRemote(fn=fake_inpaint))
    img = Image.fromarray(np.zeros((40, 40, 3), dtype=np.uint8), mode="RGB")
    mask = np.ones((40, 40), dtype=bool)
    with pytest.raises(ValueError, match="dimensions changed"):
        inpaint(img, mask, lama_modal=lama, dilation_px=0)


def test_inpaint_shape_mismatch_raises():
    lama = _FakeLaMaModal(
        inpaint=_FakeRemote(fn=lambda *_: pytest.fail("should not be called"))
    )
    img = Image.fromarray(np.zeros((10, 10, 3), dtype=np.uint8), mode="RGB")
    bad = np.zeros((5, 5), dtype=bool)
    with pytest.raises(ValueError, match="does not match"):
        inpaint(img, bad, lama_modal=lama)


def test_inpaint_converts_non_rgb_input():
    arr = np.full((20, 20), 128, dtype=np.uint8)
    img = Image.fromarray(arr, mode="L")  # grayscale
    mask = np.ones((20, 20), dtype=bool)
    lama = _FakeLaMaModal(
        inpaint=_FakeRemote(fn=lambda *_: _solid_jpeg_bytes((0, 0, 0), h=20, w=20))
    )
    out, _ = inpaint(img, mask, lama_modal=lama, dilation_px=0)
    assert out.mode == "RGB"
