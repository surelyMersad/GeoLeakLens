from pathlib import Path

import pytest
from PIL import Image

from geoleaklens.data.image_io import sha256_file
from geoleaklens.data.strip_exif import strip_and_resize


def _make_jpeg_with_exif(path: Path, size=(2048, 1024)) -> None:
    img = Image.new("RGB", size, color=(120, 80, 40))
    exif = img.getexif()
    # 0x010F = Make, 0x010E = ImageDescription. Both well-known EXIF tags.
    exif[0x010F] = "TestCamera"
    exif[0x010E] = "ImageDescription with city=SecretCity"
    img.save(path, format="JPEG", exif=exif.tobytes(), quality=92)


def test_strip_exif_removes_metadata(tmp_path):
    src = tmp_path / "src.jpg"
    dst = tmp_path / "out" / "stripped.jpg"
    _make_jpeg_with_exif(src)

    # Sanity: original carries EXIF.
    with Image.open(src) as orig:
        assert len(orig.getexif()) > 0

    strip_and_resize(src, dst, max_side=512)

    assert dst.exists()
    with Image.open(dst) as out:
        # No EXIF tags should survive.
        assert len(out.getexif()) == 0
        # Longest side resized to <= max_side, aspect preserved.
        w, h = out.size
        assert max(w, h) <= 512
        assert abs((w / h) - (2048 / 1024)) < 0.02


def test_strip_exif_no_resize_if_smaller(tmp_path):
    src = tmp_path / "small.jpg"
    dst = tmp_path / "out.jpg"
    Image.new("RGB", (400, 300), color=(10, 10, 10)).save(src, format="JPEG")

    strip_and_resize(src, dst, max_side=1024)

    with Image.open(dst) as out:
        assert out.size == (400, 300)


def test_strip_exif_preserves_visual_content(tmp_path):
    """Pixel content should survive (allowing for JPEG re-encode)."""
    src = tmp_path / "src.jpg"
    dst = tmp_path / "out.jpg"
    Image.new("RGB", (256, 256), color=(200, 100, 50)).save(src, format="JPEG", quality=95)

    strip_and_resize(src, dst, max_side=1024)

    with Image.open(dst) as out:
        out_rgb = out.convert("RGB")
        r, g, b = out_rgb.getpixel((128, 128))
        # JPEG re-encode introduces small color shifts; allow ±3 per channel.
        assert abs(r - 200) <= 3
        assert abs(g - 100) <= 3
        assert abs(b - 50) <= 3


def test_strip_exif_changes_sha256_but_not_dimensions_for_clean_input(tmp_path):
    """Re-encoding produces a new file (different sha256) even if no EXIF was present."""
    src = tmp_path / "clean.jpg"
    dst = tmp_path / "clean_out.jpg"
    Image.new("RGB", (300, 200), color=(50, 50, 50)).save(src, format="JPEG", quality=95)

    src_sha = sha256_file(src)
    strip_and_resize(src, dst, max_side=1024)
    dst_sha = sha256_file(dst)

    # Re-encoded file is byte-different but should have same dimensions.
    assert src_sha != dst_sha
    with Image.open(dst) as out:
        assert out.size == (300, 200)
