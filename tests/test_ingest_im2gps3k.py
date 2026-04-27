import io
import struct

import pytest
from PIL import Image

from geoleaklens.data.ingest_im2gps3k import (
    _walk_com_markers,
    parse_im2gps3k_metadata,
)


def _build_jpeg_with_coms(coms: list[bytes]) -> bytes:
    """Render a minimal valid JPEG with the given COM payloads injected.

    Pillow doesn't expose multi-COM writes, so we build a tiny JPEG and splice
    the COM segments into the byte stream right after SOI.
    """
    img = Image.new("RGB", (4, 4), color=(0, 0, 0))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    base = buf.getvalue()
    # base starts with FFD8 (SOI). Splice COMs immediately after.
    head = base[:2]
    tail = base[2:]
    com_blob = b""
    for payload in coms:
        seg_len = len(payload) + 2  # length field includes itself
        com_blob += b"\xff\xfe" + struct.pack(">H", seg_len) + payload
    return head + com_blob + tail


def test_walk_com_markers_returns_each_payload_in_order():
    payloads = [b"photo: 1234 abcd 1000", b"latitude: 40.5", b"longitude: -74.0"]
    jpeg = _build_jpeg_with_coms(payloads)
    out = list(_walk_com_markers(jpeg))
    assert out == payloads


def test_walk_com_markers_skips_segment_after_sos():
    """COMs after the SOS marker must NOT be returned (they don't exist in
    valid JPEGs; this guards the parser from reading entropy-coded data)."""
    payloads = [b"latitude: 1.0"]
    jpeg = _build_jpeg_with_coms(payloads)
    # SOS marker (FFDA) comes before the entropy-coded image data; no COMs
    # should be parsed from after that point even if a stray FFFE appeared.
    out = list(_walk_com_markers(jpeg))
    assert out == [b"latitude: 1.0"]


def test_parse_basic_lat_lon(tmp_path):
    jpeg = _build_jpeg_with_coms(
        [
            b"photo: 1000269685 e60e9cdfb4 1125",
            b"owner: 78841376@N00",
            b"latitude: 32.325436",
            b"longitude: -64.764404",
            b"accuracy: 8",
        ]
    )
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg)
    meta = parse_im2gps3k_metadata(p)
    assert meta["photo_id"] == "1000269685"
    assert meta["owner"] == "78841376@N00"
    assert meta["lat"] == pytest.approx(32.325436)
    assert meta["lon"] == pytest.approx(-64.764404)
    assert meta["accuracy"] == 8


def test_parse_owner_strips_trailing_junk(tmp_path):
    """Real Im2GPS3k owner payload has buffer-junk concatenated with a period
    (e.g. 'owner: 78841376@N00.0e9cdfb4 1125.rd & '). The parser must extract
    only the valid NSID."""
    jpeg = _build_jpeg_with_coms(
        [b"owner: 78841376@N00.0e9cdfb4 1125.rd & "]
    )
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg)
    meta = parse_im2gps3k_metadata(p)
    assert meta["owner"] == "78841376@N00"


def test_parse_handles_trailing_junk_in_com_payload(tmp_path):
    """Real Im2GPS3k images have buffer-junk after the field value; parser
    must extract just the numeric prefix."""
    jpeg = _build_jpeg_with_coms(
        [
            b"latitude: 32.325436.2 21:41:18.25.rd & ",
            b"longitude: -64.764404.21:41:18.25.rd & ",
        ]
    )
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg)
    meta = parse_im2gps3k_metadata(p)
    assert meta["lat"] == pytest.approx(32.325436)
    assert meta["lon"] == pytest.approx(-64.764404)


def test_parse_negative_coords(tmp_path):
    jpeg = _build_jpeg_with_coms(
        [b"latitude: -33.8688", b"longitude: -179.9999"]
    )
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg)
    meta = parse_im2gps3k_metadata(p)
    assert meta["lat"] == pytest.approx(-33.8688)
    assert meta["lon"] == pytest.approx(-179.9999)


def test_parse_missing_gps_returns_none(tmp_path):
    jpeg = _build_jpeg_with_coms(
        [b"photo: 9999 ssss 1000", b"tags: nogps"]
    )
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg)
    meta = parse_im2gps3k_metadata(p)
    assert meta["lat"] is None
    assert meta["lon"] is None
    assert meta["photo_id"] == "9999"


def test_parse_invalid_lat_value_returns_none(tmp_path):
    jpeg = _build_jpeg_with_coms([b"latitude: not_a_number_at_all"])
    p = tmp_path / "x.jpg"
    p.write_bytes(jpeg)
    meta = parse_im2gps3k_metadata(p)
    assert meta["lat"] is None


def test_parse_real_im2gps3k_sample_if_present():
    """If the user has actually downloaded Im2GPS3k, sanity-check one image.

    Skipped silently when the data isn't present (tests must work in CI without
    the 455 MB dataset).
    """
    from pathlib import Path

    candidate = Path("data/raw/im2gps3k/im2gps3ktest")
    if not candidate.exists():
        pytest.skip("Im2GPS3k not downloaded — skipping real-sample check")
    jpgs = list(candidate.glob("*.jpg"))
    if not jpgs:
        pytest.skip("Im2GPS3k extracted dir empty")
    meta = parse_im2gps3k_metadata(jpgs[0])
    assert meta["lat"] is not None and -90 <= meta["lat"] <= 90
    assert meta["lon"] is not None and -180 <= meta["lon"] <= 180
