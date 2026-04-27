"""Im2GPS3k ingest: parse JPEG COM markers for ground-truth GPS.

Background
----------
Im2GPS3k (Vo et al. ICCV 2017) ships ground-truth lat/lon as JPEG COM
markers — *one COM segment per Flickr metadata field*. PIL's
`Image.info['comment']` only surfaces the last COM, which loses the
`latitude:` / `longitude:` lines. We parse the markers ourselves.

A typical image has these segments (one COM each):
  photo: <flickr_id> <secret> <server>
  owner: <user_nsid>
  title: <title>
  originalsecret / originalformat
  datetaken: YYYY-MM-DD HH:MM:SS
  tags: <space-separated>
  license: <flickr license code>
  latitude: <decimal>
  longitude: <decimal>
  accuracy: <flickr GPS accuracy 1-16>
  interestingness: ...

The COM payloads have trailing junk left over from buffer reuse during
encoding, so we extract only the first valid token after `<field>:`.

Why not extend `build_manifest.py`
-----------------------------------
build_manifest.py reads ground truth from a sidecar CSV and EXIF GPS.
Im2GPS3k has neither — its ground truth lives only inside the JPEG byte
stream. Different ingest path, same §7.1 output schema.
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd

from .build_manifest import MANIFEST_COLUMNS, _stable_image_id
from .image_io import image_dims, sha256_file
from .strip_exif import _IMG_SUFFIXES, strip_and_resize


# A valid signed decimal at the start of the value, allowing optional sign and
# at least one digit before/after the decimal point.
_DECIMAL_RE = re.compile(rb"-?\d+\.\d+")
# Flickr accuracy is an integer 1-16.
_INTEGER_RE = re.compile(rb"\d+")
# Flickr NSID format: digits + "@N" + digits (e.g. "78841376@N00"). Match at the
# start so we stop before any buffer-junk that happens to follow with a period.
_NSID_RE = re.compile(rb"\d+@N\d+")


def _walk_com_markers(jpeg_bytes: bytes) -> Iterable[bytes]:
    """Yield each COM (0xFFFE) segment payload in order."""
    i = 0
    n = len(jpeg_bytes)
    while i < n - 1:
        if jpeg_bytes[i] != 0xFF:
            i += 1
            continue
        m = jpeg_bytes[i + 1]
        if m in (0x00, 0xFF):
            i += 1
            continue
        if m in (0xD8, 0xD9):  # SOI / EOI — no length field
            i += 2
            continue
        if m == 0xDA:  # SOS — entropy-coded data follows; no more COMs
            return
        if i + 4 > n:
            return
        seg_len = (jpeg_bytes[i + 2] << 8) | jpeg_bytes[i + 3]
        payload = jpeg_bytes[i + 4 : i + 2 + seg_len]
        if m == 0xFE:  # COM
            yield payload
        i += 2 + seg_len


def _extract_field_after(prefix: bytes, payload: bytes) -> Optional[bytes]:
    """Return bytes immediately after `prefix:` in `payload`, before any null
    or marker that looks like buffer-junk."""
    if not payload.startswith(prefix + b":"):
        return None
    return payload[len(prefix) + 1 :].lstrip()


def parse_im2gps3k_metadata(jpeg_path: Path) -> dict:
    """Return {photo_id, lat, lon, accuracy, datetaken, tags, license, owner}.

    Any field that fails to parse is None / empty.
    """
    with open(jpeg_path, "rb") as f:
        data = f.read()

    out: dict = {
        "photo_id": None,
        "owner": None,
        "lat": None,
        "lon": None,
        "accuracy": None,
        "datetaken": None,
        "tags": None,
        "license": None,
    }

    for payload in _walk_com_markers(data):
        if payload.startswith(b"photo:"):
            tail = _extract_field_after(b"photo", payload)
            if tail:
                # "photo: <id> <secret> <server>" — take first whitespace-token.
                out["photo_id"] = tail.split()[0].decode("latin-1", errors="replace")
        elif payload.startswith(b"owner:"):
            tail = _extract_field_after(b"owner", payload)
            if tail:
                m = _NSID_RE.match(tail)
                if m:
                    out["owner"] = m.group(0).decode("latin-1")
        elif payload.startswith(b"latitude:"):
            tail = _extract_field_after(b"latitude", payload)
            if tail:
                m = _DECIMAL_RE.match(tail)
                if m:
                    try:
                        out["lat"] = float(m.group(0))
                    except ValueError:
                        pass
        elif payload.startswith(b"longitude:"):
            tail = _extract_field_after(b"longitude", payload)
            if tail:
                m = _DECIMAL_RE.match(tail)
                if m:
                    try:
                        out["lon"] = float(m.group(0))
                    except ValueError:
                        pass
        elif payload.startswith(b"accuracy:"):
            tail = _extract_field_after(b"accuracy", payload)
            if tail:
                m = _INTEGER_RE.match(tail)
                if m:
                    try:
                        out["accuracy"] = int(m.group(0))
                    except ValueError:
                        pass
        elif payload.startswith(b"datetaken:"):
            tail = _extract_field_after(b"datetaken", payload)
            if tail:
                # Flickr "YYYY-MM-DD HH:MM:SS" is exactly 19 chars.
                out["datetaken"] = tail[:19].decode("latin-1", errors="replace").strip()
        elif payload.startswith(b"tags:"):
            tail = _extract_field_after(b"tags", payload)
            if tail:
                # Tags run until the next null or the trailing junk-marker.
                first_null = tail.find(b"\x00")
                end = first_null if first_null >= 0 else min(len(tail), 60)
                out["tags"] = tail[:end].decode("latin-1", errors="replace").strip()
        elif payload.startswith(b"license:"):
            tail = _extract_field_after(b"license", payload)
            if tail:
                m = _INTEGER_RE.match(tail)
                if m:
                    out["license"] = m.group(0).decode("latin-1")

    return out


def build_im2gps3k_manifest(
    raw_dir: Path,
    interim_root: Path,
    out_path: Path,
    *,
    dataset: str = "im2gps3k",
    split: str = "test",
    max_side: int = 1024,
    is_allowed_for_public_demo: bool = False,
) -> pd.DataFrame:
    """Walk raw_dir, parse COM-marker GPS per image, write §7.1 manifest.

    Images are also re-encoded without EXIF and resized into
    `interim_root/{dataset}/`, matching the build_manifest.py convention.
    Rows where lat or lon couldn't be parsed are dropped — that's
    incomplete ground truth, not useful for evaluation.
    """
    raw_dir = Path(raw_dir)
    interim_root = Path(interim_root)
    out_path = Path(out_path)

    rows: list[dict] = []
    skipped_no_gps = 0
    n_total = 0

    sources = sorted(p for p in raw_dir.rglob("*") if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES)
    for src in sources:
        n_total += 1
        meta = parse_im2gps3k_metadata(src)
        if meta["lat"] is None or meta["lon"] is None:
            skipped_no_gps += 1
            continue

        rel = src.relative_to(raw_dir)
        dst = interim_root / dataset / rel
        strip_and_resize(src, dst, max_side=max_side)

        sha = sha256_file(src)
        w, h = image_dims(dst)

        rows.append(
            {
                "image_id": _stable_image_id(sha),
                "dataset": dataset,
                "split": split,
                "image_path": str(dst),
                "original_image_path": str(src),
                "width": int(w),
                "height": int(h),
                "sha256": sha,
                "source_url": (
                    f"https://www.flickr.com/photos/{meta['owner']}/{meta['photo_id']}"
                    if meta["owner"] and meta["photo_id"]
                    else None
                ),
                "license": meta["license"],
                "lat": float(meta["lat"]),
                "lon": float(meta["lon"]),
                "country": None,        # filled by a later reverse-geocoding pass
                "country_iso": None,    # ditto
                "region": None,
                "city": None,
                "place_id": meta["photo_id"],
                "contains_people_flag": None,
                "contains_faces_flag": None,
                "contains_license_plate_flag": None,
                "contains_ocr_text_flag": None,
                "is_allowed_for_public_demo": bool(is_allowed_for_public_demo),
                "notes": None
                if not meta.get("tags")
                else f"flickr_tags={meta['tags']!r};accuracy={meta['accuracy']};datetaken={meta['datetaken']}",
            }
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)

    print(
        f"manifest: {len(df)} rows  ({n_total} input images, "
        f"{skipped_no_gps} skipped for missing lat/lon) → {out_path}"
    )
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build the §7.1 manifest for Im2GPS3k.")
    ap.add_argument(
        "--raw-dir",
        default="data/raw/im2gps3k/im2gps3ktest",
        help="Directory containing the 3000 .jpg files (extracted from im2gps3ktest.zip).",
    )
    ap.add_argument(
        "--interim-root",
        default="data/interim/stripped_exif",
    )
    ap.add_argument(
        "--out",
        default="data/processed/manifests/im2gps3k_test.parquet",
    )
    ap.add_argument("--max-side", type=int, default=1024)
    args = ap.parse_args()

    build_im2gps3k_manifest(
        raw_dir=Path(args.raw_dir),
        interim_root=Path(args.interim_root),
        out_path=Path(args.out),
        max_side=args.max_side,
    )


if __name__ == "__main__":
    main()
