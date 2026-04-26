"""Build a §7.1 image manifest as Parquet.

Inputs:
  - raw_root: directory of raw source images.
  - interim_root: directory under which stripped + resized copies are written.
  - dataset / split: identifiers for the manifest rows.
  - metadata_csv (optional): per-image ground truth (image_path, lat, lon, ...).

For each raw image:
  1. Read EXIF GPS as a fallback ground-truth source (used only if the
     metadata CSV does not supply a row).
  2. Re-encode without EXIF and resize, writing into interim_root.
  3. Compute sha256 of the *original* file — this seeds the cache key
     downstream (§4.1).
  4. Emit one row per the §7.1 manifest schema.

The §7.1 detector-derived flags (contains_people_flag, contains_faces_flag,
contains_license_plate_flag, contains_ocr_text_flag) are left as null here
and populated by a later detector pass at manifest-build time.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import pandas as pd
from PIL import ExifTags, Image

from .image_io import image_dims, sha256_file
from .strip_exif import _IMG_SUFFIXES, strip_and_resize

PathLike = Union[str, Path]

# Order matches the §7.1 schema. Keep stable: downstream code reads by name.
MANIFEST_COLUMNS = [
    "image_id",
    "dataset",
    "split",
    "image_path",
    "original_image_path",
    "width",
    "height",
    "sha256",
    "source_url",
    "license",
    "lat",
    "lon",
    "country",
    "country_iso",
    "region",
    "city",
    "place_id",
    "contains_people_flag",
    "contains_faces_flag",
    "contains_license_plate_flag",
    "contains_ocr_text_flag",
    "is_allowed_for_public_demo",
    "notes",
]


def _gps_to_decimal(value, ref) -> Optional[float]:
    """Convert PIL EXIF GPS (degrees, minutes, seconds + ref) to decimal degrees."""
    try:
        d, m, s = value
        deg = float(d) + float(m) / 60.0 + float(s) / 3600.0
    except Exception:
        return None
    if ref in ("S", "W"):
        deg = -deg
    return deg


def extract_exif_gps(path: PathLike) -> Tuple[Optional[float], Optional[float]]:
    """Best-effort EXIF GPS extraction. Returns (None, None) if unavailable."""
    try:
        with Image.open(path) as img:
            raw_exif = img.getexif()
            if not raw_exif:
                return None, None
            # GPS sub-IFD lives at tag 0x8825.
            gps_ifd = raw_exif.get_ifd(0x8825)
            if not gps_ifd:
                return None, None
            gps_tags = {ExifTags.GPSTAGS.get(k, k): v for k, v in gps_ifd.items()}
            lat_val = gps_tags.get("GPSLatitude")
            lat_ref = gps_tags.get("GPSLatitudeRef")
            lon_val = gps_tags.get("GPSLongitude")
            lon_ref = gps_tags.get("GPSLongitudeRef")
            if lat_val and lon_val and lat_ref and lon_ref:
                return (
                    _gps_to_decimal(lat_val, lat_ref),
                    _gps_to_decimal(lon_val, lon_ref),
                )
    except Exception:
        return None, None
    return None, None


def _stable_image_id(sha256: str) -> str:
    """Short deterministic id derived from sha256."""
    return f"img_{sha256[:16]}"


def _walk_raw_images(root: Path) -> Iterable[Path]:
    for p in sorted(root.rglob("*")):
        if p.is_file() and p.suffix.lower() in _IMG_SUFFIXES:
            yield p


def build_manifest(
    raw_root: PathLike,
    interim_root: PathLike,
    out_path: PathLike,
    dataset: str,
    split: str = "mvp",
    max_side: int = 1024,
    metadata_csv: Optional[PathLike] = None,
    is_allowed_for_public_demo: bool = False,
) -> pd.DataFrame:
    raw_root = Path(raw_root)
    interim_root = Path(interim_root)
    out_path = Path(out_path)

    metadata: dict = {}
    if metadata_csv is not None and Path(metadata_csv).exists():
        meta_df = pd.read_csv(metadata_csv)
        for row in meta_df.to_dict(orient="records"):
            rel = row.get("image_path") or row.get("path")
            if rel:
                metadata[str(rel)] = row

    rows = []
    for src in _walk_raw_images(raw_root):
        rel = src.relative_to(raw_root)
        dst = interim_root / dataset / rel
        strip_and_resize(src, dst, max_side=max_side)

        sha = sha256_file(src)
        w, h = image_dims(dst)
        exif_lat, exif_lon = extract_exif_gps(src)
        meta_row: dict = metadata.get(str(rel), {})

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
                "source_url": meta_row.get("source_url"),
                "license": meta_row.get("license"),
                "lat": meta_row.get("lat", exif_lat),
                "lon": meta_row.get("lon", exif_lon),
                "country": meta_row.get("country"),
                "country_iso": meta_row.get("country_iso"),
                "region": meta_row.get("region"),
                "city": meta_row.get("city"),
                "place_id": meta_row.get("place_id"),
                "contains_people_flag": None,
                "contains_faces_flag": None,
                "contains_license_plate_flag": None,
                "contains_ocr_text_flag": None,
                "is_allowed_for_public_demo": bool(
                    meta_row.get("is_allowed_for_public_demo", is_allowed_for_public_demo)
                ),
                "notes": meta_row.get("notes"),
            }
        )

    df = pd.DataFrame(rows, columns=MANIFEST_COLUMNS)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    return df


def main() -> None:
    ap = argparse.ArgumentParser(description="Build a §7.1 image manifest.")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--raw-root", required=True)
    ap.add_argument("--interim-root", default="data/interim/stripped_exif")
    ap.add_argument("--out", required=True)
    ap.add_argument("--split", default="mvp")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--metadata-csv", default=None)
    args = ap.parse_args()

    df = build_manifest(
        raw_root=Path(args.raw_root),
        interim_root=Path(args.interim_root),
        out_path=Path(args.out),
        dataset=args.dataset,
        split=args.split,
        max_side=args.max_side,
        metadata_csv=Path(args.metadata_csv) if args.metadata_csv else None,
    )
    print(f"manifest rows: {len(df)} -> {args.out}")


if __name__ == "__main__":
    main()
