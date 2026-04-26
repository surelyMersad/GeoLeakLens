"""Strip EXIF / metadata and resize images for the pipeline.

§1.4 ethical guardrails: every image fed to a model must have EXIF removed,
because EXIF GPS or camera-make tags would let the model cheat the inference
task. §6.1 also calls this out for Im2GPS3k, whose source images often retain
EXIF.

Strategy: re-encode through a fresh `Image.new()` + `paste()` so the saved file
inherits an empty `info` dict (no EXIF, no PNG text chunks, no XMP).
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Union

from PIL import Image

from .image_io import load_image, resize_max_side

PathLike = Union[str, Path]

_IMG_SUFFIXES = {".jpg", ".jpeg", ".png", ".webp"}


def strip_and_resize(
    src_path: PathLike,
    dst_path: PathLike,
    max_side: int = 1024,
    quality: int = 95,
) -> Path:
    """Load src, drop metadata, resize so longest side <= max_side, save to dst.

    Returns the destination Path. Creates parent directories as needed.
    """
    src_path = Path(src_path)
    dst_path = Path(dst_path)
    dst_path.parent.mkdir(parents=True, exist_ok=True)

    img = load_image(src_path)
    img = resize_max_side(img, max_side)

    # Re-host pixels in a fresh Image so `info` (EXIF, PNG text, XMP, ICC) is
    # empty. `paste` copies pixels without metadata.
    clean = Image.new(img.mode, img.size)
    clean.paste(img)

    suffix = dst_path.suffix.lower()
    if suffix in (".jpg", ".jpeg"):
        clean.save(dst_path, format="JPEG", quality=quality, optimize=True)
    elif suffix == ".png":
        clean.save(dst_path, format="PNG", optimize=True)
    elif suffix == ".webp":
        clean.save(dst_path, format="WEBP", quality=quality)
    else:
        clean.save(dst_path)

    return dst_path


def _walk_images(root: Path):
    for path in sorted(root.rglob("*")):
        if path.is_file() and path.suffix.lower() in _IMG_SUFFIXES:
            yield path


def main() -> None:
    ap = argparse.ArgumentParser(description="Strip EXIF and resize images.")
    ap.add_argument("--src", required=True, help="Source image or directory.")
    ap.add_argument("--out-root", required=True, help="Destination directory root.")
    ap.add_argument("--max-side", type=int, default=1024)
    ap.add_argument("--quality", type=int, default=95)
    args = ap.parse_args()

    src = Path(args.src)
    out_root = Path(args.out_root)

    if src.is_file():
        dst = out_root / src.name
        strip_and_resize(src, dst, args.max_side, args.quality)
        print(dst)
        return

    n = 0
    for path in _walk_images(src):
        rel = path.relative_to(src)
        dst = out_root / rel
        strip_and_resize(path, dst, args.max_side, args.quality)
        n += 1
    print(f"stripped {n} image(s) -> {out_root}")


if __name__ == "__main__":
    main()
