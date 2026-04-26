"""Image I/O helpers: hashing, loading, resizing.

`sha256_file` is the canonical input to the cache key
`(model, image_sha256, prompt_id, region_set_hash, intervention_type)`
that the §4.1 cost discipline depends on. It must hash the bytes-on-disk
exactly, not a re-encoded version, so that re-running the pipeline against
the same files is free.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Tuple, Union

from PIL import Image

PathLike = Union[str, Path]

_HASH_CHUNK_BYTES = 1024 * 1024  # 1 MiB


def sha256_file(path: PathLike) -> str:
    """SHA-256 of the file contents at `path`, hex-encoded."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(_HASH_CHUNK_BYTES), b""):
            h.update(chunk)
    return h.hexdigest()


def load_image(path: PathLike) -> Image.Image:
    """Open an image and force materialization. Always returns RGB."""
    img = Image.open(path)
    img.load()
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def resize_max_side(img: Image.Image, max_side: int) -> Image.Image:
    """Resize so the longer side equals max_side; no-op if already smaller."""
    w, h = img.size
    longest = max(w, h)
    if longest <= max_side:
        return img
    scale = max_side / float(longest)
    new_size = (max(1, int(round(w * scale))), max(1, int(round(h * scale))))
    return img.resize(new_size, Image.LANCZOS)


def image_dims(path: PathLike) -> Tuple[int, int]:
    """Return (width, height) without fully decoding the image."""
    with Image.open(path) as img:
        return img.size
