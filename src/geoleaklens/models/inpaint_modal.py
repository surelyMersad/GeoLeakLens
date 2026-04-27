"""§10.4 LaMa inpainting as a Modal class.

Same registration pattern as `segmentation/sam_modal.py` and
`models/qwen_modal.py`: imports `app` from `modal_app` and attaches an
`@app.cls(...)` so a single `with app.run():` brings up every cls the
orchestrator needs.

Why LaMa
--------
§10.4 names "LaMa/simple-lama" as the spec's preferred inpainter. We use
`simple-lama-inpainting`, a thin wrapper around Suvorov et al.'s Big LaMa
that auto-downloads weights on first use. Weights land in
`/lama_cache/torch/hub/checkpoints/` (persistent volume), so subsequent
cold-starts skip the ~200 MB download.

Why a Modal cls (vs running locally)
------------------------------------
LaMa runs ~10× faster on a GPU than CPU for typical 1024×768 images,
and we'll be inpainting many (image, region) pairs for the §10.7
min_across score. Batching them through a warm Modal container amortizes
the model-load cost.

Wire format
-----------
Inputs and outputs are plain bytes — JPEG-encoded image, PNG-encoded
single-channel mask. This keeps the Modal boundary independent of any
PIL/numpy version mismatch between local and remote, and matches the
§4.1 caching model where everything is keyed by `image_sha256`.
"""
from __future__ import annotations

import io
import os

import modal

from geoleaklens.modal_app import (
    LAMA_CACHE_MOUNT,
    app,
    lama_cache,
    lama_image,
)


@app.cls(
    image=lama_image,
    gpu="L4",
    timeout=900,
    volumes={LAMA_CACHE_MOUNT: lama_cache},
)
class LaMaModal:
    @modal.enter()
    def load(self) -> None:
        # simple-lama-inpainting downloads weights via torch.hub the first
        # time it's instantiated. Pin the cache to the persistent volume so
        # we don't re-download per cold-start.
        os.environ["TORCH_HOME"] = os.path.join(LAMA_CACHE_MOUNT, "torch")
        os.makedirs(os.environ["TORCH_HOME"], exist_ok=True)

        from simple_lama_inpainting import SimpleLama

        self.lama = SimpleLama()
        # Persist any newly-downloaded weights to the volume.
        try:
            lama_cache.commit()
        except Exception:
            # `commit` is idempotent and best-effort; if it raises we still
            # have the weights in container-local FS for this run.
            pass

    @modal.method()
    def inpaint(self, image_bytes: bytes, mask_bytes: bytes) -> bytes:
        """Run LaMa on a (image, mask) pair, return JPEG-encoded result.

        `mask_bytes` is a single-channel PNG where non-zero pixels are the
        region to inpaint. SimpleLama expects PIL images; we convert via
        `Image.open(BytesIO(...))` on both ends.

        Note on dimensions: LaMa internally pads input dimensions up to a
        multiple of 8 (Fourier-convolution architecture requirement) and
        returns the padded canvas. SimpleLama doesn't crop back. We do —
        the original content sits at the top-left of LaMa's output, so a
        simple crop to the input size restores §10.4's "dimensions
        unchanged" invariant.
        """
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        mask = Image.open(io.BytesIO(mask_bytes)).convert("L")
        if mask.size != img.size:
            raise ValueError(
                f"mask size {mask.size} does not match image size {img.size}"
            )
        result = self.lama(img, mask)
        # Crop back to the input dimensions if LaMa padded.
        if result.size != img.size:
            w, h = img.size
            rw, rh = result.size
            if rw < w or rh < h:
                raise ValueError(
                    f"LaMa returned smaller image {result.size} than input {img.size}"
                )
            result = result.crop((0, 0, w, h))
        out = io.BytesIO()
        result.convert("RGB").save(out, format="JPEG", quality=92)
        return out.getvalue()
