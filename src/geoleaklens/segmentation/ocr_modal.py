"""§9.3 EasyOCR text-region detection on Modal.

Same registration pattern as `sam_modal.py` / `qwen_modal.py` — imports
`app` from `modal_app` and registers `@app.cls(...)` so a single
`with app.run():` brings every cls up.

Why EasyOCR (rather than PaddleOCR / Tesseract)
-----------------------------------------------
The §9.3 spec lists "EasyOCR or PaddleOCR". EasyOCR has the simpler
pip path and the lightest model footprint (~150 MB across detection +
recognition). PaddleOCR ships better Asian-script accuracy but pulls
PaddlePaddle, which doubles install size. For Im2GPS3k (Western-tourist
photo bias) EasyOCR with `['en']` is enough; we can add a second
language pack later if needed for E5 cross-region transfer.

Wire format
-----------
Input: JPEG-encoded image bytes (matches the rest of the pipeline).
Output: list of dicts with `quad`, `bbox`, `text`, `confidence`. The
local `ocr_regions.py` module turns these into mask arrays.

Why GPU="T4" rather than "L4" / CPU
-----------------------------------
EasyOCR's CRAFT detection backbone benefits ~10x from GPU vs CPU.
T4 (16 GB) is the cheapest Modal GPU and far more memory than EasyOCR
needs. L4 would also work but costs ~2x more and isn't faster for
this workload. Pure CPU is plausible (~30 s / 1024×768 image) but
slow enough to hurt iteration.
"""
from __future__ import annotations

import io

import modal

from geoleaklens.modal_app import (
    EASYOCR_CACHE_MOUNT,
    app,
    easyocr_cache,
    easyocr_image,
)


@app.cls(
    image=easyocr_image,
    gpu="T4",
    timeout=900,
    volumes={EASYOCR_CACHE_MOUNT: easyocr_cache},
)
class EasyOCRModal:
    @modal.enter()
    def load(self) -> None:
        # Point EasyOCR at the persistent volume so weight downloads survive
        # cold-starts. `download_enabled=True` does the first-time pull.
        import easyocr

        self.reader = easyocr.Reader(
            ["en"],
            gpu=True,
            model_storage_directory=EASYOCR_CACHE_MOUNT,
            download_enabled=True,
            verbose=False,
        )
        try:
            easyocr_cache.commit()
        except Exception:
            # `commit` is best-effort. We still have the weights in the
            # container's local FS for this run.
            pass

    @modal.method()
    def detect(
        self,
        image_bytes: bytes,
        *,
        min_confidence: float = 0.3,
    ) -> list[dict]:
        """Detect text boxes in `image_bytes`. Returns serializable dicts.

        Each entry:
          {
            "quad": [[x, y], [x, y], [x, y], [x, y]],   # original quadrilateral
            "bbox": [x1, y1, x2, y2],                   # axis-aligned bounding box
            "text": str,
            "confidence": float,
          }

        Filtered by `min_confidence` so the parquet doesn't fill with noise
        from EasyOCR's low-confidence shadows.
        """
        import numpy as np
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        arr = np.asarray(img)

        # `detail=1` returns (quad, text, confidence) tuples.
        raw = self.reader.readtext(arr, detail=1)

        out: list[dict] = []
        for quad, text, conf in raw:
            try:
                conf_f = float(conf)
            except Exception:
                continue
            if conf_f < min_confidence:
                continue
            quad_pts = [[int(round(float(x))), int(round(float(y)))]
                        for x, y in quad]
            xs = [p[0] for p in quad_pts]
            ys = [p[1] for p in quad_pts]
            out.append({
                "quad": quad_pts,
                "bbox": [int(min(xs)), int(min(ys)),
                         int(max(xs)), int(max(ys))],
                "text": str(text),
                "confidence": conf_f,
            })
        return out
