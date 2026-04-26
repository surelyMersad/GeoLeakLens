"""Modal app for GeoLeakLens GPU workloads.

Architecture
------------
Orchestration code (manifests, schemas, scoring math, region merging,
intervention masks) runs **locally** on whatever machine drives the
experiment. Only GPU-bound work — segmentation, model scoring, inpainting —
runs as Modal containers.

This split lets the iteration loop stay fast (edit + run pure-Python tests
locally), while the heavy GPU passes fan out across many Modal workers.

Images
------
We define narrow images per component so cold-start downloads stay scoped:
- `sam_image`     — segment-anything + torch (CUDA), used by SAM mask gen
- `geoclip_image` — geo-clip pkg + torch (CUDA), used by GeoCLIP scoring
- (later) `lama_image`, `clip_image`, `vlm_image` for inpainting / utility / VLMs

Volumes
-------
- `geoleaklens-sam-checkpoints`: persistent SAM checkpoint (~375 MB) so we
  don't re-download per cold-start.
- `geoleaklens-cache`: §4.1 prediction/mask cache keyed by
  (model, image_sha256, prompt_id, region_set_hash, intervention_type).

Usage
-----
Local (after `modal token new` once):
    modal run -m geoleaklens.modal_app::ping

Programmatic from local orchestrator:
    from geoleaklens.modal_app import GeoCLIPModal, app
    with app.run():
        out = GeoCLIPModal().predict.remote(image_bytes)
"""
from __future__ import annotations

import modal

_PYTHON = "3.11"

# ---- Images -----------------------------------------------------------------

sam_image = (
    modal.Image.debian_slim(python_version=_PYTHON)
    .apt_install("libgl1", "libglib2.0-0")  # opencv runtime deps
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "segment-anything>=1.0",
        "opencv-python-headless>=4.9",
        "numpy>=1.24",
        "pillow>=10",
    )
)

geoclip_image = (
    modal.Image.debian_slim(python_version=_PYTHON)
    .pip_install(
        "torch==2.4.1",
        "torchvision==0.19.1",
        extra_index_url="https://download.pytorch.org/whl/cu121",
    )
    .pip_install(
        "geoclip",
        "numpy>=1.24",
        "pillow>=10",
    )
)

# ---- Volumes ----------------------------------------------------------------

sam_checkpoints = modal.Volume.from_name(
    "geoleaklens-sam-checkpoints", create_if_missing=True
)
cache_volume = modal.Volume.from_name(
    "geoleaklens-cache", create_if_missing=True
)

CACHE_MOUNT = "/cache"
SAM_CKPT_MOUNT = "/checkpoints"

app = modal.App("geoleaklens")


# ---- GeoCLIP ----------------------------------------------------------------
@app.cls(
    image=geoclip_image,
    gpu="L4",
    timeout=600,
    volumes={CACHE_MOUNT: cache_volume},
)
class GeoCLIPModal:
    """Loads GeoCLIP once per container (`@modal.enter`) and serves predictions.

    Returns a dict matching the §8.1 `GeolocationModel.predict` shape so the
    local scoring pipeline doesn't care whether inference ran on Modal or on
    a local GPU.
    """

    @modal.enter()
    def load(self) -> None:
        import torch
        from geoclip import GeoCLIP

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = GeoCLIP().to(self.device)
        self.model.eval()

    @modal.method()
    def predict(self, image_bytes: bytes, top_k: int = 5) -> dict:
        import io
        import torch
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        with torch.inference_mode():
            top_pred_gps, top_pred_prob = self.model.predict(img, top_k=top_k)
        # `top_pred_gps` shape (top_k, 2): [[lat, lon], ...]; `top_pred_prob` shape (top_k,).
        top_pred_gps = top_pred_gps.cpu().tolist()
        top_pred_prob = top_pred_prob.cpu().tolist()

        if not top_pred_gps:
            return {
                "lat": None,
                "lon": None,
                "confidence": None,
                "raw_response": "",
                "parse_success": False,
                "topk": [],
                "error": "geoclip returned no predictions",
            }

        lat, lon = top_pred_gps[0]
        topk = [
            {"lat": float(la), "lon": float(lo), "score": float(p)}
            for (la, lo), p in zip(top_pred_gps, top_pred_prob)
        ]
        return {
            "lat": float(lat),
            "lon": float(lon),
            "confidence": float(top_pred_prob[0]),
            "raw_response": str(topk),
            "parse_success": True,
            "topk": topk,
            "error": None,
        }


# ---- Health check -----------------------------------------------------------
@app.function(image=geoclip_image, gpu="L4", timeout=120)
def ping() -> dict:
    """Sanity check: container starts, GPU is visible, geoclip imports."""
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device_name": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    return info


@app.local_entrypoint()
def main():
    """`modal run -m geoleaklens.modal_app` invokes this."""
    print(ping.remote())
