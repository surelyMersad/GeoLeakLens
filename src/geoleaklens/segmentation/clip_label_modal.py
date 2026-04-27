"""§9.5 CLIP zero-shot region labeling on Modal.

For each (image, bbox) the cls crops the bbox, runs CLIP-ViT-L/14, and
returns the top-1 prompt by cosine similarity. Used as the §9.5 fallback
label source when neither OCR nor an object detector matched a SAM
region.

Why reuse `geoclip_image` (instead of a new clip_image)
-------------------------------------------------------
geoclip_image already has `transformers==4.45.2` + torch on CUDA. That
matters because `CLIPModel.get_image_features` changed return type to
`BaseModelOutputWithPooling` in transformers >= 4.50 (the same change
that broke geoclip in commit 89f4add). Pinning to 4.45.2 keeps the old
tensor-return API which our code uses directly. Building a separate
`clip_image` would either duplicate the pin or trip on the breaking
change.

Wire format
-----------
Input: JPEG-encoded image bytes + list of bboxes [x1, y1, x2, y2] +
prompt list. Output: list of (best_prompt, score) tuples — one per bbox,
in input order. Top-1 only; callers that want top-k can re-run with a
different prompt subset or extend the cls if needed.

Cost
----
Per call: encode N text prompts (once per cls instantiation, cached) +
N_regions image-patch encodes. CLIP-ViT-L/14 inference is ~5ms/patch on
L4 in FP16, plus encoding overhead. For N=80 regions × 30 images =
2400 patches: ~12s of pure compute, ~30s with overhead. Fast enough
that batching all regions in one image into one call is the obvious
default — done in the `label_bboxes` method.
"""
from __future__ import annotations

import io

import modal

from geoleaklens.modal_app import app, geoclip_image


_MODEL_ID = "openai/clip-vit-large-patch14"


# §9.5 default prompt list. Caller can override if they want a tighter or
# domain-specific set (e.g. for the §13.E5 model-transfer experiment we
# might want non-tourist prompts).
DEFAULT_CLIP_PROMPTS: tuple[str, ...] = (
    "a street sign",
    "a storefront",
    "a building facade",
    "a road marking",
    "a sidewalk",
    "a curb",
    "a utility pole",
    "vegetation",
    "mountains",
    "a skyline",
    "a vehicle",
    "a license plate",
    "a person",
    "the sky",
    "a window",
    "a shop sign",
    "a flag",
    "a transit sign",
)


@app.cls(
    image=geoclip_image,
    gpu="L4",
    timeout=600,
)
class CLIPLabelModal:
    """Zero-shot region labeler. Loads CLIP once per container, encodes
    prompts on demand and caches them by tuple-key for cheap reuse."""

    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import CLIPModel, CLIPProcessor

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on Ampere+/Hopper, otherwise fp16. CPU stays fp32.
        if self.device == "cuda":
            self.dtype = (
                torch.bfloat16
                if torch.cuda.is_bf16_supported()
                else torch.float16
            )
        else:
            self.dtype = torch.float32

        self.model = CLIPModel.from_pretrained(_MODEL_ID).to(self.device)
        self.model.eval()
        if self.device == "cuda":
            self.model = self.model.to(self.dtype)
        self.processor = CLIPProcessor.from_pretrained(_MODEL_ID)

        # Cache for pre-encoded text features keyed by the prompts tuple.
        # Avoids re-encoding when the same prompt list is reused across
        # `label_bboxes` calls within one container's lifetime.
        self._text_cache: dict[tuple[str, ...], "torch.Tensor"] = {}

    def _encode_prompts(self, prompts: tuple[str, ...]):
        import torch
        import torch.nn.functional as F

        if prompts in self._text_cache:
            return self._text_cache[prompts]
        inputs = self.processor(
            text=list(prompts), padding=True, return_tensors="pt"
        ).to(self.device)
        with torch.inference_mode():
            text_features = self.model.get_text_features(**inputs)
            text_features = F.normalize(text_features, dim=-1)
        self._text_cache[prompts] = text_features
        return text_features

    @modal.method()
    def cosine_similarity_pairs(
        self,
        pairs: list[tuple[bytes, bytes]],
    ) -> list[float]:
        """Cosine similarity between matched (a, b) image pairs.

        Used by §13.E4 for the per-(method, budget, image) utility metric:
        sim(original, redacted). Encodes all 2*N images in one batch and
        returns N cos-sim values.

        We avoid de-dup of repeated `a` bytes (each pair carries its own
        copy) — the runner could pre-de-dup by image_id if many pairs
        share the same original, but the encoding cost is small enough
        that we don't bother here.
        """
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if not pairs:
            return []

        # Flatten pairs into a single list, encode in one batch.
        flat: list[Image.Image] = []
        for a, b in pairs:
            flat.append(Image.open(io.BytesIO(a)).convert("RGB"))
            flat.append(Image.open(io.BytesIO(b)).convert("RGB"))

        inputs = self.processor(images=flat, return_tensors="pt").to(self.device)
        if self.device == "cuda":
            inputs = {k: v.to(self.dtype) if v.is_floating_point() else v
                      for k, v in inputs.items()}
        with torch.inference_mode():
            features = self.model.get_image_features(**inputs)
            features = F.normalize(features.float(), dim=-1)

        # Each pair occupies two consecutive rows in `features`.
        a_feats = features[0::2]
        b_feats = features[1::2]
        sims = (a_feats * b_feats).sum(dim=-1).cpu().tolist()
        return [float(s) for s in sims]

    @modal.method()
    def label_bboxes(
        self,
        image_bytes: bytes,
        bboxes: list[list[int]],
        prompts: list[str] | None = None,
    ) -> list[dict]:
        """Crop each bbox and return CLIP's top-1 prompt + score per region.

        `bboxes` is a list of [x1, y1, x2, y2] in pixel coords (matches the
        §7.2 region schema). Bboxes that fall outside the image are clamped
        before cropping. A bbox with zero area after clamping returns
        `{"label": None, "score": 0.0}` so the caller's per-region row
        order stays intact.

        `prompts` defaults to `DEFAULT_CLIP_PROMPTS` (the §9.5 list).
        """
        import numpy as np
        import torch
        import torch.nn.functional as F
        from PIL import Image

        if not bboxes:
            return []

        prompts_t = tuple(prompts) if prompts else DEFAULT_CLIP_PROMPTS
        text_features = self._encode_prompts(prompts_t)

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        W, H = img.size

        # Crop each bbox; collect those with nonzero area for batched encode.
        crops: list[Image.Image] = []
        valid_mask: list[bool] = []
        for b in bboxes:
            x1 = max(0, min(int(b[0]), W - 1))
            y1 = max(0, min(int(b[1]), H - 1))
            x2 = max(0, min(int(b[2]), W))
            y2 = max(0, min(int(b[3]), H))
            if x2 <= x1 or y2 <= y1:
                valid_mask.append(False)
                continue
            crops.append(img.crop((x1, y1, x2, y2)))
            valid_mask.append(True)

        if not crops:
            return [{"label": None, "score": 0.0} for _ in bboxes]

        # Batched image encode.
        inputs = self.processor(
            images=crops, return_tensors="pt"
        ).to(self.device)
        if self.device == "cuda":
            inputs = {k: v.to(self.dtype) if v.is_floating_point() else v
                      for k, v in inputs.items()}
        with torch.inference_mode():
            image_features = self.model.get_image_features(**inputs)
            image_features = F.normalize(image_features.float(), dim=-1)

        # cosine sim → top-1 per crop. shape (N, P).
        sims = image_features @ text_features.float().T
        top_idx = sims.argmax(dim=-1).cpu().tolist()
        top_scores = sims.max(dim=-1).values.cpu().tolist()

        # Re-thread results back into the original bbox order with
        # (None, 0.0) entries for the clamped-empty crops.
        out: list[dict] = []
        crop_iter = iter(zip(top_idx, top_scores))
        for is_valid in valid_mask:
            if not is_valid:
                out.append({"label": None, "score": 0.0})
                continue
            idx, score = next(crop_iter)
            out.append({
                "label": prompts_t[idx],
                "score": float(score),
            })
        return out
