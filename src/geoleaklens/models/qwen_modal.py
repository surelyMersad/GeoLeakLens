"""Qwen2.5-VL-7B-Instruct as a Modal class — §8.5 open-VLM adversary.

Used as the §13.E1 VLM-on-subset (vlm_subset_n=100). Per the spec, the
paper's threat model is VLMs, not retrieval models, so at least one VLM in
E1 is non-negotiable. Qwen2.5-VL-7B is the cheapest open option that's
still capable enough to plausibly succeed at the geolocation task.

Architecture
------------
Same pattern as `segmentation/sam_modal.py`: imports `app` from
`modal_app` and registers `@app.cls(...)` so a single `with app.run():`
in the orchestrator brings up GeoCLIP, SAM, and Qwen together.

Loaded model: `Qwen/Qwen2.5-VL-7B-Instruct`. ~16 GB in FP16, fits on L4
(24 GB) with room for KV cache. We use `device_map="auto"` and fall back
to bf16 when the GPU supports it.

Output contract
---------------
`generate(image_bytes, prompt) -> dict` matches the §8.1 base interface,
returning whatever string the VLM emitted as `raw_response`. Parsing
into the §7.3 row happens locally via `parse_geolocation_response()` —
keeps the Modal cls free of project schema knowledge.

Generation knobs
----------------
- `max_new_tokens = 384`: enough headroom for the longest plausible JSON
  per `geo_neutral_v1`, but caps cost on hallucinated essays.
- `temperature = 0.0`: §8.2 / §13 default — deterministic for
  reproducibility.
- `do_sample = False`: explicit greedy decoding, paired with temperature
  0 so transformers doesn't warn.
"""
from __future__ import annotations

import io
from typing import Any

import modal

from geoleaklens.modal_app import app, qwen_image


_MODEL_ID = "Qwen/Qwen2.5-VL-7B-Instruct"


@app.cls(
    image=qwen_image,
    gpu="L4",
    timeout=900,
)
class QwenVLModal:
    @modal.enter()
    def load(self) -> None:
        import torch
        from transformers import (
            AutoProcessor,
            Qwen2_5_VLForConditionalGeneration,
        )

        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        # bf16 on Ampere+/Hopper, otherwise fp16 (L4 supports bf16).
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16

        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            _MODEL_ID,
            torch_dtype=dtype,
            device_map="auto",
        )
        self.model.eval()
        self.processor = AutoProcessor.from_pretrained(_MODEL_ID)

    @modal.method()
    def generate(
        self,
        image_bytes: bytes,
        prompt: str,
        *,
        max_new_tokens: int = 384,
    ) -> dict[str, Any]:
        import torch
        from PIL import Image
        from qwen_vl_utils import process_vision_info

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": img},
                    {"type": "text", "text": prompt},
                ],
            }
        ]
        text = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self.processor(
            text=[text],
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
        ).to(self.device)

        with torch.inference_mode():
            generated_ids = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                temperature=0.0,
            )
        # Strip the prompt tokens from the start of each generated sequence.
        trimmed = generated_ids[:, inputs.input_ids.shape[1] :]
        out_text = self.processor.batch_decode(
            trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
        )[0]

        return {
            "raw_response": out_text,
            "model_id": _MODEL_ID,
        }
