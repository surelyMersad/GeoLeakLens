"""§8.6 closed VLM wrapper for OpenAI Chat Completions (GPT-4o etc.).

Used as one of the §13.E1 VLM adversaries. Closed APIs cost real money per
call, so the §8.6 caching discipline is mandatory:

  cache_key = (model_name, image_sha256, prompt_id)

The wrapper writes each call to a JSON file under `cache_dir`, keyed by the
SHA-256 of `(model + sha256(image_bytes) + prompt_id + prompt_hash)`. A
re-run with the same triple is a disk read, not a billed API call. We
include a hash of the prompt body in the key as well, so editing the prompt
text invalidates the cache automatically — protects against silent drift
where the prompt changes but a stale cached response leaks through.

Other §8.6 contract requirements implemented here:
  - Retry with exponential backoff on 429 / 5xx.
  - No image path / dataset city / GPS in the prompt (caller's job — we
    only forward what we're given).
  - Raw response saved in private local cache only — `predict()` returns
    the raw string for the caller to log into the §7.3 predictions JSONL.

Auth: reads `OPENAI_API_KEY` from env via the SDK's default behavior. We do
not accept a key parameter; never thread secrets through call sites.
"""
from __future__ import annotations

import base64
import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional


# §13.E1 default headline prompt — kept in sync with `run_baseline_geolocation.GEO_NEUTRAL_V1`
# but accepted as a parameter so callers control prompt selection.

_DEFAULT_CACHE_DIR = Path("data/processed/cache/openai")


def _sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _cache_key(
    *, model: str, image_sha256: str, prompt_id: str, prompt: str
) -> str:
    """Stable filename for a (model, image, prompt) combination."""
    h = hashlib.sha256()
    h.update(model.encode("utf-8"))
    h.update(b"\x00")
    h.update(image_sha256.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt_id.encode("utf-8"))
    h.update(b"\x00")
    h.update(prompt.encode("utf-8"))
    return h.hexdigest()


@dataclass
class OpenAIVLMWrapper:
    """Calls a vision-capable Chat Completions model and returns the raw text.

    Construction is a pure side-effect-free dataclass; the OpenAI SDK client
    is created lazily on first `predict()` call so unit tests can stub the
    factory via `client_factory`.
    """

    model: str = "gpt-4o"
    temperature: float = 0.0
    max_tokens: int = 384
    cache_dir: Path = field(default_factory=lambda: _DEFAULT_CACHE_DIR)
    max_retries: int = 4
    initial_backoff_s: float = 2.0

    # Test seam: callers can inject a fake client (e.g. a mock) without
    # hitting the network. Default factory creates the real SDK client.
    client_factory: Any = None

    def __post_init__(self) -> None:
        self.cache_dir = Path(self.cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self._client = None

    @property
    def name(self) -> str:
        return self.model

    def _get_client(self):
        if self._client is None:
            if self.client_factory is not None:
                self._client = self.client_factory()
            else:
                from openai import OpenAI

                self._client = OpenAI()
        return self._client

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _read_cache(self, key: str) -> Optional[dict]:
        p = self._cache_path(key)
        if not p.exists():
            return None
        try:
            with open(p) as f:
                return json.load(f)
        except (OSError, ValueError):
            return None

    def _write_cache(self, key: str, payload: dict) -> None:
        p = self._cache_path(key)
        tmp = p.with_suffix(".json.tmp")
        with open(tmp, "w") as f:
            json.dump(payload, f)
        os.replace(tmp, p)

    def predict(
        self,
        *,
        image_bytes: bytes,
        prompt: str,
        prompt_id: str = "geo_neutral_v1",
        image_mime: str = "image/jpeg",
    ) -> dict:
        """Send (image, prompt) to the configured model. Returns:

            {
              "raw_response": str,
              "model_id": str,
              "cache_hit": bool,
              "error": str | None,
            }

        Cache-miss path makes one billed call with up-to-`max_retries`
        exponential-backoff retries on 429/5xx.
        """
        sha = _sha256_hex(image_bytes)
        key = _cache_key(
            model=self.model, image_sha256=sha, prompt_id=prompt_id, prompt=prompt
        )

        cached = self._read_cache(key)
        if cached is not None:
            return {
                "raw_response": cached.get("raw_response", ""),
                "model_id": self.model,
                "cache_hit": True,
                "error": None,
            }

        b64 = base64.b64encode(image_bytes).decode("ascii")
        data_url = f"data:{image_mime};base64,{b64}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": data_url, "detail": "high"},
                    },
                ],
            }
        ]

        last_err: Optional[Exception] = None
        for attempt in range(self.max_retries + 1):
            try:
                client = self._get_client()
                resp = client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                self._write_cache(
                    key,
                    {
                        "raw_response": text,
                        "model": self.model,
                        "prompt_id": prompt_id,
                        "image_sha256": sha,
                    },
                )
                return {
                    "raw_response": text,
                    "model_id": self.model,
                    "cache_hit": False,
                    "error": None,
                }
            except Exception as e:
                last_err = e
                if attempt >= self.max_retries:
                    break
                # Exponential backoff with a cap. We don't introspect the
                # exception type — any failure that's not the final retry
                # gets retried. OpenAI SDK exposes `RateLimitError` and
                # `APIStatusError` but importing them creates a hard test
                # dependency we'd rather not.
                sleep_s = self.initial_backoff_s * (2 ** attempt)
                time.sleep(min(sleep_s, 30.0))

        return {
            "raw_response": "",
            "model_id": self.model,
            "cache_hit": False,
            "error": f"{type(last_err).__name__}: {last_err}" if last_err else "unknown error",
        }
