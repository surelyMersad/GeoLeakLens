"""Unit tests for the OpenAI wrapper. No real API calls — uses a fake client."""
from __future__ import annotations

import base64
import time
from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import pytest

from geoleaklens.models.openai_wrapper import OpenAIVLMWrapper


def _fake_response(text: str):
    """Build a stub matching the SDK's `chat.completions.create` return shape."""
    msg = MagicMock()
    msg.content = text
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


@dataclass
class _FakeClient:
    """Minimal stand-in for `openai.OpenAI()`."""

    side_effects: list[Any]
    calls: list = None

    def __post_init__(self):
        if self.calls is None:
            self.calls = []
        # Mirror the SDK shape `client.chat.completions.create(...)`.
        self.chat = MagicMock()
        self.chat.completions = MagicMock()
        self.chat.completions.create = self._create

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        nxt = self.side_effects.pop(0)
        if isinstance(nxt, Exception):
            raise nxt
        return nxt


def test_cache_miss_calls_api_then_cache_hit_skips(tmp_path):
    fake = _FakeClient(side_effects=[_fake_response('{"city":"Paris"}')])
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path, client_factory=lambda: fake, max_retries=0
    )

    img = b"\xff\xd8\xff\xe0fakejpegbytes"
    out1 = w.predict(image_bytes=img, prompt="where?")
    assert out1["cache_hit"] is False
    assert out1["error"] is None
    assert "Paris" in out1["raw_response"]
    assert len(fake.calls) == 1

    # Same (image, prompt) → cache hit, no new API call.
    out2 = w.predict(image_bytes=img, prompt="where?")
    assert out2["cache_hit"] is True
    assert out2["raw_response"] == out1["raw_response"]
    assert len(fake.calls) == 1


def test_changing_prompt_invalidates_cache(tmp_path):
    fake = _FakeClient(
        side_effects=[
            _fake_response("answer A"),
            _fake_response("answer B"),
        ]
    )
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path, client_factory=lambda: fake, max_retries=0
    )
    img = b"img"
    a = w.predict(image_bytes=img, prompt="prompt A", prompt_id="p1")
    b = w.predict(image_bytes=img, prompt="prompt B", prompt_id="p1")
    assert a["raw_response"] == "answer A"
    assert b["raw_response"] == "answer B"
    assert len(fake.calls) == 2


def test_changing_image_invalidates_cache(tmp_path):
    fake = _FakeClient(
        side_effects=[
            _fake_response("first image"),
            _fake_response("second image"),
        ]
    )
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path, client_factory=lambda: fake, max_retries=0
    )
    p = "where?"
    out1 = w.predict(image_bytes=b"image_one", prompt=p)
    out2 = w.predict(image_bytes=b"image_two", prompt=p)
    assert out1["raw_response"] == "first image"
    assert out2["raw_response"] == "second image"


def test_retries_then_succeeds(tmp_path, monkeypatch):
    """Two transient failures → third call returns. Backoff sleeps are mocked."""
    err = RuntimeError("rate limit (429)")
    fake = _FakeClient(
        side_effects=[err, err, _fake_response("recovered")]
    )
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path,
        client_factory=lambda: fake,
        max_retries=3,
        initial_backoff_s=0.0,
    )
    out = w.predict(image_bytes=b"img", prompt="p")
    assert out["error"] is None
    assert out["raw_response"] == "recovered"
    assert len(fake.calls) == 3


def test_retries_exhausted_returns_error(tmp_path, monkeypatch):
    err = RuntimeError("persistent failure")
    fake = _FakeClient(side_effects=[err, err, err, err, err])
    monkeypatch.setattr(time, "sleep", lambda _s: None)
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path,
        client_factory=lambda: fake,
        max_retries=2,
        initial_backoff_s=0.0,
    )
    out = w.predict(image_bytes=b"img", prompt="p")
    assert out["error"] is not None
    assert "persistent failure" in out["error"]
    assert out["raw_response"] == ""


def test_payload_contains_image_data_url_and_text(tmp_path):
    fake = _FakeClient(side_effects=[_fake_response("{}")])
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path, client_factory=lambda: fake, max_retries=0
    )
    img = b"\x00\x01\x02hello"
    w.predict(image_bytes=img, prompt="my-prompt-text", image_mime="image/png")

    sent = fake.calls[0]
    msgs = sent["messages"]
    assert msgs[0]["role"] == "user"
    contents = msgs[0]["content"]
    text_block = next(c for c in contents if c["type"] == "text")
    image_block = next(c for c in contents if c["type"] == "image_url")
    assert text_block["text"] == "my-prompt-text"
    expected_b64 = base64.b64encode(img).decode("ascii")
    assert image_block["image_url"]["url"] == f"data:image/png;base64,{expected_b64}"


def test_temperature_and_model_passed_through(tmp_path):
    fake = _FakeClient(side_effects=[_fake_response("ok")])
    w = OpenAIVLMWrapper(
        model="gpt-4o-mini",
        temperature=0.0,
        max_tokens=128,
        cache_dir=tmp_path,
        client_factory=lambda: fake,
        max_retries=0,
    )
    w.predict(image_bytes=b"x", prompt="p")
    sent = fake.calls[0]
    assert sent["model"] == "gpt-4o-mini"
    assert sent["temperature"] == 0.0
    assert sent["max_tokens"] == 128


def test_cache_files_are_atomic(tmp_path):
    """No `.tmp` files left behind after a successful write."""
    fake = _FakeClient(side_effects=[_fake_response("ok")])
    w = OpenAIVLMWrapper(
        cache_dir=tmp_path, client_factory=lambda: fake, max_retries=0
    )
    w.predict(image_bytes=b"x", prompt="p")
    leftovers = list(tmp_path.glob("*.tmp"))
    assert leftovers == []
