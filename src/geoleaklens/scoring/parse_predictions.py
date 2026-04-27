"""§8.3 prediction parsing.

Turns whatever a geolocation model returned into the §7.3 `parsed` dict, with
a `parse_success` flag. Two input shapes:

  1. Already-parsed `dict` (the GeoCLIP wrapper at §8.4 returns this directly,
     keyed `lat`/`lon`/`confidence`/`topk`/...). We just normalize fields and
     validate ranges.
  2. Raw VLM string — strict `json.loads`, falling back to a regex that pulls
     the first `{...}` block. Used by E1+ when a VLM returns prose-with-JSON.

Anything outside the WGS84 lat/lon ranges, NaN, inf, or non-numeric => the
field is set to `None`. `parse_success` is `False` only when no JSON could be
recovered at all; valid JSON with null lat/lon counts as `parse_success=True`
with `lat=lon=None` (the model said "I can't tell" — a successful parse).
"""
from __future__ import annotations

import json
import math
import re
from typing import Any, Optional

from geoleaklens.data.geo_utils import is_valid_latlon

# §7.3 parsed-prediction schema: every key present, missing values => None / [].
PARSED_KEYS: tuple[str, ...] = (
    "lat",
    "lon",
    "country",
    "region",
    "city",
    "confidence",
    "evidence",
)

# First top-level JSON object in a string — non-greedy, single-line tolerant.
# Matches `{...}` allowing nested braces only one level deep, which is enough
# for the §8.2 prompt schemas (flat objects with array values).
_JSON_OBJECT_RE = re.compile(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", re.DOTALL)


def _coerce_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _coerce_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    if isinstance(x, str):
        s = x.strip()
        return s if s else None
    return str(x)


def _coerce_evidence(x: Any) -> list[str]:
    if x is None:
        return []
    if isinstance(x, str):
        # Some VLMs return a single string instead of a list — wrap it.
        return [x] if x.strip() else []
    if isinstance(x, (list, tuple)):
        return [str(e) for e in x if e is not None and str(e).strip()]
    return []


def _normalize_dict(d: dict) -> dict:
    """Apply §7.3 normalization to whichever keys are present in `d`.

    Accepts both `latitude`/`longitude` (VLM JSON schemas) and `lat`/`lon`
    (GeoCLIP wrapper). Out-of-range / non-finite coords => None.
    """
    lat = _coerce_float(d.get("lat", d.get("latitude")))
    lon = _coerce_float(d.get("lon", d.get("longitude")))
    if not is_valid_latlon(lat, lon):
        # Don't keep half a coordinate — it can't produce a haversine error.
        lat = None
        lon = None

    confidence = _coerce_float(d.get("confidence"))
    if confidence is not None:
        confidence = max(0.0, min(1.0, confidence))

    return {
        "lat": lat,
        "lon": lon,
        "country": _coerce_str(d.get("country")),
        "region": _coerce_str(d.get("region")),
        "city": _coerce_str(d.get("city")),
        "confidence": confidence,
        "evidence": _coerce_evidence(d.get("evidence", d.get("visual_evidence"))),
    }


def _extract_first_json_object(text: str) -> Optional[dict]:
    """Regex-based fallback: find the first `{...}` and json.loads it."""
    m = _JSON_OBJECT_RE.search(text)
    if not m:
        return None
    try:
        obj = json.loads(m.group(0))
    except (ValueError, TypeError):
        return None
    return obj if isinstance(obj, dict) else None


def parse_geolocation_response(raw: Any) -> dict:
    """Parse a model response into the §7.3 prediction row.

    Returns:
        {
          "raw_response": str,
          "parsed": {lat, lon, country, region, city, confidence, evidence},
          "parse_success": bool,
          "error": str | None,
        }
    """
    if isinstance(raw, dict):
        # Already structured (GeoCLIP wrapper, etc.).
        return {
            "raw_response": str(raw.get("raw_response", raw)),
            "parsed": _normalize_dict(raw),
            "parse_success": True,
            "error": None,
        }

    text = "" if raw is None else str(raw)
    raw_response = text

    # 1. Strict JSON.
    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        obj = None

    # 2. Regex-extracted first object.
    if not isinstance(obj, dict):
        obj = _extract_first_json_object(text)

    if not isinstance(obj, dict):
        return {
            "raw_response": raw_response,
            "parsed": {k: (None if k != "evidence" else []) for k in PARSED_KEYS},
            "parse_success": False,
            "error": "no JSON object found in response",
        }

    return {
        "raw_response": raw_response,
        "parsed": _normalize_dict(obj),
        "parse_success": True,
        "error": None,
    }
