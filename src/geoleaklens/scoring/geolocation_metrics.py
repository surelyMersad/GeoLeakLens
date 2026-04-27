"""§11.2 reporting metrics for §13.E1.

Aggregates per-image errors into the cell of the baseline table:

  - median geodesic error (PRIMARY per §11.2)
  - Acc@thresholds (§11.2 secondary curve, §13.E1 reports 1/25/200/750/2500 km)
  - parse success rate
  - mean clipped error (tertiary, kept for reference)

Inputs come as a list of `(error_km, parse_success)` tuples — one per image
attempt. Inf or None errors are clipped to MAX_ERROR_KM via §11.1, then
medians and means are computed on the clipped values. Parse failures count
as success=False at every threshold (per §11.2: "If lat/lon null, geodesic
error = infinity for threshold accuracy").

The §13.E1 default threshold list lives in `default.yaml`:
  thresholds_km: [1, 25, 200, 750, 2500]
"""
from __future__ import annotations

import math
from statistics import median
from typing import Iterable, Optional

from geoleaklens.data.geo_utils import clip_error_km, success_at

DEFAULT_THRESHOLDS_KM: tuple[float, ...] = (1.0, 25.0, 200.0, 750.0, 2500.0)


def aggregate_errors(
    errors_km: Iterable[Optional[float]],
    parse_success: Iterable[bool],
    *,
    thresholds_km: Iterable[float] = DEFAULT_THRESHOLDS_KM,
) -> dict:
    """Per-cell aggregate (one row of `tables/baseline_geolocation.csv`).

    Pairs `errors_km[i]` with `parse_success[i]`. Returns a dict ready to
    flatten into a CSV row via `pd.DataFrame([result])`.
    """
    errs = list(errors_km)
    parses = list(parse_success)
    if len(errs) != len(parses):
        raise ValueError(
            f"errors_km ({len(errs)}) and parse_success ({len(parses)}) must align"
        )
    n = len(errs)
    if n == 0:
        return {"n": 0}

    clipped = [clip_error_km(e) for e in errs]
    out: dict = {
        "n": n,
        "n_parsed": int(sum(1 for p in parses if p)),
        "parse_success_rate": float(sum(1 for p in parses if p) / n),
        "median_error_km": float(median(clipped)),
        "mean_clipped_error_km": float(sum(clipped) / n),
    }
    for tau in thresholds_km:
        # Threshold success requires both a parsed prediction AND error <= tau.
        # `clip_error_km` already saturates None/inf to MAX_ERROR_KM, which
        # fails any tau < MAX_ERROR_KM, so the parse check is redundant in
        # practice but kept for clarity.
        hits = sum(
            1
            for e, p in zip(clipped, parses)
            if p and success_at(e, tau)
        )
        out[f"acc_{int(tau)}km"] = float(hits / n)
    return out


def per_image_errors(
    pred_lat: Iterable[Optional[float]],
    pred_lon: Iterable[Optional[float]],
    true_lat: Iterable[float],
    true_lon: Iterable[float],
) -> list[float]:
    """Convenience: zip predictions with ground truth, return a list of
    error_km (or +inf when prediction is missing).

    Used by the E1 runner to flatten predictions JSONL into the input shape
    `aggregate_errors` expects.
    """
    from geoleaklens.data.geo_utils import haversine_km

    out: list[float] = []
    for plat, plon, tlat, tlon in zip(pred_lat, pred_lon, true_lat, true_lon):
        d = haversine_km(plat, plon, float(tlat), float(tlon))
        out.append(float("inf") if math.isinf(d) else d)
    return out
