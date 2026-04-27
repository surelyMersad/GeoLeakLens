"""§11.3 single-region intervention effect (a.k.a. "leakage attribution").

Per §11.3 we deliberately do **not** call this quantity "causal" — it ignores
interaction effects between regions. Joint causal scoring (dynamic-greedy /
Shapley) lands in a later commit.

For a single region r and intervention edit(.):
    orig_error = d(A(x), y)
    edit_error = d(A(edit(x, r)), y)

    binary_attribution     = int(orig_success) - int(edit_success)
    continuous_attribution = log1p(edit_clip) - log1p(orig_clip)

`score_per_area` is the §11.4 primary ranking key — `continuous_attribution`
divided by region area-fraction with a small floor so a 1px region doesn't get
infinite leverage.
"""
from __future__ import annotations

import math
from typing import Optional

from geoleaklens.data.geo_utils import (
    clip_error_km,
    haversine_km,
    success_at,
)

# §11.4 floors. Mirror the constants spelled out in the spec so a future config
# tweak only has to update one place.
_AREA_FRAC_FLOOR = 0.005
_UTILITY_COST_FLOOR = 0.01


def single_region_attribution(
    *,
    image_id: str,
    region_id: str,
    model_name: str,
    intervention_type: str,
    threshold_km: float,
    true_lat: float,
    true_lon: float,
    orig_lat: Optional[float],
    orig_lon: Optional[float],
    edit_lat: Optional[float],
    edit_lon: Optional[float],
    area_frac: float,
    utility_cost: Optional[float] = None,
    method: str = "static_greedy",
) -> dict:
    """Compute one row of the §7.5 leakage-score schema for a single region.

    `utility_cost` may be `None` when the §11.5 utility metrics haven't been
    computed yet (E0 skips them); `score_per_utility` is then `None`.

    Joint-causal fields (`binary_joint_causal`, `continuous_joint_causal`) are
    explicitly `None` here — they're the dynamic-greedy / Shapley territory.
    """
    orig_error_km = haversine_km(orig_lat, orig_lon, true_lat, true_lon)
    edit_error_km = haversine_km(edit_lat, edit_lon, true_lat, true_lon)

    orig_success = success_at(orig_error_km, threshold_km)
    edit_success = success_at(edit_error_km, threshold_km)

    orig_clip = clip_error_km(orig_error_km)
    edit_clip = clip_error_km(edit_error_km)

    binary_attribution = float(int(orig_success) - int(edit_success))
    continuous_attribution = math.log1p(edit_clip) - math.log1p(orig_clip)

    score_raw = continuous_attribution
    score_per_area = continuous_attribution / max(float(area_frac), _AREA_FRAC_FLOOR)
    if utility_cost is None:
        score_per_utility: Optional[float] = None
    else:
        score_per_utility = continuous_attribution / max(
            float(utility_cost), _UTILITY_COST_FLOOR
        )

    return {
        "image_id": image_id,
        "region_id": region_id,
        "model_name": model_name,
        "threshold_km": float(threshold_km),
        "original_error_km": float(orig_clip),
        "edited_error_km": float(edit_clip),
        "original_success": bool(orig_success),
        "edited_success": bool(edit_success),
        "binary_attribution": binary_attribution,
        "continuous_attribution": float(continuous_attribution),
        "binary_joint_causal": None,
        "continuous_joint_causal": None,
        "method": method,
        "intervention_type": intervention_type,
        "area_frac": float(area_frac),
        "utility_cost": None if utility_cost is None else float(utility_cost),
        "score_raw": float(score_raw),
        "score_per_area": float(score_per_area),
        "score_per_utility": (
            None if score_per_utility is None else float(score_per_utility)
        ),
        "rank_within_image": -1,  # filled after sorting per image at the end of the run
    }


def assign_ranks_within_image(rows: list[dict]) -> list[dict]:
    """Populate `rank_within_image` by `score_per_area` desc, per `image_id`.

    Mutates and returns the input list. NaN/inf scores rank last.
    """
    by_image: dict[str, list[dict]] = {}
    for r in rows:
        by_image.setdefault(r["image_id"], []).append(r)

    for _, group in by_image.items():
        def sort_key(r: dict) -> float:
            s = r.get("score_per_area")
            if s is None or math.isnan(s) or math.isinf(s):
                return -math.inf
            return float(s)

        group.sort(key=sort_key, reverse=True)
        for i, r in enumerate(group):
            r["rank_within_image"] = i
    return rows
