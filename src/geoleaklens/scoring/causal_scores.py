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


# Default §10.7 intervention set — overridable so callers can extend (e.g. add
# `replace`, `crop`) without editing this module.
DEFAULT_INTERVENTION_SET: tuple[str, ...] = ("mean_mask", "blur", "inpaint")


def min_across_interventions(
    scores,  # pandas.DataFrame
    intervention_types: tuple[str, ...] = DEFAULT_INTERVENTION_SET,
):
    """§10.7 conservative score combiner.

    For each `(image_id, region_id)` group that has a row for *every* listed
    intervention_type, emit one new row with `intervention_type = "min_across"`
    whose scoring fields are the per-group minimum:

        continuous_attribution_min = min over intervention rows
        binary_attribution_min     = min over intervention rows
        score_per_area_min         = continuous_attribution_min / max(area_frac, 0.005)

    Per the spec, taking the **min** is the conservative reading: a region only
    counts as high-leakage if *every* intervention agrees it is. A
    mean_mask-only artifact (e.g. flat-color patch detected as "redacted") will
    have a small Δ under blur or inpaint and so the min collapses toward zero.

    Other §7.5 fields are filled deterministically:
      - `original_error_km` / `original_success`: identical across the
        intervention rows (unedited prediction), so we just pick from one.
      - `edited_error_km` / `edited_success`: copied from the intervention
        whose `continuous_attribution` produced the min — i.e., the most
        conservative intervention's edited prediction.
      - `method`: preserved from the contributing rows when consistent;
        otherwise `"min_across"`.
      - `utility_cost` / `score_per_utility`: unchanged from the most-
        conservative intervention's row (utility metrics aren't aggregated
        the same way as leakage attribution).

    Groups missing one or more of the listed interventions are *excluded* from
    the output — partial coverage shouldn't silently undercount.

    Returns a new DataFrame containing **only** the min_across rows. Concat
    with the input if you want both per-intervention and aggregated rows in
    one frame:

        out = pd.concat([scores, min_across_interventions(scores)],
                        ignore_index=True)
    """
    import pandas as pd  # local import — keeps the module pandas-optional

    if scores.empty:
        return scores.iloc[0:0].copy()

    wanted = list(intervention_types)
    sub = scores[scores["intervention_type"].isin(wanted)].copy()
    if sub.empty:
        return sub.iloc[0:0]

    out_rows: list[dict] = []
    for (image_id, region_id), group in sub.groupby(["image_id", "region_id"]):
        present = set(group["intervention_type"])
        if not all(t in present for t in wanted):
            continue  # partial coverage — skip per §10.7 conservatism

        # Pick the intervention with the minimum continuous_attribution
        # (i.e., the one that produced the smallest claim of leakage). That
        # row's edited_* fields are the conservative ones to surface.
        min_idx = group["continuous_attribution"].astype(float).idxmin()
        argmin_row = group.loc[min_idx]
        cont_min = float(argmin_row["continuous_attribution"])
        bin_min = float(group["binary_attribution"].astype(float).min())

        area_frac = float(argmin_row["area_frac"])
        score_per_area_min = cont_min / max(area_frac, 0.005)
        utility = argmin_row.get("utility_cost", None)
        if utility is None or (isinstance(utility, float) and math.isnan(utility)):
            score_per_utility = None
        else:
            score_per_utility = cont_min / max(float(utility), 0.01)

        methods = set(group["method"]) if "method" in group.columns else set()
        method = next(iter(methods)) if len(methods) == 1 else "min_across"

        out_rows.append({
            "image_id": image_id,
            "region_id": region_id,
            "model_name": argmin_row.get("model_name"),
            "threshold_km": argmin_row.get("threshold_km"),
            "original_error_km": argmin_row.get("original_error_km"),
            "edited_error_km": argmin_row.get("edited_error_km"),
            "original_success": argmin_row.get("original_success"),
            "edited_success": argmin_row.get("edited_success"),
            "binary_attribution": bin_min,
            "continuous_attribution": cont_min,
            "binary_joint_causal": None,
            "continuous_joint_causal": None,
            "method": method,
            "intervention_type": "min_across",
            "area_frac": area_frac,
            "utility_cost": utility,
            "score_raw": cont_min,
            "score_per_area": score_per_area_min,
            "score_per_utility": score_per_utility,
            "rank_within_image": -1,  # filled later via assign_ranks_within_image
        })

    return pd.DataFrame(out_rows)
