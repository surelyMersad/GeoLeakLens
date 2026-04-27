"""§12.1 GeoLeakLens static greedy redaction.

This is the **fast approximation**, not the headline method. The spec is explicit
(§12.1, §11.3): static greedy treats single-region attribution scores as
additive, which they aren't — it ignores suppression and saturation. Use it for
diagnostic plots, cue taxonomy, and the §13.E0 smoke step. The §12.2 dynamic
greedy (re-score after each pick on the edited image) is the primary method
for headline causal claims and lands in a later commit.

Common interface (§12 preamble):

    class RedactionMethod:
        name: str
        def select_regions(image_id, regions, scores, budget) -> list[str]
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import pandas as pd


class RedactionMethod(Protocol):
    name: str

    def select_regions(
        self,
        image_id: str,
        regions: pd.DataFrame,
        scores: pd.DataFrame,
        budget: dict,
    ) -> list[str]: ...


@dataclass
class StaticGreedy:
    """Sort regions by `score_per_area` desc, fit until area budget is full.

    `budget["area_frac"]` is the target — e.g., 0.10 for the §13.E0 step 7
    "10% budget" check. We don't overshoot: a region is added only if it fits
    in the remaining budget. If a high-scoring region is too big, we skip it
    and try the next one. The selected set's actual `area_frac` will sit at
    or below `target` — log it so the §12.3 area-match audit can verify.

    `intervention_type` is purely a label written onto downstream artifacts
    (parquet rows, filenames). The actual painting happens in
    `interventions/mask.py` etc.
    """

    name: str = "static_greedy"
    intervention_type: str = "mean_mask"

    def select_regions(
        self,
        image_id: str,
        regions: pd.DataFrame,
        scores: pd.DataFrame,
        budget: dict,
    ) -> list[str]:
        target = float(budget.get("area_frac", 0.10))
        if target <= 0:
            return []

        s = scores.loc[
            (scores["image_id"] == image_id)
            & (scores["intervention_type"] == self.intervention_type),
            ["region_id", "score_per_area"],
        ]
        r = regions.loc[
            regions["image_id"] == image_id,
            ["region_id", "area_frac"],
        ]
        merged = s.merge(r, on="region_id", how="inner")
        if merged.empty:
            return []

        # Sort by score_per_area desc; ties broken by smaller area first
        # (smaller regions are cheaper to add and rarely worse than larger ties).
        merged = merged.sort_values(
            ["score_per_area", "area_frac"], ascending=[False, True]
        )

        selected: list[str] = []
        used = 0.0
        for _, row in merged.iterrows():
            af = float(row["area_frac"])
            if used + af > target:
                # Skip oversized region; keep walking — a smaller region down
                # the list may still fit.
                continue
            selected.append(str(row["region_id"]))
            used += af
            # Once we've eaten >=99% of the budget we stop — chasing the last
            # 0.1pp on tiny regions is just noise.
            if used >= target * 0.99:
                break
        return selected


def selection_summary(
    selected_ids: list[str],
    regions: pd.DataFrame,
    image_id: str,
    target_area_frac: float,
) -> dict:
    """Bookkeeping row for the §7 redactions parquet."""
    r = regions[regions["image_id"] == image_id]
    sel = r[r["region_id"].isin(selected_ids)]
    actual = float(sel["area_frac"].sum()) if not sel.empty else 0.0
    return {
        "image_id": image_id,
        "n_selected": int(len(selected_ids)),
        "selected_region_ids": list(selected_ids),
        "target_area_frac": float(target_area_frac),
        "actual_area_frac": actual,
        "area_gap_pp": (actual - target_area_frac) * 100.0,
    }
