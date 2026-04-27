"""§12.3 / §12.4 redaction baselines for §13.E2 sparsity comparisons.

Each baseline implements the same interface as `redaction.methods.StaticGreedy`
so the E2 runner can swap them in/out:

    method.select_regions(image_id, regions, scores, budget) -> list[str]

`scores` is accepted but ignored by these baselines (they're score-blind).

§12.3 random baseline — area-matched
-------------------------------------
Per §12.3 the random comparator MUST hit the same per-image area-fraction
that GeoLeakLens uses on that image (within ±0.5pp). Otherwise the
"random vs targeted" comparison is biased: a random selection that ate
12% of the image when GeoLeakLens ate 8% is comparing 12-vs-8, not
"random-vs-targeted at the same budget."

Protocol implemented here:
1. Permute the available regions uniformly at random (without replacement).
2. Walk the permutation, accumulating area; stop *before* the next region
   would exceed `target_area_frac + 0.5pp`.
3. Caller is responsible for invoking with the right `target_area_frac` —
   typically the actual area GeoLeakLens used on that same image.

The grid-fallback step in §12.3 ("if accumulated area is below
target - 0.5pp, fill with grid patches") is deferred until grid regions
are emitted by `merge_regions`. For E2 v0 we report the actual achieved
area alongside the target so any audit can flag mismatches.

§12.4 largest baseline
----------------------
Sort regions by `area_frac` desc, fit until budget. Controls for "lots of
content removed regardless of what it was" — protects against a
GeoLeakLens win that's actually just an area-fraction win.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class RandomRegions:
    """§12.3 area-matched random baseline.

    `seed` is used for the per-call RNG so multiple seeds (the spec asks for
    10 per image, §12.3 step 6) can be obtained by varying it. The runner
    is expected to drive the seed loop, not us.
    """

    name: str = "random"
    intervention_type: str = "mean_mask"
    seed: int = 0
    area_match_tolerance_pp: float = 0.5

    def select_regions(
        self,
        image_id: str,
        regions: pd.DataFrame,
        scores: Optional[pd.DataFrame],  # accepted for interface parity, ignored
        budget: dict,
    ) -> list[str]:
        target = float(budget.get("area_frac", 0.10))
        if target <= 0:
            return []
        r = regions.loc[
            regions["image_id"] == image_id, ["region_id", "area_frac"]
        ].copy()
        if r.empty:
            return []

        rng = np.random.default_rng(self.seed)
        order = rng.permutation(len(r))
        r = r.iloc[order].reset_index(drop=True)

        cap = target + (self.area_match_tolerance_pp / 100.0)
        selected: list[str] = []
        used = 0.0
        for _, row in r.iterrows():
            af = float(row["area_frac"])
            if used + af > cap:
                # Skip — would exceed the area-match tolerance. Walk on.
                continue
            selected.append(str(row["region_id"]))
            used += af
        return selected


@dataclass
class LargestRegions:
    """§12.4 — pick largest regions until budget is full.

    Sort by area_frac desc, fit greedily within budget without overshoot.
    """

    name: str = "largest"
    intervention_type: str = "mean_mask"

    def select_regions(
        self,
        image_id: str,
        regions: pd.DataFrame,
        scores: Optional[pd.DataFrame],
        budget: dict,
    ) -> list[str]:
        target = float(budget.get("area_frac", 0.10))
        if target <= 0:
            return []
        r = regions.loc[
            regions["image_id"] == image_id, ["region_id", "area_frac"]
        ].sort_values("area_frac", ascending=False)
        if r.empty:
            return []

        selected: list[str] = []
        used = 0.0
        for _, row in r.iterrows():
            af = float(row["area_frac"])
            if used + af > target:
                # Don't overshoot; try the next smaller region.
                continue
            selected.append(str(row["region_id"]))
            used += af
            if used >= target * 0.99:
                break
        return selected
