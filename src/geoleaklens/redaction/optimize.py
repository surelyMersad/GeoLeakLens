"""§12.2 dynamic greedy — the primary headline redaction method.

Why this is in `optimize.py` and not `methods.py`
--------------------------------------------------
StaticGreedy and the §12.3/§12.4 baselines run on cached single-region
attribution scores — they don't need the model during selection. Dynamic
greedy is fundamentally different: at every pick it queries the model
(GeoCLIP, VLM, ...) on the running edited image. That changes the
interface — selection here needs callbacks for image scoring,
intervention application, and mask loading. Keeping it in a separate
module makes the interface split obvious and matches the §3 layout
(`redaction/optimize.py` for joint-causal selectors, `redaction/methods.py`
for score-based ones).

Algorithm (matches §12.2 closely; deviations called out in comments)
---------------------------------------------------------------------
1. Filter `scores` to the configured intervention_type and inner-join with
   `regions` to get area_frac and mask_path. Sort by score_per_area desc;
   take the top-K (default K=30 per spec) as the candidate shortlist.
2. State: `selected = []`, `used_area = 0.0`, `current_error_km = original_error_km`.
3. While there's at least one candidate that fits in the remaining budget:
     a. For each remaining candidate r, paint the union of (selected ∪ {r})
        on the *original* image with the configured intervention, run
        score_image to get edit_lat/lon, compute edit_error_km via
        haversine, and Δ_r = edit_error_km - current_error_km.
     b. Pick r* = argmax_r Δ_r over candidates whose addition keeps
        used_area within max(budgets) + 0.005pp tolerance.
     c. If Δ_r* ≤ 0, stop — no remaining candidate increases joint error.
     d. Append r*, update used_area + current_error_km, record state at
        any budget thresholds we just crossed.
     e. Stop-early check: if log1p marginal increment < `early_stop_delta`
        for `early_stop_consecutive` picks in a row, break.
4. For each requested budget, return the selection state captured at (or
   nearest below) that budget.

Deviation from spec §12.2 step 6 ("Recompute scores for the remaining
candidates on x_current"): the joint-effect computation in step 3a above
is *itself* the recomputation — Δ_r naturally accounts for what's already
in the selected set. No separate per-candidate single-region rescoring
pass.

Cost / concurrency
------------------
Per (image, full budget sweep): top_k × n_picks GeoCLIP calls.
Default K=30, ~10 picks → 300 calls/image.

The `score_images` callback is **batched** — it takes a list of trial
images and returns a list of (lat, lon) tuples. The runner implements
this via `geoclip.predict.map(bytes_list)` so all K candidate
evaluations at one iteration step fan out to Modal in parallel. With
Modal's default container scaling (~100 concurrent), one step's
latency drops from K * roundtrip to ~1 roundtrip + cold-spin.
Sequential ~4h E2 run → ~30 min concurrent.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.data.geo_utils import (
    MAX_ERROR_KM,
    clip_error_km,
    haversine_km,
)


# Tolerance band around budget thresholds when recording state and pruning
# oversized candidates. 0.005 = 0.5pp matches the §12.3 audit tolerance.
_AREA_TOLERANCE_PP = 0.005


@dataclass
class BudgetSnapshot:
    """One row of the dynamic-greedy log for one (image, budget) pair."""

    selected_region_ids: list[str] = field(default_factory=list)
    actual_area_frac: float = 0.0
    edited_error_km: float = 0.0
    edited_lat: Optional[float] = None
    edited_lon: Optional[float] = None


@dataclass
class DynamicGreedyResult:
    """Per-image dynamic-greedy run, with a snapshot at each requested budget."""

    image_id: str
    snapshots: dict  # {budget: BudgetSnapshot}
    n_picks: int
    stopped_early: bool


@dataclass
class DynamicGreedy:
    """§12.2 dynamic greedy — primary headline redaction method.

    `top_k` caps the candidate shortlist per the spec (default 30). A larger
    K barely changes the selection in pilot runs but linearly inflates GPU
    cost. `early_stop_delta` and `early_stop_consecutive` implement the
    §12.2 step 5 stopping rule (stop when marginal log-error increase falls
    below threshold for two iterations in a row).
    """

    name: str = "dynamic_greedy"
    intervention_type: str = "mean_mask"
    top_k: int = 30
    early_stop_delta: float = 0.05
    early_stop_consecutive: int = 2

    def run_image(
        self,
        *,
        image_id: str,
        regions: pd.DataFrame,        # filter to this image OK either way
        scores: pd.DataFrame,         # filter to this image OK either way
        budgets: list[float],
        original_image: Image.Image,
        true_lat: float,
        true_lon: float,
        original_error_km: float,
        load_mask: Callable[[str], np.ndarray],
        apply_intervention: Callable[[Image.Image, np.ndarray], Image.Image],
        score_images: Callable[
            [list[Image.Image]],
            list[tuple[Optional[float], Optional[float]]],
        ],
    ) -> DynamicGreedyResult:
        """Run dynamic greedy on one image, capturing per-budget snapshots.

        `score_images` is **batched**: takes a list of trial images and returns
        a same-length list of (lat, lon) tuples. The runner implementation
        uses Modal `.map()` so the K candidate scorings at one step fan out
        in parallel.
        """
        budgets = sorted({float(b) for b in budgets if b > 0})
        if not budgets:
            return DynamicGreedyResult(
                image_id=image_id, snapshots={}, n_picks=0, stopped_early=False
            )
        max_budget = max(budgets)

        # 1. Initial top-K candidates by score_per_area on the original image.
        sub_scores = scores.loc[
            (scores["image_id"] == image_id)
            & (scores["intervention_type"] == self.intervention_type),
            ["region_id", "score_per_area"],
        ]
        sub_regions = regions.loc[
            regions["image_id"] == image_id,
            ["region_id", "area_frac", "mask_path"],
        ]
        merged = sub_scores.merge(sub_regions, on="region_id", how="inner")
        if merged.empty:
            empty_snap = BudgetSnapshot(
                edited_error_km=original_error_km
            )
            return DynamicGreedyResult(
                image_id=image_id,
                snapshots={b: empty_snap for b in budgets},
                n_picks=0,
                stopped_early=False,
            )
        merged = merged.sort_values("score_per_area", ascending=False).head(self.top_k)

        candidate_ids: list[str] = list(merged["region_id"])
        candidate_areas: dict[str, float] = dict(
            zip(merged["region_id"], merged["area_frac"].astype(float))
        )
        candidate_masks: dict[str, np.ndarray] = {}
        for rid, mask_path in zip(merged["region_id"], merged["mask_path"]):
            candidate_masks[rid] = load_mask(str(mask_path))

        # Reference shape from one mask — all of an image's masks share dims.
        h, w = next(iter(candidate_masks.values())).shape

        # 2. Iterate picks.
        selected: list[str] = []
        used_area = 0.0
        current_error_km = float(original_error_km)
        consecutive_small = 0
        n_picks = 0
        stopped_early = False
        snapshots: dict[float, BudgetSnapshot] = {}

        def _union_for(extra: Optional[str] = None) -> np.ndarray:
            out = np.zeros((h, w), dtype=bool)
            for rid in selected:
                out |= candidate_masks[rid]
            if extra is not None:
                out |= candidate_masks[extra]
            return out

        def _record_threshold_crossings(latest: BudgetSnapshot) -> None:
            for b in budgets:
                if b in snapshots:
                    continue
                if used_area >= b - _AREA_TOLERANCE_PP:
                    snapshots[b] = BudgetSnapshot(
                        selected_region_ids=list(selected),
                        actual_area_frac=used_area,
                        edited_error_km=latest.edited_error_km,
                        edited_lat=latest.edited_lat,
                        edited_lon=latest.edited_lon,
                    )

        latest_snap = BudgetSnapshot(
            edited_error_km=original_error_km, edited_lat=None, edited_lon=None
        )

        while True:
            remaining = [c for c in candidate_ids if c not in selected]
            if not remaining:
                break

            # Filter to the candidates that still fit in the loosest budget.
            # We do this BEFORE the model call so we don't waste fan-out
            # on regions that can never be picked at any budget level.
            fitting = [
                r for r in remaining
                if used_area + candidate_areas[r]
                   <= max_budget + _AREA_TOLERANCE_PP
            ]
            if not fitting:
                break

            # Build all trial images locally (CPU-bound mask union +
            # intervention application). This is fast for mean_mask / blur;
            # inpaint-as-edit is a Modal call per region — that path isn't
            # batched yet.
            trial_images: list[Image.Image] = [
                apply_intervention(original_image, _union_for(extra=r))
                for r in fitting
            ]

            # Single fan-out: the runner's `score_images` is `predict.map(...)`
            # so all K trial images get scored in parallel by Modal.
            batch_results = score_images(trial_images)
            if len(batch_results) != len(fitting):
                raise RuntimeError(
                    f"score_images returned {len(batch_results)} results for "
                    f"{len(fitting)} inputs — batch size mismatch"
                )

            best_r: Optional[str] = None
            best_delta = -math.inf
            best_lat: Optional[float] = None
            best_lon: Optional[float] = None
            best_err: Optional[float] = None
            for r, (edit_lat, edit_lon) in zip(fitting, batch_results):
                trial_err = haversine_km(edit_lat, edit_lon, true_lat, true_lon)
                trial_err_clipped = clip_error_km(trial_err)
                delta = trial_err_clipped - current_error_km
                if delta > best_delta:
                    best_delta = delta
                    best_r = r
                    best_lat = edit_lat
                    best_lon = edit_lon
                    best_err = trial_err_clipped

            if best_r is None:
                # Nothing fit in budget; we're done.
                break
            if best_delta <= 0.0:
                # No remaining region helps — stop rather than waste budget.
                break

            # Commit pick.
            prev_error = current_error_km
            selected.append(best_r)
            used_area += candidate_areas[best_r]
            current_error_km = best_err if best_err is not None else current_error_km
            n_picks += 1
            latest_snap = BudgetSnapshot(
                selected_region_ids=list(selected),
                actual_area_frac=used_area,
                edited_error_km=current_error_km,
                edited_lat=best_lat,
                edited_lon=best_lon,
            )

            # Stop-early on log1p marginal.
            log_marginal = (
                math.log1p(current_error_km) - math.log1p(prev_error)
            )
            if log_marginal < self.early_stop_delta:
                consecutive_small += 1
            else:
                consecutive_small = 0
            if consecutive_small >= self.early_stop_consecutive:
                stopped_early = True
                _record_threshold_crossings(latest_snap)
                break

            _record_threshold_crossings(latest_snap)

            if used_area >= max_budget - _AREA_TOLERANCE_PP:
                # We've hit the max budget; no need to keep iterating.
                break

        # 3. Fill any unrecorded budgets with the final state.
        for b in budgets:
            if b not in snapshots:
                snapshots[b] = BudgetSnapshot(
                    selected_region_ids=list(selected),
                    actual_area_frac=used_area,
                    edited_error_km=current_error_km,
                    edited_lat=latest_snap.edited_lat,
                    edited_lon=latest_snap.edited_lon,
                )

        return DynamicGreedyResult(
            image_id=image_id,
            snapshots=snapshots,
            n_picks=n_picks,
            stopped_early=stopped_early,
        )
