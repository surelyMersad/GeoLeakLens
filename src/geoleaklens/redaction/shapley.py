"""§12.11 Shapley-style validation of single-region attribution.

The §12.11 spec asks: does static greedy's region ranking roughly agree
with the joint-causal Shapley ranking? If yes, single-region attribution
is a defensible fast approximation; if no, only Shapley / dynamic greedy
support the paper's causal claim.

Algorithm
---------
For each image's top-K candidate regions:
  1. Sample N random orderings (default 200) of the K regions.
  2. For each ordering, walk prefixes 0..K and compute the post-edit
     error for each prefix's region set.
  3. Region r's marginal contribution in an ordering is
       err(prefix_with_r) - err(prefix_without_r)
  4. Approximate Shapley(r) = mean of marginals across orderings.

Caching
-------
Many orderings share prefixes. We collect the *set* of unique
region_sets across all (ordering × prefix-length) pairs, score each
once, and re-use. With K=10 regions and N=200 orderings, the worst
case is ~1023 unique region sets per image (powerset-bounded), but in
practice we hit fewer because orderings repeat prefixes.

Pure orchestration — accepts callbacks for `apply_intervention` and
`score_image` so the same code works with mocks (tests) or live
GeoCLIP (runner). Same pattern as DynamicGreedy.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Callable, Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image


@dataclass
class ShapleyImageResult:
    image_id: str
    region_ids: list[str]                  # top-K candidates, in input order
    shapley_values: dict[str, float]       # region_id -> Shapley value
    n_orderings: int
    n_unique_sets_scored: int


def _union_for_set(
    region_set: frozenset[str],
    masks: dict[str, np.ndarray],
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    for rid in region_set:
        m = masks.get(rid)
        if m is not None and m.shape == shape:
            out |= m
    return out


def shapley_spot_check_image(
    *,
    image_id: str,
    region_ids: list[str],          # top-K candidates ordered any way
    region_masks: dict[str, np.ndarray],
    original_image: Image.Image,
    original_error_km: float,
    apply_intervention: Callable[[Image.Image, np.ndarray], Image.Image],
    score_image: Callable[
        [list[Image.Image]],
        list[tuple[Optional[float], Optional[float]]],
    ],
    error_for_pred: Callable[
        [Optional[float], Optional[float]], float
    ],
    n_orderings: int = 200,
    seed: int = 42,
) -> ShapleyImageResult:
    """Approximate Shapley values for `region_ids` on this image.

    `score_image` is the BATCH callback: takes a list of edited PIL
    images, returns list of (lat, lon) — same contract as DynamicGreedy.
    `error_for_pred(lat, lon) -> error_km` lets the caller decide how
    to compute the error (we don't hardcode haversine here so tests can
    use a synthetic scoring scheme).

    Returns Shapley values per region. Higher = removing this region
    contributes more to error increase (i.e. the region is "leakier").
    """
    rng = np.random.default_rng(seed)
    K = len(region_ids)
    if K == 0:
        return ShapleyImageResult(
            image_id=image_id,
            region_ids=[],
            shapley_values={},
            n_orderings=0,
            n_unique_sets_scored=0,
        )

    # 1. Sample N random orderings (each is a permutation of indices 0..K-1).
    orderings: list[list[int]] = []
    for _ in range(n_orderings):
        orderings.append(rng.permutation(K).tolist())

    # 2. Collect unique region sets needed (all prefixes of all orderings).
    unique_sets: set[frozenset[str]] = {frozenset()}  # empty = original image
    for order in orderings:
        for k in range(1, K + 1):
            unique_sets.add(frozenset(region_ids[i] for i in order[:k]))

    # 3. Score each unique set in batches (the runner wraps `score_image`
    #    around Modal `.map()` for parallelism).
    sets_list = sorted(unique_sets, key=lambda s: (len(s), tuple(sorted(s))))
    # Build edited images for the non-empty sets. Empty set = original
    # error (already known).
    h = original_image.size[1]
    w = original_image.size[0]
    nonempty = [s for s in sets_list if s]
    trial_images: list[Image.Image] = []
    for s in nonempty:
        union = _union_for_set(s, region_masks, (h, w))
        trial_images.append(apply_intervention(original_image, union))

    # Batch score.
    if trial_images:
        results = score_image(trial_images)
    else:
        results = []

    set_to_error: dict[frozenset[str], float] = {frozenset(): float(original_error_km)}
    for s, (lat, lon) in zip(nonempty, results):
        set_to_error[s] = error_for_pred(lat, lon)

    # 4. Compute Shapley values.
    shapley: dict[str, float] = {rid: 0.0 for rid in region_ids}
    for order in orderings:
        for k in range(K):
            prev_set = frozenset(region_ids[i] for i in order[:k])
            new_set = frozenset(region_ids[i] for i in order[:k + 1])
            marginal = set_to_error[new_set] - set_to_error[prev_set]
            shapley[region_ids[order[k]]] += marginal
    for rid in shapley:
        shapley[rid] /= float(n_orderings)

    return ShapleyImageResult(
        image_id=image_id,
        region_ids=region_ids,
        shapley_values=shapley,
        n_orderings=n_orderings,
        n_unique_sets_scored=len(unique_sets),
    )


def static_vs_shapley_agreement(
    static_scores: pd.DataFrame,
    shapley_results: list[ShapleyImageResult],
    *,
    intervention_type: str = "min_across",
    top_k: int = 5,
) -> dict:
    """Aggregate top-1 / top-K agreement across images.

    `static_scores` is a §7.5-shaped DataFrame; we filter to
    `intervention_type` and rank by score_per_area desc.
    """
    if not shapley_results:
        return {
            "n_images": 0,
            "top1_agreement": float("nan"),
            f"top{top_k}_jaccard": float("nan"),
            "mean_spearman": float("nan"),
        }

    n_top1 = 0
    jaccards: list[float] = []
    spearmans: list[float] = []
    for res in shapley_results:
        sub = static_scores[
            (static_scores["image_id"] == res.image_id)
            & (static_scores["intervention_type"] == intervention_type)
            & (static_scores["region_id"].isin(res.region_ids))
        ]
        if sub.empty or not res.shapley_values:
            continue

        # Static rank by score_per_area desc.
        static_order = (
            sub.sort_values("score_per_area", ascending=False)["region_id"].tolist()
        )
        # Shapley rank by value desc.
        shapley_order = sorted(
            res.shapley_values.keys(),
            key=lambda r: res.shapley_values[r],
            reverse=True,
        )

        if static_order and shapley_order and static_order[0] == shapley_order[0]:
            n_top1 += 1

        s_top = set(static_order[:top_k])
        sh_top = set(shapley_order[:top_k])
        union = s_top | sh_top
        jaccards.append(len(s_top & sh_top) / len(union) if union else float("nan"))

        # Spearman over the regions both sides ranked. Implemented as
        # Pearson on ranks (numerically equivalent and avoids the
        # sample-vs-population-variance mismatch a hand-rolled formula
        # invites).
        common = list(set(static_order) & set(shapley_order))
        if len(common) >= 2:
            srank = pd.Series([static_order.index(r) for r in common]).rank().to_numpy()
            shrank = pd.Series([shapley_order.index(r) for r in common]).rank().to_numpy()
            if np.std(srank) > 0 and np.std(shrank) > 0:
                rho = float(np.corrcoef(srank, shrank)[0, 1])
                spearmans.append(rho)

    n = len(shapley_results)
    return {
        "n_images": n,
        "top1_agreement": n_top1 / n if n else float("nan"),
        f"top{top_k}_jaccard": float(np.nanmean(jaccards)) if jaccards else float("nan"),
        "mean_spearman": float(np.nanmean(spearmans)) if spearmans else float("nan"),
    }
