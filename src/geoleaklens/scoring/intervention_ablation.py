"""§13.E6 intervention-type ablation analyzer.

Pure-Python analysis on a §7.5 leakage-scores parquet that has rows for
multiple intervention types per (image_id, region_id). Computes:

  - top-1 agreement      % of images where the rank-1 region by
                          score_per_area is the SAME under both
                          intervention types
  - top-k Jaccard        per-image Jaccard similarity of the top-k
                          region sets, averaged across images
  - Spearman correlation per-image rank correlation across all regions,
                          averaged across images

Plus per-intervention redaction performance (Acc@25km after applying the
intervention's static-greedy pick at fixed budget) — though that needs
the redactions parquet, so the helper there is optional.

No models, no Modal, no plots. This module is the §13.E6 metric core
that the runner / plot tools sit on top of.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable

import numpy as np
import pandas as pd


def _spearman_corr(a: np.ndarray, b: np.ndarray) -> float:
    """Spearman rank correlation. NaN-safe; returns NaN when either input
    has variance zero (no rank variation to correlate)."""
    if len(a) != len(b) or len(a) < 2:
        return float("nan")
    ra = pd.Series(a).rank(method="average").to_numpy()
    rb = pd.Series(b).rank(method="average").to_numpy()
    if np.std(ra) == 0 or np.std(rb) == 0:
        return float("nan")
    cov = np.mean((ra - ra.mean()) * (rb - rb.mean()))
    denom = ra.std() * rb.std()
    if denom == 0:
        return float("nan")
    return float(cov / denom)


def per_image_agreement(
    scores: pd.DataFrame,
    intervention_a: str,
    intervention_b: str,
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """Return a per-image table of {top1_match, topk_jaccard, spearman}.

    `scores` is the §7.5 schema with columns image_id, region_id,
    intervention_type, score_per_area. Both `intervention_a` and
    `intervention_b` must appear in `intervention_type`. Images that
    don't have rows for *both* interventions are excluded.
    """
    a = scores.loc[
        scores["intervention_type"] == intervention_a,
        ["image_id", "region_id", "score_per_area"],
    ].rename(columns={"score_per_area": "spa_a"})
    b = scores.loc[
        scores["intervention_type"] == intervention_b,
        ["image_id", "region_id", "score_per_area"],
    ].rename(columns={"score_per_area": "spa_b"})
    merged = a.merge(b, on=["image_id", "region_id"], how="inner")

    rows: list[dict] = []
    for image_id, group in merged.groupby("image_id"):
        if len(group) < 2:
            continue
        spa_a = group["spa_a"].to_numpy(dtype=np.float64)
        spa_b = group["spa_b"].to_numpy(dtype=np.float64)
        order_a = np.argsort(-spa_a)
        order_b = np.argsort(-spa_b)
        rids = group["region_id"].to_numpy()
        top_a = set(rids[order_a[:1]])
        top_b = set(rids[order_b[:1]])
        topk_a = set(rids[order_a[:top_k]])
        topk_b = set(rids[order_b[:top_k]])
        union = topk_a | topk_b
        jaccard = (
            len(topk_a & topk_b) / len(union) if union else float("nan")
        )
        rows.append({
            "image_id": image_id,
            "intervention_a": intervention_a,
            "intervention_b": intervention_b,
            "n_regions": int(len(group)),
            "top1_match": bool(top_a == top_b),
            f"top{top_k}_jaccard": float(jaccard),
            "spearman": _spearman_corr(spa_a, spa_b),
        })
    return pd.DataFrame(rows)


def pairwise_agreement_summary(
    scores: pd.DataFrame,
    intervention_types: Iterable[str],
    *,
    top_k: int = 5,
) -> pd.DataFrame:
    """One row per unordered intervention pair, aggregating per-image
    metrics into mean across images.

    `intervention_types` is the set the spec asks to compare — typically
    {mean_mask, blur, inpaint}. min_across is allowed too if you want to
    see how it correlates with each base intervention.
    """
    intervention_types = list(intervention_types)
    out_rows: list[dict] = []
    for a, b in combinations(intervention_types, 2):
        per_image = per_image_agreement(scores, a, b, top_k=top_k)
        if per_image.empty:
            continue
        out_rows.append({
            "intervention_a": a,
            "intervention_b": b,
            "n_images": int(len(per_image)),
            "top1_match_rate": float(per_image["top1_match"].mean()),
            f"mean_top{top_k}_jaccard": float(
                per_image[f"top{top_k}_jaccard"].mean()
            ),
            f"median_top{top_k}_jaccard": float(
                per_image[f"top{top_k}_jaccard"].median()
            ),
            "mean_spearman": float(per_image["spearman"].dropna().mean()),
        })
    return pd.DataFrame(out_rows)
