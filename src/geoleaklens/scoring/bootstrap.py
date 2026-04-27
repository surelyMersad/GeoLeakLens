"""§11.6 paired bootstrap.

For each (model × dataset) cell of the §13.E1 baseline table, we resample
image_ids with replacement, recompute the median error on each resample,
and report the 95% percentile CI plus the standard error.

For method-vs-method comparisons (E2, E4) the bootstrap is *paired*: every
resample uses the same image_ids for both methods, so the CI reflects
within-image variation — the right paired comparison.

`stratify_by` per §11.6: when ≥30 images per stratum, sample with
replacement *within each stratum* and then aggregate. Otherwise fall back
to unstratified sampling. The default stratification key on the project
is `country_iso` (§11.6).

This module is pure NumPy. Callers feed in plain Python lists or arrays.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional, Sequence

import numpy as np


DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_SEED = 123
DEFAULT_STRATIFY_MIN_PER_STRATUM = 30

Statistic = Literal["median", "mean"]


def _apply_statistic(samples: np.ndarray, statistic: Statistic) -> np.ndarray:
    """Apply the requested statistic across rows. samples shape (B, n)."""
    if statistic == "median":
        return np.median(samples, axis=1)
    if statistic == "mean":
        return np.mean(samples, axis=1)
    raise ValueError(f"unknown statistic: {statistic!r}")


def _point_estimate(values: np.ndarray, statistic: Statistic) -> float:
    if statistic == "median":
        return float(np.median(values))
    if statistic == "mean":
        return float(np.mean(values))
    raise ValueError(f"unknown statistic: {statistic!r}")


@dataclass
class BootstrapResult:
    point_estimate: float
    se: float
    ci_low: float
    ci_high: float
    n_bootstrap: int
    n_samples: int
    stratified: bool


def _sample_indices_unstratified(
    n: int, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Return a (n_bootstrap, n) int array of indices into [0, n)."""
    return rng.integers(0, n, size=(n_bootstrap, n))


def _sample_indices_stratified(
    strata: np.ndarray, n_bootstrap: int, rng: np.random.Generator
) -> np.ndarray:
    """Stratified sampling: each row of the output is a permutation-with-
    replacement that draws within each stratum so per-stratum counts match
    the input."""
    n = len(strata)
    out = np.empty((n_bootstrap, n), dtype=np.int64)
    # Group indices by stratum once, sample within each group per bootstrap.
    unique = np.unique(strata)
    groups = {s: np.where(strata == s)[0] for s in unique}
    for b in range(n_bootstrap):
        chunks = []
        for s in unique:
            idx = groups[s]
            chosen = rng.integers(0, len(idx), size=len(idx))
            chunks.append(idx[chosen])
        # Concatenate but preserve the original index order's count alignment;
        # the actual index ordering within a bootstrap row doesn't matter for
        # median/mean aggregation.
        out[b] = np.concatenate(chunks)
    return out


def bootstrap_statistic(
    values: Sequence[float],
    *,
    statistic: Statistic = "median",
    strata: Optional[Sequence] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    stratify_min_per_stratum: int = DEFAULT_STRATIFY_MIN_PER_STRATUM,
    ci: float = 0.95,
) -> BootstrapResult:
    """Bootstrap CI for an arbitrary statistic (median or mean) of `values`.

    `statistic="mean"` is needed for Acc@25km bootstraps (0/1 indicator
    where median collapses to 0 or 1). `statistic="median"` is the §11.2
    default for geodesic error.

    Stratifies when `strata` is given and every stratum has at least
    `stratify_min_per_stratum` samples (§11.6 stratification rule).

    Returns the percentile-CI variant; bias-corrected accelerated (BCa)
    is listed in default.yaml as a follow-up — implement when needed.
    """
    arr = np.asarray(values, dtype=np.float64)
    n = len(arr)
    if n == 0:
        return BootstrapResult(
            point_estimate=float("nan"),
            se=float("nan"),
            ci_low=float("nan"),
            ci_high=float("nan"),
            n_bootstrap=n_bootstrap,
            n_samples=0,
            stratified=False,
        )

    rng = np.random.default_rng(seed)

    use_strata = False
    if strata is not None:
        s = np.asarray(strata)
        if len(s) != n:
            raise ValueError("strata length must match values length")
        unique, counts = np.unique(s, return_counts=True)
        if (counts >= stratify_min_per_stratum).all() and len(unique) > 1:
            use_strata = True

    if use_strata:
        idx = _sample_indices_stratified(np.asarray(strata), n_bootstrap, rng)
    else:
        idx = _sample_indices_unstratified(n, n_bootstrap, rng)

    samples = arr[idx]                                  # shape (B, n)
    stat = _apply_statistic(samples, statistic)         # shape (B,)

    point = _point_estimate(arr, statistic)
    alpha = (1.0 - ci) / 2.0
    lo = float(np.quantile(stat, alpha))
    hi = float(np.quantile(stat, 1.0 - alpha))
    se = float(np.std(stat, ddof=1)) if len(stat) > 1 else 0.0

    return BootstrapResult(
        point_estimate=point,
        se=se,
        ci_low=lo,
        ci_high=hi,
        n_bootstrap=n_bootstrap,
        n_samples=n,
        stratified=use_strata,
    )


def bootstrap_median(values: Sequence[float], **kwargs) -> BootstrapResult:
    """Backward-compatible alias for the median variant."""
    return bootstrap_statistic(values, statistic="median", **kwargs)


def bootstrap_mean(values: Sequence[float], **kwargs) -> BootstrapResult:
    """Convenience alias for the mean variant — needed for 0/1 indicator
    bootstraps (Acc@25km, parse rate, etc.) where median collapses."""
    return bootstrap_statistic(values, statistic="mean", **kwargs)


def bootstrap_paired_difference(
    values_a: Sequence[float],
    values_b: Sequence[float],
    *,
    statistic: Statistic = "median",
    strata: Optional[Sequence] = None,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    seed: int = DEFAULT_SEED,
    stratify_min_per_stratum: int = DEFAULT_STRATIFY_MIN_PER_STRATUM,
    ci: float = 0.95,
) -> BootstrapResult:
    """Paired bootstrap of `statistic(a - b)` over matching indices.

    For E2 / E4 method comparisons where `values_a[i]` and `values_b[i]`
    are the same image scored under two methods, this is the right
    comparison: it captures within-image variation rather than treating
    the two samples as independent.

    `statistic="mean"` is the right choice for 0/1 indicator data
    (Acc@25km drop) where median collapses. `statistic="median"` is the
    §11.2 default for continuous error data.
    """
    a = np.asarray(values_a, dtype=np.float64)
    b = np.asarray(values_b, dtype=np.float64)
    if len(a) != len(b):
        raise ValueError("paired bootstrap requires equal-length inputs")
    diffs = a - b
    return bootstrap_statistic(
        diffs,
        statistic=statistic,
        strata=strata,
        n_bootstrap=n_bootstrap,
        seed=seed,
        stratify_min_per_stratum=stratify_min_per_stratum,
        ci=ci,
    )
