"""§11.6 paired-bootstrap analysis on the cached E2 v2 redactions parquet.

Closes the §17 MVP gap on criterion 7: "Sparsity curve showing GeoLeakLens
vs random/obvious baselines, **with paired-bootstrap CIs**."

Two outputs:

  1. Marginal CIs per (method, budget) — bootstraps the per-image
     Acc@25km-after-redaction across the conditional set. `statistic=mean`
     because Acc@25km is a 0/1 indicator; the §11.2 default of `median`
     collapses.

  2. Paired-difference CIs for each (target_method, budget) vs the
     reference method (default: random, area-matched per §12.3). At each
     (image, budget) we compute Δ = success_target - success_reference
     and bootstrap the mean Δ across images. The §17 acceptance check is
     "5pp drop AND CI excludes zero" at 5% budget; we evaluate it for
     every budget, not just 5%, to surface the full curve's significance.

Pure-Python; no Modal calls. Reads the cached redactions parquet, writes
the bootstrap CSV, and prints the §17 acceptance check.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from geoleaklens.scoring.bootstrap import (
    bootstrap_paired_difference,
    bootstrap_statistic,
)


def marginal_ci_per_cell(
    redactions: pd.DataFrame,
    *,
    n_bootstrap: int = 1000,
    seed: int = 123,
) -> pd.DataFrame:
    """Per (method, budget) marginal bootstrap CI of mean Acc@25km after
    redaction across the conditional set."""
    rows: list[dict] = []
    for (method, budget), group in redactions.groupby(
        ["method", "target_area_frac"]
    ):
        successes = group["edited_success_25km"].astype(float).to_numpy()
        result = bootstrap_statistic(
            successes,
            statistic="mean",
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        rows.append({
            "method": method,
            "target_area_frac": float(budget),
            "n": int(len(successes)),
            "mean_acc25km_after": float(result.point_estimate),
            "acc25km_after_ci_low": float(result.ci_low),
            "acc25km_after_ci_high": float(result.ci_high),
            "acc25km_after_se": float(result.se),
        })
    return pd.DataFrame(rows).sort_values(["method", "target_area_frac"])


def paired_difference_vs_reference(
    redactions: pd.DataFrame,
    reference_method: str,
    *,
    n_bootstrap: int = 1000,
    seed: int = 123,
) -> pd.DataFrame:
    """For each (target_method, budget), paired-bootstrap difference in
    Acc@25km-drop vs `reference_method` at the same budget.

    Convention: drop_method = baseline_acc - acc_after_method, where the
    conditional-set baseline is 1.0. So drop = 1 - success_after. We
    bootstrap mean(drop_target - drop_reference). Positive value = target
    beats reference at finding leaky regions.
    """
    rows: list[dict] = []
    methods = sorted(redactions["method"].unique())
    if reference_method not in methods:
        raise ValueError(
            f"reference method {reference_method!r} not in redactions; "
            f"available: {methods}"
        )

    for budget in sorted(redactions["target_area_frac"].unique()):
        b = float(budget)
        ref = redactions[
            (redactions["method"] == reference_method)
            & (redactions["target_area_frac"] == b)
        ].set_index("image_id")
        if ref.empty:
            continue

        for method in methods:
            if method == reference_method:
                continue
            tgt = redactions[
                (redactions["method"] == method)
                & (redactions["target_area_frac"] == b)
            ].set_index("image_id")
            shared_ids = ref.index.intersection(tgt.index)
            if len(shared_ids) == 0:
                continue
            tgt_drop = (
                1.0 - tgt.loc[shared_ids, "edited_success_25km"].astype(float)
            ).to_numpy()
            ref_drop = (
                1.0 - ref.loc[shared_ids, "edited_success_25km"].astype(float)
            ).to_numpy()
            result = bootstrap_paired_difference(
                tgt_drop, ref_drop,
                statistic="mean",
                n_bootstrap=n_bootstrap, seed=seed,
            )
            rows.append({
                "target_method": method,
                "reference_method": reference_method,
                "target_area_frac": b,
                "n_paired": int(len(shared_ids)),
                "mean_drop_diff_pp": float(result.point_estimate * 100),
                "drop_diff_ci_low_pp":  float(result.ci_low  * 100),
                "drop_diff_ci_high_pp": float(result.ci_high * 100),
                "se_pp": float(result.se * 100),
                "ci_excludes_zero": bool(result.ci_low > 0 or result.ci_high < 0),
            })
    return pd.DataFrame(rows).sort_values(["target_method", "target_area_frac"])


def mvp_acceptance_check(
    paired_summary: pd.DataFrame,
    *,
    target_method: str = "dynamic_greedy",
    reference_method: str = "random",
    target_budget: float = 0.05,
    min_diff_pp: float = 5.0,
) -> dict:
    """§17 MVP criterion: at 5% budget, target beats reference by ≥ 5pp
    drop with paired-bootstrap 95% CI of the difference excluding zero."""
    sel = paired_summary[
        (paired_summary["target_method"] == target_method)
        & (paired_summary["reference_method"] == reference_method)
        & np.isclose(paired_summary["target_area_frac"], target_budget)
    ]
    if sel.empty:
        return {
            "passed": False,
            "reason": (
                f"no row for {target_method} vs {reference_method} "
                f"at budget {target_budget:.0%}"
            ),
        }
    row = sel.iloc[0]
    diff = float(row["mean_drop_diff_pp"])
    ci_low = float(row["drop_diff_ci_low_pp"])
    ci_high = float(row["drop_diff_ci_high_pp"])
    big_enough = diff >= min_diff_pp
    excludes_zero = ci_low > 0 or ci_high < 0
    passed = bool(big_enough and excludes_zero)
    return {
        "passed": passed,
        "target_method": target_method,
        "reference_method": reference_method,
        "budget": target_budget,
        "mean_drop_diff_pp": diff,
        "ci_low_pp": ci_low,
        "ci_high_pp": ci_high,
        "min_diff_pp_required": min_diff_pp,
        "diff_clears_threshold": big_enough,
        "ci_excludes_zero": excludes_zero,
        "n_paired": int(row["n_paired"]),
    }


def main() -> None:
    ap = argparse.ArgumentParser(
        description="§17 paired-bootstrap CIs on the cached E2 sparsity curve."
    )
    ap.add_argument(
        "--redactions",
        default="data/processed/redactions/E2_im2gps3k_v2.parquet",
    )
    ap.add_argument(
        "--marginal-out",
        default="outputs/tables/e2_sparsity_bootstrap_marginal.csv",
    )
    ap.add_argument(
        "--paired-out",
        default="outputs/tables/e2_sparsity_bootstrap_paired.csv",
    )
    ap.add_argument("--reference-method", default="random")
    ap.add_argument("--n-bootstrap", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=123)
    args = ap.parse_args()

    redactions = pd.read_parquet(args.redactions)
    print(f"[bootstrap] {len(redactions)} rows, "
          f"{redactions['method'].nunique()} methods, "
          f"{redactions['target_area_frac'].nunique()} budgets, "
          f"{redactions['image_id'].nunique()} images")

    marginal = marginal_ci_per_cell(
        redactions, n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    Path(args.marginal_out).parent.mkdir(parents=True, exist_ok=True)
    marginal.to_csv(args.marginal_out, index=False, float_format="%.4f")
    print(f"[bootstrap] wrote {args.marginal_out}")

    paired = paired_difference_vs_reference(
        redactions, args.reference_method,
        n_bootstrap=args.n_bootstrap, seed=args.seed,
    )
    paired.to_csv(args.paired_out, index=False, float_format="%.4f")
    print(f"[bootstrap] wrote {args.paired_out}")

    print("\n=== marginal CIs (mean Acc@25km after redaction) ===")
    show = marginal.copy()
    show["mean_acc25km_after"] = show["mean_acc25km_after"].map("{:5.1%}".format)
    show["acc25km_after_ci_low"] = show["acc25km_after_ci_low"].map("{:5.1%}".format)
    show["acc25km_after_ci_high"] = show["acc25km_after_ci_high"].map("{:5.1%}".format)
    show["target_area_frac"] = show["target_area_frac"].map("{:5.1%}".format)
    print(show.to_string(index=False))

    print(f"\n=== paired-difference CIs vs {args.reference_method} "
          f"(target_drop − reference_drop, percentage points) ===")
    show2 = paired.copy()
    for c in ("mean_drop_diff_pp", "drop_diff_ci_low_pp",
              "drop_diff_ci_high_pp", "se_pp"):
        show2[c] = show2[c].map("{:+6.2f}".format)
    show2["target_area_frac"] = show2["target_area_frac"].map("{:5.1%}".format)
    print(show2.to_string(index=False))

    # §17 acceptance check at 5%, 10%, 20% for the headline pair.
    print("\n=== §17 MVP acceptance check (dynamic_greedy vs random) ===")
    for budget in (0.05, 0.10, 0.20):
        check = mvp_acceptance_check(
            paired,
            target_method="dynamic_greedy",
            reference_method=args.reference_method,
            target_budget=budget,
        )
        if "reason" in check:
            print(f"  budget={budget:>5.0%}: SKIPPED — {check['reason']}")
            continue
        verdict = "PASS" if check["passed"] else "FAIL"
        print(
            f"  budget={budget:>5.0%}: {verdict} | "
            f"diff={check['mean_drop_diff_pp']:+6.2f}pp, "
            f"95% CI [{check['ci_low_pp']:+6.2f}, {check['ci_high_pp']:+6.2f}], "
            f"clears {check['min_diff_pp_required']:.0f}pp = "
            f"{check['diff_clears_threshold']}, "
            f"CI excludes 0 = {check['ci_excludes_zero']}, "
            f"n={check['n_paired']}"
        )


if __name__ == "__main__":
    main()
