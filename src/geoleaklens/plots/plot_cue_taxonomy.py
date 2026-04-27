"""§13.E3 cue-taxonomy bar plot.

Two-panel figure produced from `outputs/tables/cue_taxonomy.csv`:

  Top panel (PRIMARY per §13.E3):
    Bars of `mean_continuous_attribution` per bucket.
    Annotated with n_regions so the reader can tell "this bucket's
    leakage is real" from "this bucket only has 1 region in it".

  Bottom panel (companion):
    Bars of `frac_top5_dominant` per bucket — the share of top-5 ranked
    regions across images whose dominant bucket is this. Surfaces
    "where do the most-leaky regions live?" semantically.

The §13.E3 spec also wants a secondary-overlap panel and a qualitative
gallery; both are deferred to follow-up commits — this module sticks
to the primary bar deliverable.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot(table_path: Path, output_path: Path, *, top_k: int = 5) -> None:
    df = pd.read_csv(table_path)
    if df.empty:
        raise SystemExit(f"[plot] {table_path} is empty")

    # Drop buckets with zero regions to keep the bars from going negative-
    # zero or flat-zero clutter — but preserve `unknown` and `other` if
    # they have data, since the spec wants a complete view.
    plotted = df[df["n_regions"] > 0].copy()

    bucket_order = list(plotted["bucket"])

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), constrained_layout=True)
    fig.suptitle(
        "§13.E3 cue taxonomy — leakage attribution by semantic bucket\n"
        f"unnormalized mean continuous_attribution + top-{top_k} share, "
        "Im2GPS3k conditional set (n images = those baseline-success at Acc@25km)",
        fontsize=12,
    )

    # ---- Top panel: mean continuous_attribution -----------------------
    ax_attr = axes[0]
    bars = ax_attr.bar(
        bucket_order,
        plotted["mean_continuous_attribution"],
        color="#2c7fb8",
        edgecolor="black",
        linewidth=0.5,
    )
    ax_attr.set_ylabel("mean continuous_attribution\n(log1p(edited_err) - log1p(orig_err))")
    ax_attr.set_title(
        "PRIMARY: where leakage lives (unnormalized; min_across intervention)",
        fontsize=11,
    )
    ax_attr.tick_params(axis="x", rotation=45)
    for tick in ax_attr.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax_attr.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_attr.axhline(0, color="black", linewidth=0.5)

    # Annotate bars with n_regions
    for bar, n in zip(bars, plotted["n_regions"]):
        h = bar.get_height()
        ax_attr.text(
            bar.get_x() + bar.get_width() / 2,
            h + (max(plotted["mean_continuous_attribution"]) * 0.02
                 if max(plotted["mean_continuous_attribution"]) > 0 else 0.01),
            f"n={int(n)}",
            ha="center", va="bottom", fontsize=8,
        )

    # ---- Bottom panel: frac_top5_dominant -----------------------------
    ax_top = axes[1]
    col = f"frac_top{top_k}_dominant"
    if col not in plotted.columns:
        # Fall back to top1 if the table was generated with a different K.
        col = "frac_top1_dominant"
    ax_top.bar(
        bucket_order,
        plotted[col] * 100.0,
        color="#fdae61",
        edgecolor="black",
        linewidth=0.5,
    )
    ax_top.set_ylabel(f"% of top-{top_k} ranked regions\n(per image, by score_per_area)")
    ax_top.set_title(
        f"COMPANION: which buckets dominate the top-{top_k} per image",
        fontsize=11,
    )
    ax_top.tick_params(axis="x", rotation=45)
    for tick in ax_top.get_xticklabels():
        tick.set_horizontalalignment("right")
    ax_top.grid(True, axis="y", linestyle="--", alpha=0.4)
    ax_top.set_ylim(0, max(plotted[col].max() * 110, 5))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E3 cue-taxonomy bar plot.")
    ap.add_argument(
        "--table",
        default="outputs/tables/cue_taxonomy.csv",
    )
    ap.add_argument(
        "--out",
        default="outputs/figures/cue_taxonomy_bar.png",
    )
    ap.add_argument("--top-k", type=int, default=5)
    args = ap.parse_args()

    plot(Path(args.table), Path(args.out), top_k=args.top_k)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
