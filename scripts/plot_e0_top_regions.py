"""Quick visualization of E0 single-region attribution.

For each of the 10 wikimedia smoke-set images, paints up to 5 SAM regions on
the original image, colored by `continuous_attribution` (Δlog). Red = removing
the region pushed the prediction further from ground truth; blue = it pulled
the prediction closer; gray = no effect.

Output: outputs/figures/e0_top_regions.png

This isn't part of the §13.E0 acceptance contact sheet (that's §10.5 +
plots/make_qualitative_figures.py — bigger scope); it's a sanity-check
inspection figure.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SCORES = ROOT / "data/processed/metrics/leakage_scores_E0_smoke.parquet"
REGIONS = ROOT / "data/processed/regions/wikimedia_smoke_sam.parquet"
MANIFEST = ROOT / "data/processed/manifests/wikimedia_smoke_mvp.parquet"
OUT = ROOT / "outputs/figures/e0_top_regions.png"

TOP_K = 5
EPS = 1e-3  # below this |Δlog|, don't bother drawing the region

# Diverging colormap: red for positive Δlog (real leakage), blue for negative
# (edit helped the model).
CMAP = plt.get_cmap("RdBu_r")


def _load_mask(mask_path: Path) -> np.ndarray:
    """Inverse of run_e0_smoke._save_mask_npz."""
    npz = np.load(mask_path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def _overlay_rgba(mask: np.ndarray, rgba: tuple[float, float, float, float]) -> np.ndarray:
    """Build a (H, W, 4) RGBA array with `rgba` inside `mask`, transparent outside."""
    h, w = mask.shape
    out = np.zeros((h, w, 4), dtype=np.float32)
    out[mask] = rgba
    return out


def main() -> None:
    OUT.parent.mkdir(parents=True, exist_ok=True)

    scores = pd.read_parquet(SCORES)
    regions = pd.read_parquet(REGIONS)
    manifest = pd.read_parquet(MANIFEST)

    df = scores.merge(
        regions[["image_id", "region_id", "mask_path"]],
        on=["image_id", "region_id"],
        how="left",
    )
    df = df.merge(
        manifest[["image_id", "image_path", "city", "country", "width", "height"]],
        on="image_id",
        how="left",
    )

    image_ids = sorted(df["image_id"].unique())
    n = len(image_ids)
    assert n == 10, f"expected 10 wikimedia smoke images, got {n}"

    # Color scale: symmetric around 0, using the largest |attribution| seen.
    vmax = max(EPS, df["continuous_attribution"].abs().max())
    norm = Normalize(vmin=-vmax, vmax=+vmax)

    fig, axes = plt.subplots(2, 5, figsize=(22, 10), constrained_layout=True)
    fig.suptitle(
        "E0 single-region attribution (Δlog of geodesic error after mean_mask)\n"
        "Top-5 regions per image by |Δlog|.  red = leakage, blue = helped, gray = no effect",
        fontsize=14,
    )

    for ax, image_id in zip(axes.flat, image_ids):
        sub = df[df["image_id"] == image_id]
        image_path = ROOT / sub["image_path"].iloc[0]
        city = sub["city"].iloc[0]
        country = sub["country"].iloc[0]
        orig_err = float(sub["original_error_km"].iloc[0])

        img = np.asarray(Image.open(image_path).convert("RGB"))
        ax.imshow(img)
        ax.set_axis_off()
        ax.set_title(
            f"{city}, {country}\norig_err={orig_err:.1f} km",
            fontsize=10,
        )

        # Pick top-K regions by |Δlog|, with a minimum-effect cutoff so we don't
        # paint a panel solid gray when nothing happened.
        top = sub.assign(abs_attr=sub["continuous_attribution"].abs()) \
                 .sort_values("abs_attr", ascending=False) \
                 .head(TOP_K)

        meaningful = (top["abs_attr"] > EPS).sum()
        if meaningful == 0:
            ax.text(
                0.5, 0.05,
                "no region moved the prediction",
                transform=ax.transAxes,
                ha="center", va="bottom",
                fontsize=9, color="white",
                bbox=dict(facecolor="black", alpha=0.6, pad=3),
            )
            continue

        for _, r in top.iterrows():
            attr = float(r["continuous_attribution"])
            if abs(attr) <= EPS:
                continue
            mask_path = ROOT / r["mask_path"]
            if not mask_path.exists():
                continue
            mask = _load_mask(mask_path)
            color = CMAP(norm(attr))  # rgba in [0,1]
            overlay = _overlay_rgba(mask, (*color[:3], 0.55))
            ax.imshow(overlay)

            # Annotate at the mask centroid (more accurate than bbox center
            # when masks are weirdly shaped).
            ys, xs = np.where(mask)
            if xs.size == 0:
                continue
            cx, cy = float(xs.mean()), float(ys.mean())
            sign = "+" if attr >= 0 else "-"
            ax.text(
                cx, cy,
                f"Δ={sign}{abs(attr):.2f}",
                fontsize=9, fontweight="bold",
                ha="center", va="center",
                color="white",
                bbox=dict(facecolor="black", alpha=0.65, pad=1.5, edgecolor="none"),
            )

    # Shared colorbar across all panels.
    sm = ScalarMappable(norm=norm, cmap=CMAP)
    sm.set_array([])
    cbar = fig.colorbar(
        sm, ax=axes, orientation="horizontal",
        fraction=0.025, pad=0.02, aspect=60,
    )
    cbar.set_label("continuous_attribution = log1p(edit_err) - log1p(orig_err)")

    fig.savefig(OUT, dpi=140, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
