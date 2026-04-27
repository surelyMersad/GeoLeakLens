"""§13.E4 privacy-utility Pareto runner — Panel A only (visible redaction).

Reuses E2 v2 cached privacy data (per-image edited_error_km per method,
budget). Adds CLIP utility metrics (cosine similarity between original and
redacted images) and produces the Pareto plot.

Scope: Panel A only — visible redaction methods head-to-head.
Panel B (cross-paradigm with GeoShield) is deferred per the v0 budget.

Methods covered:
  - dynamic_greedy (PRIMARY per §12.2)
  - static_greedy
  - largest_regions
  - random_regions

Selections are re-derived deterministically from the cached regions
parquet + scores parquet for the score-based methods. For dynamic_greedy
we read the cached selected_region_ids from the E2 v2 dynamic
predictions JSONL — re-running dynamic greedy would cost another ~$5.

Outputs
-------
  data/processed/redactions/E4_im2gps3k_utility.parquet
  outputs/tables/redaction_method_comparison.csv
  outputs/figures/privacy_utility_panelA.png
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.data.geo_utils import haversine_km
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.modal_app import app
from geoleaklens.redaction.baselines import LargestRegions, RandomRegions
from geoleaklens.redaction.methods import StaticGreedy
from geoleaklens.segmentation.clip_label_modal import CLIPLabelModal


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mask_npz(path: Path) -> np.ndarray:
    npz = np.load(path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def _load_dynamic_selections(path: Path) -> dict[tuple[str, float], list[str]]:
    """Read the E2 v2 dynamic-greedy JSONL, key by (image_id, target_budget)."""
    out: dict[tuple[str, float], list[str]] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            iid = row.get("image_id")
            budget = row.get("target_area_frac")
            sel = row.get("selected_region_ids")
            if iid is None or budget is None or sel is None:
                continue
            out[(iid, float(budget))] = list(sel)
    return out


def _build_union_mask(
    selected_ids: list[str],
    regions: pd.DataFrame,
    shape: tuple[int, int],
) -> np.ndarray:
    out = np.zeros(shape, dtype=bool)
    if not selected_ids:
        return out
    sel = regions[regions["region_id"].isin(selected_ids)]
    for _, row in sel.iterrows():
        m = _load_mask_npz(Path(row["mask_path"]))
        if m.shape == shape:
            out |= m
    return out


def _derive_selection(
    method_name: str,
    image_id: str,
    regions: pd.DataFrame,
    scores: pd.DataFrame,
    budget: float,
    *,
    seed: int,
    ranking_intervention_type: str,
    dynamic_selections: dict[tuple[str, float], list[str]],
) -> list[str]:
    """Return the region_ids the named method would pick at this budget."""
    if method_name == "dynamic_greedy":
        return dynamic_selections.get((image_id, float(budget)), [])
    if method_name == "static_greedy":
        m = StaticGreedy(intervention_type=ranking_intervention_type)
        return m.select_regions(image_id, regions, scores, {"area_frac": budget})
    if method_name == "largest":
        m = LargestRegions(intervention_type="mean_mask")
        return m.select_regions(image_id, regions, scores, {"area_frac": budget})
    if method_name == "random":
        m = RandomRegions(intervention_type="mean_mask", seed=seed)
        return m.select_regions(image_id, regions, scores, {"area_frac": budget})
    raise ValueError(f"unknown method: {method_name!r}")


def run(
    *,
    manifest_path: Path,
    regions_path: Path,
    scores_path: Path,
    e2_redactions_path: Path,
    dynamic_jsonl: Path,
    out_parquet: Path,
    table_path: Path,
    figure_path: Path,
    seed: int,
    ranking_intervention_type: str,
    threshold_km: float,
    batch_size: int,
) -> dict:
    manifest = pd.read_parquet(manifest_path).set_index("image_id")
    regions = pd.read_parquet(regions_path)
    scores = pd.read_parquet(scores_path)
    redactions = pd.read_parquet(e2_redactions_path)
    dynamic_selections = _load_dynamic_selections(dynamic_jsonl)
    print(
        f"[e4] redactions={len(redactions)} rows, "
        f"dynamic selections cached for {len(dynamic_selections)} (image, budget) pairs"
    )

    # Pre-cache original-image bytes once per image_id.
    orig_bytes_by_id: dict[str, bytes] = {}
    image_shape_by_id: dict[str, tuple[int, int]] = {}
    for image_id in redactions["image_id"].unique():
        if image_id not in manifest.index:
            continue
        path = Path(manifest.loc[image_id, "image_path"])
        if not path.exists():
            continue
        orig_bytes_by_id[image_id] = path.read_bytes()
        with Image.open(io.BytesIO(orig_bytes_by_id[image_id])) as im:
            w, h = im.size
        image_shape_by_id[image_id] = (h, w)
    print(f"[e4] cached {len(orig_bytes_by_id)} original-image bytes")

    # Pre-build (orig_bytes, edited_bytes) pairs for every redaction row.
    # Uses the score-based methods' deterministic selection logic and the
    # dynamic_greedy cached selections. The actual model call is batched
    # below via Modal `.map()`.
    pair_rows: list[dict] = []
    pairs: list[tuple[bytes, bytes]] = []
    for i, row in redactions.iterrows():
        image_id = row["image_id"]
        method = row["method"]
        budget = float(row["target_area_frac"])
        if image_id not in orig_bytes_by_id:
            continue
        orig_bytes = orig_bytes_by_id[image_id]
        h, w = image_shape_by_id[image_id]

        try:
            selected = _derive_selection(
                method, image_id, regions, scores, budget,
                seed=seed,
                ranking_intervention_type=ranking_intervention_type,
                dynamic_selections=dynamic_selections,
            )
        except Exception as e:
            print(f"[skip-derive] {image_id} / {method} / b={budget}: {e}")
            continue

        union = _build_union_mask(selected, regions, (h, w))
        pil = Image.open(io.BytesIO(orig_bytes)).convert("RGB")
        if union.any():
            edited, _ = mean_mask(pil, union, mode="local")
        else:
            edited = pil
        buf = io.BytesIO()
        edited.save(buf, format="JPEG", quality=92)
        edit_bytes = buf.getvalue()

        pair_rows.append({
            "image_id": image_id,
            "method": method,
            "target_area_frac": budget,
            "actual_area_frac": float(union.sum()) / float(h * w) if h * w else 0.0,
            "n_selected": int(len(selected)),
            "edited_error_km": float(row["edited_error_km"]),
            "edited_success_25km": bool(row.get("edited_success_25km", False)),
        })
        pairs.append((orig_bytes, edit_bytes))
        if (i + 1) % 100 == 0:
            print(f"[prep] {i + 1}/{len(redactions)} pairs prepared")
    print(f"[e4] total pairs prepared: {len(pairs)}")

    # Batched CLIP cos-sim. Splitting into chunks of `batch_size` keeps any
    # single Modal call from going over container memory.
    sims: list[float] = []
    with app.run():
        clip = CLIPLabelModal()
        for start in range(0, len(pairs), batch_size):
            chunk = pairs[start:start + batch_size]
            chunk_sims = clip.cosine_similarity_pairs.remote(chunk)
            sims.extend(chunk_sims)
            print(f"[clip] {min(start + batch_size, len(pairs))}/{len(pairs)} encoded")

    if len(sims) != len(pair_rows):
        raise RuntimeError(
            f"sim count {len(sims)} != row count {len(pair_rows)}"
        )

    # Compute orig errors per image (haversine of E1 GeoCLIP prediction
    # vs ground truth). We approximate this from the redactions parquet —
    # any redaction row's `edited_error_km` minus its delta is the orig
    # error, but since we don't have orig per row directly, we re-derive
    # from per-image baseline using the FIRST row per image as a proxy.
    # Actually: take the minimum edited_error_km across all rows per image
    # as a stable lower bound, OR compute haversine from E1 jsonl.
    # Cleanest: read the E1 GeoCLIP predictions JSONL.
    e1_path = Path("data/processed/predictions/geoclip/E1_im2gps3k.jsonl")
    orig_err_by_id: dict[str, float] = {}
    if e1_path.exists():
        with open(e1_path) as f:
            for line in f:
                r = json.loads(line)
                iid = r["image_id"]
                p = r.get("parsed", {})
                plat = p.get("lat")
                plon = p.get("lon")
                if iid in manifest.index and plat is not None and plon is not None:
                    err = haversine_km(
                        plat, plon,
                        float(manifest.loc[iid, "lat"]),
                        float(manifest.loc[iid, "lon"]),
                    )
                    orig_err_by_id[iid] = err

    # Attach utility + privacy delta to rows.
    for row, sim in zip(pair_rows, sims):
        row["clip_similarity"] = float(sim)
        row["utility_cost"] = max(0.0, 1.0 - float(sim))
        orig_err = orig_err_by_id.get(row["image_id"])
        if orig_err is None:
            row["error_increase_km"] = float("nan")
        else:
            row["error_increase_km"] = float(row["edited_error_km"] - orig_err)

    out_df = pd.DataFrame(pair_rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_parquet(out_parquet, index=False)
    print(f"[e4] wrote {out_parquet} ({len(out_df)} rows)")

    # Aggregate per (method, budget): median error increase, mean CLIP sim.
    summary = (
        out_df
        .groupby(["method", "target_area_frac"])
        .agg(
            n=("image_id", "size"),
            mean_clip_similarity=("clip_similarity", "mean"),
            median_clip_similarity=("clip_similarity", "median"),
            median_error_increase_km=("error_increase_km", "median"),
            mean_error_increase_km=("error_increase_km", "mean"),
            mean_actual_area_frac=("actual_area_frac", "mean"),
            mean_acc25km_after=("edited_success_25km", "mean"),
        )
        .reset_index()
        .sort_values(["method", "target_area_frac"])
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False, float_format="%.4f")
    print(f"[e4] wrote {table_path}")

    # Pareto plot.
    _plot_pareto(summary, figure_path)
    print(f"[e4] wrote {figure_path}")

    print("\n=== §13.E4 Pareto summary ===")
    print(summary.to_string(index=False, formatters={
        "mean_clip_similarity": "{:6.3f}".format,
        "median_error_increase_km": "{:8.1f}".format,
        "target_area_frac": "{:5.1%}".format,
        "mean_actual_area_frac": "{:5.1%}".format,
        "mean_acc25km_after": "{:5.1%}".format,
    }))
    return {
        "n_rows": int(len(out_df)),
        "table": str(table_path),
        "figure": str(figure_path),
    }


def _plot_pareto(summary: pd.DataFrame, output_path: Path) -> None:
    """Two-panel Pareto: Acc@25km drop (primary, has signal at our scale)
    + median error increase (informational; collapses to zero at n=30
    because most images don't move past the 25 km threshold).
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)
    methods = sorted(summary["method"].unique())
    colors = {
        "dynamic_greedy": "#2c7fb8",
        "static_greedy":  "#41ab5d",
        "random":         "#fdae61",
        "largest":        "#e34a33",
    }
    markers = {
        "dynamic_greedy": "o",
        "static_greedy":  "s",
        "random":         "^",
        "largest":        "D",
    }

    # Conditional set is filtered to baseline-success-at-Acc@25km, so the
    # baseline Acc@25km is 1.0 by construction; drop = 1 - acc_after.
    summary = summary.copy()
    summary["acc25km_drop"] = 1.0 - summary["mean_acc25km_after"]

    for panel_idx, (ycol, ylabel, title) in enumerate([
        ("acc25km_drop", "Acc@25km drop (privacy ↑)",
         "PRIMARY: how often does redaction break Acc@25km?"),
        ("median_error_increase_km", "median geodesic error increase, km",
         "informational: collapses to ~0 at n=30 because <50% of "
         "images move past 25 km"),
    ]):
        ax = axes[panel_idx]
        for m in methods:
            sub = summary[summary["method"] == m].sort_values("target_area_frac")
            ax.plot(
                sub["mean_clip_similarity"],
                sub[ycol] * (100 if "drop" in ycol else 1),
                marker=markers.get(m, "o"),
                color=colors.get(m, None),
                linewidth=2,
                markersize=8,
                label=m,
            )
            for _, row in sub.iterrows():
                yval = row[ycol] * (100 if "drop" in ycol else 1)
                ax.annotate(
                    f"{int(row['target_area_frac']*100)}%",
                    xy=(row["mean_clip_similarity"], yval),
                    xytext=(4, 4),
                    textcoords="offset points",
                    fontsize=7,
                    color=colors.get(m, "black"),
                )
        ax.set_xlabel("mean CLIP similarity (utility preserved →)")
        ax.set_ylabel(ylabel + (" (%)" if "drop" in ycol else ""))
        ax.set_title(title, fontsize=10)
        ax.grid(True, linestyle="--", alpha=0.4)
        ax.legend(loc="best", fontsize=9)

    fig.suptitle(
        "§13.E4 privacy-utility Pareto — Panel A (visible redaction)\n"
        "Im2GPS3k conditional set, mean_mask intervention, GeoCLIP threat model",
        fontsize=11,
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E4 privacy-utility Pareto runner.")
    ap.add_argument(
        "--manifest", default="data/processed/manifests/im2gps3k_test.parquet"
    )
    ap.add_argument(
        "--regions", default="data/processed/regions/E2_im2gps3k_sam.parquet"
    )
    ap.add_argument(
        "--scores",
        default="data/processed/metrics/leakage_scores_E2_im2gps3k_v1.parquet",
    )
    ap.add_argument(
        "--e2-redactions",
        default="data/processed/redactions/E2_im2gps3k_v2.parquet",
    )
    ap.add_argument(
        "--dynamic-jsonl",
        default="data/processed/predictions/geoclip/E2_im2gps3k_redacted_v2__dynamic.jsonl",
    )
    ap.add_argument(
        "--out-parquet",
        default="data/processed/redactions/E4_im2gps3k_utility.parquet",
    )
    ap.add_argument(
        "--table",
        default="outputs/tables/redaction_method_comparison.csv",
    )
    ap.add_argument(
        "--figure",
        default="outputs/figures/privacy_utility_panelA.png",
    )
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument(
        "--ranking-intervention-type", default="min_across",
        help="Score field StaticGreedy ranks by — match what E2 v2 used.",
    )
    ap.add_argument("--threshold-km", type=float, default=25.0)
    ap.add_argument(
        "--batch-size", type=int, default=64,
        help="CLIP cos-sim batch size per Modal call.",
    )
    args = ap.parse_args()

    run(
        manifest_path=Path(args.manifest),
        regions_path=Path(args.regions),
        scores_path=Path(args.scores),
        e2_redactions_path=Path(args.e2_redactions),
        dynamic_jsonl=Path(args.dynamic_jsonl),
        out_parquet=Path(args.out_parquet),
        table_path=Path(args.table),
        figure_path=Path(args.figure),
        seed=args.seed,
        ranking_intervention_type=args.ranking_intervention_type,
        threshold_km=args.threshold_km,
        batch_size=args.batch_size,
    )


if __name__ == "__main__":
    main()
