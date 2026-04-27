"""§13.E2 leakage sparsity — does redacting top-ranked semantic regions reduce
geolocation accuracy faster than random / largest-area baselines?

Scope of this v0
----------------
This runner is the §13.E2 *primary conditional* version: filter to images
where the baseline GeoCLIP prediction succeeds at Acc@25km, then sweep
multiple area budgets for each of three redaction methods.

Methods compared (subset of §12, scoped down for v0):
  - StaticGreedy (§12.1) — uses single-region attribution scores
  - RandomRegions (§12.3) — area-matched
  - LargestRegions (§12.4)

Intervention: mean_mask only for v0. The §10.7 min_across_interventions
headline uses min(mean_mask, blur, inpaint); blur is implemented but the
runner doesn't fold all three together yet — that's a follow-up commit.

Pipeline
--------
1. Load manifest + the E1 GeoCLIP predictions JSONL.
2. Filter to images whose baseline error <= threshold_km (conditional set).
3. Optionally subsample to `--max-images` (E2 v0 default: 30 — keeps the
   full sweep under ~$2 of GPU time).
4. For each image: run SAM on Modal, save regions parquet + mask npz files.
5. For each (image, region): apply mean_mask, run GeoCLIP, score
   continuous_attribution. Persist to a leakage-scores parquet.
6. For each method × budget × image: select regions, union masks, paint
   with mean_mask, run GeoCLIP, record edited error.
7. Aggregate Acc@25km per (method, budget) and write the §13.E2 sparsity
   curve to `outputs/figures/e2_sparsity_curve.png`.

Outputs
-------
  data/processed/regions/E2_im2gps3k_sam.parquet
  data/processed/metrics/leakage_scores_E2_im2gps3k.parquet
  data/processed/redactions/E2_im2gps3k.parquet  (per (method, budget, image))
  data/processed/predictions/geoclip/E2_im2gps3k_redacted.jsonl
  outputs/tables/e2_sparsity.csv
  outputs/figures/e2_sparsity_curve.png
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.data.geo_utils import haversine_km, success_at
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.modal_app import GeoCLIPModal, app
from geoleaklens.redaction.baselines import LargestRegions, RandomRegions
from geoleaklens.redaction.methods import StaticGreedy
from geoleaklens.scoring.causal_scores import single_region_attribution
from geoleaklens.scoring.parse_predictions import parse_geolocation_response
from geoleaklens.segmentation.sam_modal import SAMModal, decode_masks


REGION_COLUMNS = [
    "image_id", "region_id", "source", "label",
    "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
    "area_px", "area_frac", "mask_path",
    "stability_score", "predicted_iou",
    "ocr_text", "object_confidence", "parent_region_id", "dedup_group",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _save_mask_npz(path: Path, mask: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(mask.astype(np.uint8).reshape(-1))
    np.savez_compressed(path, packed=packed, shape=np.array(mask.shape, dtype=np.int64))


def _load_mask_npz(path: Path) -> np.ndarray:
    npz = np.load(path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def _read_existing_jsonl(path: Path) -> dict[str, dict]:
    if not path.exists():
        return {}
    out: dict[str, dict] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "image_id" in row:
                # Allow multiple variants per image; key off image_variant_id when present.
                key = row.get("image_variant_id") or row["image_id"]
                out[key] = row
    return out


def _build_conditional_set(
    manifest: pd.DataFrame,
    geoclip_jsonl: Path,
    threshold_km: float,
    max_images: Optional[int],
    seed: int,
) -> pd.DataFrame:
    """Return manifest rows whose baseline GeoCLIP error <= threshold_km."""
    if not geoclip_jsonl.exists():
        raise SystemExit(
            f"[e2] no E1 GeoCLIP predictions at {geoclip_jsonl}. "
            "Run run_baseline_geolocation first."
        )
    preds: dict[str, dict] = {}
    with open(geoclip_jsonl) as f:
        for line in f:
            r = json.loads(line)
            preds[r["image_id"]] = r["parsed"]

    keep_ids: list[str] = []
    for _, row in manifest.iterrows():
        p = preds.get(row["image_id"])
        if not p:
            continue
        plat, plon = p.get("lat"), p.get("lon")
        if plat is None or plon is None:
            continue
        e = haversine_km(plat, plon, float(row["lat"]), float(row["lon"]))
        if e <= threshold_km:
            keep_ids.append(row["image_id"])

    sub = manifest[manifest["image_id"].isin(keep_ids)].reset_index(drop=True)
    print(f"[e2] conditional set: {len(sub)}/{len(manifest)} images "
          f"with baseline error ≤ {threshold_km} km")
    if max_images and len(sub) > max_images:
        rng = np.random.default_rng(seed)
        idx = rng.choice(len(sub), size=max_images, replace=False)
        sub = sub.iloc[sorted(idx)].reset_index(drop=True)
        print(f"[e2] subsampled to {len(sub)} images (seed={seed})")
    return sub


def _run_sam_phase(
    df: pd.DataFrame,
    sam: SAMModal,
    masks_dir: Path,
    *,
    sam_max_regions: int,
) -> pd.DataFrame:
    """Returns a §7.2-shaped regions DataFrame with mask_paths on disk."""
    rows: list[dict] = []
    for i, row in df.iterrows():
        image_id = row["image_id"]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            print(f"[sam-skip] missing image: {image_path}")
            continue
        with open(image_path, "rb") as f:
            img_bytes = f.read()
        out = sam.generate_masks.remote(
            img_bytes,
            min_area_frac=0.001,
            max_area_frac=0.45,
            dedup_iou_threshold=0.85,
            max_regions=sam_max_regions,
        )
        n_masks = int(out["n_masks"])
        if n_masks == 0:
            print(f"[sam] {image_id}: 0 masks — skipping")
            continue
        masks = decode_masks(
            out["masks_packed_b64"], n_masks,
            int(out["image_height"]), int(out["image_width"]),
        )
        for j, meta in enumerate(out["masks_meta"]):
            region_id = f"{image_id}__sam_{j:03d}"
            mask_path = masks_dir / image_id / f"{region_id}.npz"
            _save_mask_npz(mask_path, masks[j])
            rows.append({
                "image_id": image_id,
                "region_id": region_id,
                "source": "sam",
                "label": None,
                "bbox_x1": int(meta["bbox_x1"]),
                "bbox_y1": int(meta["bbox_y1"]),
                "bbox_x2": int(meta["bbox_x2"]),
                "bbox_y2": int(meta["bbox_y2"]),
                "area_px": int(meta["area_px"]),
                "area_frac": float(meta["area_frac"]),
                "mask_path": str(mask_path),
                "stability_score": float(meta["stability_score"]),
                "predicted_iou": float(meta["predicted_iou"]),
                "ocr_text": None,
                "object_confidence": None,
                "parent_region_id": None,
                "dedup_group": None,
            })
        print(f"[sam] {image_id}: {n_masks} masks ({i+1}/{len(df)})")
    return pd.DataFrame(rows, columns=REGION_COLUMNS)


def _run_attribution_phase(
    df: pd.DataFrame,
    regions: pd.DataFrame,
    geoclip: GeoCLIPModal,
    *,
    threshold_km: float,
    region_cap: int,
    pred_jsonl: Path,
) -> pd.DataFrame:
    """Per-region single-attribution scores under mean_mask only."""
    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    score_rows: list[dict] = []
    pred_fp = open(pred_jsonl, "w")

    try:
        for i, row in df.iterrows():
            image_id = row["image_id"]
            true_lat, true_lon = float(row["lat"]), float(row["lon"])
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            sub = regions[regions["image_id"] == image_id]
            if sub.empty:
                continue
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            orig_pred = geoclip.predict.remote(img_bytes)
            orig_parsed = parse_geolocation_response(orig_pred)
            orig_lat = orig_parsed["parsed"].get("lat")
            orig_lon = orig_parsed["parsed"].get("lon")

            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Score top-`region_cap` regions by area to keep cost bounded.
            picked = sub.sort_values("area_frac", ascending=False).head(region_cap)
            for _, r in picked.iterrows():
                mask = _load_mask_npz(Path(r["mask_path"]))
                edited, _ = mean_mask(pil, mask, mode="local")
                buf = io.BytesIO()
                edited.save(buf, format="JPEG", quality=92)
                edit_pred = geoclip.predict.remote(buf.getvalue())
                edit_parsed = parse_geolocation_response(edit_pred)
                pred_fp.write(json.dumps({
                    "image_id": image_id,
                    "image_variant_id": f"{r['region_id']}__mean_mask",
                    "model_name": "geoclip",
                    "raw_response": edit_parsed["raw_response"],
                    "parsed": edit_parsed["parsed"],
                    "parse_success": edit_parsed["parse_success"],
                    "created_at": _now_iso(),
                }) + "\n")
                pred_fp.flush()
                score_rows.append(single_region_attribution(
                    image_id=image_id,
                    region_id=r["region_id"],
                    model_name="geoclip",
                    intervention_type="mean_mask",
                    threshold_km=threshold_km,
                    true_lat=true_lat,
                    true_lon=true_lon,
                    orig_lat=orig_lat, orig_lon=orig_lon,
                    edit_lat=edit_parsed["parsed"].get("lat"),
                    edit_lon=edit_parsed["parsed"].get("lon"),
                    area_frac=float(r["area_frac"]),
                    method="static_greedy",
                ))
            print(f"[attribution] {image_id}: {len(picked)} regions ({i+1}/{len(df)})")
    finally:
        pred_fp.close()
    return pd.DataFrame(score_rows)


def _run_redaction_sweep(
    df: pd.DataFrame,
    regions: pd.DataFrame,
    scores: pd.DataFrame,
    geoclip: GeoCLIPModal,
    *,
    methods: list,
    budgets: list[float],
    threshold_km: float,
    pred_jsonl: Path,
) -> pd.DataFrame:
    """For each (method, budget, image), apply intervention to union of
    selected regions and score the post-redaction GeoCLIP prediction."""
    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pred_fp = open(pred_jsonl, "w")
    try:
        for i, manifest_row in df.iterrows():
            image_id = manifest_row["image_id"]
            true_lat, true_lon = float(manifest_row["lat"]), float(manifest_row["lon"])
            image_path = Path(manifest_row["image_path"])
            if not image_path.exists():
                continue
            with open(image_path, "rb") as f:
                img_bytes = f.read()
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")
            w, h = pil.size

            for method in methods:
                for budget in budgets:
                    selected = method.select_regions(
                        image_id=image_id,
                        regions=regions,
                        scores=scores,
                        budget={"area_frac": float(budget)},
                    )
                    if not selected:
                        # Empty selection at this budget — record an unedited row
                        # so the curve has a well-defined point at every budget.
                        rows.append({
                            "image_id": image_id,
                            "method": method.name,
                            "target_area_frac": float(budget),
                            "actual_area_frac": 0.0,
                            "n_selected": 0,
                            "edited_error_km": haversine_km(
                                true_lat, true_lon, true_lat, true_lon  # 0 km
                            ),
                            "edited_success_25km": True,
                        })
                        continue

                    sel_regs = regions.loc[
                        regions["region_id"].isin(selected), ["mask_path", "area_frac"]
                    ]
                    union = np.zeros((h, w), dtype=bool)
                    for _, rr in sel_regs.iterrows():
                        m = _load_mask_npz(Path(rr["mask_path"]))
                        if m.shape == (h, w):
                            union |= m

                    edited, applied = mean_mask(pil, union, mode="local")
                    buf = io.BytesIO()
                    edited.save(buf, format="JPEG", quality=92)
                    edit_bytes = buf.getvalue()
                    edit_pred = geoclip.predict.remote(edit_bytes)
                    edit_parsed = parse_geolocation_response(edit_pred)
                    plat = edit_parsed["parsed"].get("lat")
                    plon = edit_parsed["parsed"].get("lon")
                    err = haversine_km(plat, plon, true_lat, true_lon)
                    rows.append({
                        "image_id": image_id,
                        "method": method.name,
                        "target_area_frac": float(budget),
                        "actual_area_frac": float(applied.sum()) / float(h * w),
                        "n_selected": len(selected),
                        "edited_error_km": float(err),
                        "edited_success_25km": bool(success_at(err, threshold_km)),
                    })
                    pred_fp.write(json.dumps({
                        "image_id": image_id,
                        "image_variant_id": f"{method.name}__b{int(budget*100)}__mean_mask",
                        "method": method.name,
                        "target_area_frac": float(budget),
                        "model_name": "geoclip",
                        "raw_response": edit_parsed["raw_response"],
                        "parsed": edit_parsed["parsed"],
                        "parse_success": edit_parsed["parse_success"],
                        "created_at": _now_iso(),
                    }) + "\n")
                    pred_fp.flush()
            print(f"[redact] {image_id} done ({i+1}/{len(df)})")
    finally:
        pred_fp.close()
    return pd.DataFrame(rows)


def _aggregate_sparsity_curve(redactions: pd.DataFrame) -> pd.DataFrame:
    """Per (method, budget): mean Acc@25km across the conditional set."""
    return (
        redactions
        .groupby(["method", "target_area_frac"])
        .agg(
            n=("image_id", "size"),
            acc_25km=("edited_success_25km", "mean"),
            mean_actual_area=("actual_area_frac", "mean"),
        )
        .reset_index()
        .sort_values(["method", "target_area_frac"])
    )


def _plot_sparsity_curve(
    summary: pd.DataFrame, threshold_km: float, output_path: Path
) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8, 5))
    methods = sorted(summary["method"].unique())
    for m in methods:
        sub = summary[summary["method"] == m]
        ax.plot(
            sub["target_area_frac"] * 100,
            sub["acc_25km"] * 100,
            marker="o",
            label=m,
        )
    ax.set_xlabel("Redaction budget (% of pixel area)")
    ax.set_ylabel(f"Acc@{int(threshold_km)} km (%)")
    ax.set_title(
        f"§13.E2 leakage sparsity — Acc@{int(threshold_km)}km vs. budget\n"
        "conditional on baseline GeoCLIP success, mean_mask intervention"
    )
    ax.set_ylim(-5, 105)
    ax.legend()
    ax.grid(True, linestyle="--", alpha=0.5)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(output_path, dpi=140)
    plt.close(fig)


def run(
    *,
    manifest_path: Path,
    geoclip_jsonl: Path,
    regions_path: Path,
    scores_path: Path,
    redactions_path: Path,
    pred_attribution_jsonl: Path,
    pred_redaction_jsonl: Path,
    masks_dir: Path,
    table_path: Path,
    figure_path: Path,
    threshold_km: float,
    max_images: Optional[int],
    seed: int,
    sam_max_regions: int,
    region_cap_for_attribution: int,
    budgets: list[float],
    skip_sam: bool,
    skip_attribution: bool,
    skip_redaction: bool,
) -> dict:
    manifest = pd.read_parquet(manifest_path)
    df = _build_conditional_set(
        manifest, geoclip_jsonl, threshold_km, max_images, seed
    )

    # --- 1. SAM phase --------------------------------------------------
    if regions_path.exists() and skip_sam:
        regions = pd.read_parquet(regions_path)
        print(f"[sam] reusing {len(regions)} regions from {regions_path}")
    else:
        with app.run():
            sam = SAMModal()
            regions = _run_sam_phase(
                df, sam, masks_dir, sam_max_regions=sam_max_regions
            )
        regions_path.parent.mkdir(parents=True, exist_ok=True)
        regions.to_parquet(regions_path, index=False)
        print(f"[sam] wrote {regions_path} ({len(regions)} regions)")

    # --- 2. Attribution phase -----------------------------------------
    if scores_path.exists() and skip_attribution:
        scores = pd.read_parquet(scores_path)
        print(f"[attribution] reusing {len(scores)} score rows from {scores_path}")
    else:
        with app.run():
            geoclip = GeoCLIPModal()
            scores = _run_attribution_phase(
                df, regions, geoclip,
                threshold_km=threshold_km,
                region_cap=region_cap_for_attribution,
                pred_jsonl=pred_attribution_jsonl,
            )
        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path, index=False)
        print(f"[attribution] wrote {scores_path} ({len(scores)} rows)")

    # --- 3. Redaction sweep -------------------------------------------
    methods = [
        StaticGreedy(intervention_type="mean_mask"),
        LargestRegions(intervention_type="mean_mask"),
        RandomRegions(intervention_type="mean_mask", seed=seed),
    ]
    if redactions_path.exists() and skip_redaction:
        redactions = pd.read_parquet(redactions_path)
        print(f"[redact] reusing {len(redactions)} rows from {redactions_path}")
    else:
        with app.run():
            geoclip = GeoCLIPModal()
            redactions = _run_redaction_sweep(
                df, regions, scores, geoclip,
                methods=methods,
                budgets=budgets,
                threshold_km=threshold_km,
                pred_jsonl=pred_redaction_jsonl,
            )
        redactions_path.parent.mkdir(parents=True, exist_ok=True)
        redactions.to_parquet(redactions_path, index=False)
        print(f"[redact] wrote {redactions_path} ({len(redactions)} rows)")

    # --- 4. Aggregate + plot ------------------------------------------
    summary = _aggregate_sparsity_curve(redactions)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False, float_format="%.4f")
    _plot_sparsity_curve(summary, threshold_km, figure_path)

    print("\n=== §13.E2 sparsity summary ===")
    print(summary.to_string(index=False, formatters={
        "acc_25km": "{:6.1%}".format,
        "target_area_frac": "{:5.1%}".format,
        "mean_actual_area": "{:5.1%}".format,
    }))
    print(f"\nwrote {table_path}")
    print(f"wrote {figure_path}")
    return {
        "n_conditional": int(len(df)),
        "table": str(table_path),
        "figure": str(figure_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E2 leakage-sparsity runner.")
    ap.add_argument(
        "--manifest", default="data/processed/manifests/im2gps3k_test.parquet"
    )
    ap.add_argument(
        "--geoclip-jsonl",
        default="data/processed/predictions/geoclip/E1_im2gps3k.jsonl",
    )
    ap.add_argument(
        "--regions",
        default="data/processed/regions/E2_im2gps3k_sam.parquet",
    )
    ap.add_argument(
        "--scores",
        default="data/processed/metrics/leakage_scores_E2_im2gps3k.parquet",
    )
    ap.add_argument(
        "--redactions",
        default="data/processed/redactions/E2_im2gps3k.parquet",
    )
    ap.add_argument(
        "--pred-attribution-jsonl",
        default="data/processed/predictions/geoclip/E2_im2gps3k_attribution.jsonl",
    )
    ap.add_argument(
        "--pred-redaction-jsonl",
        default="data/processed/predictions/geoclip/E2_im2gps3k_redacted.jsonl",
    )
    ap.add_argument(
        "--masks-dir", default="data/interim/masks/im2gps3k_e2"
    )
    ap.add_argument("--table", default="outputs/tables/e2_sparsity.csv")
    ap.add_argument("--figure", default="outputs/figures/e2_sparsity_curve.png")
    ap.add_argument("--threshold-km", type=float, default=25.0)
    ap.add_argument("--max-images", type=int, default=30)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--sam-max-regions", type=int, default=80)
    ap.add_argument("--region-cap-for-attribution", type=int, default=30)
    ap.add_argument(
        "--budgets",
        default="0.01,0.02,0.05,0.10,0.15,0.20",
        help="Comma-separated area-fraction budgets to sweep.",
    )
    ap.add_argument("--skip-sam", action="store_true")
    ap.add_argument("--skip-attribution", action="store_true")
    ap.add_argument("--skip-redaction", action="store_true")
    args = ap.parse_args()

    run(
        manifest_path=Path(args.manifest),
        geoclip_jsonl=Path(args.geoclip_jsonl),
        regions_path=Path(args.regions),
        scores_path=Path(args.scores),
        redactions_path=Path(args.redactions),
        pred_attribution_jsonl=Path(args.pred_attribution_jsonl),
        pred_redaction_jsonl=Path(args.pred_redaction_jsonl),
        masks_dir=Path(args.masks_dir),
        table_path=Path(args.table),
        figure_path=Path(args.figure),
        threshold_km=args.threshold_km,
        max_images=args.max_images,
        seed=args.seed,
        sam_max_regions=args.sam_max_regions,
        region_cap_for_attribution=args.region_cap_for_attribution,
        budgets=[float(x) for x in args.budgets.split(",")],
        skip_sam=args.skip_sam,
        skip_attribution=args.skip_attribution,
        skip_redaction=args.skip_redaction,
    )


if __name__ == "__main__":
    main()
