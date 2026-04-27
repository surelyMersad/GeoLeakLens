"""§13.E0 smoke test runner.

Goal (from §13.E0): make sure the pipeline works end-to-end on the 10-image
wikimedia smoke set. We deliberately scope this to:

    GeoCLIP (Modal) → SAM v1 masks (Modal) → mean_mask intervention (local) →
        GeoCLIP again on each edited image (Modal) → §11.3 single-region
        attribution scoring (local).

Outputs (matching the §7.x schemas):

    data/processed/predictions/geoclip/E0_smoke.jsonl   # §7.3
    data/processed/regions/wikimedia_smoke_sam.parquet  # §7.2 (sam-only subset)
    data/processed/metrics/leakage_scores_E0_smoke.parquet  # §7.5 (single-region)

We honor two caps:

* SAM emits up to `--sam-max-regions` per image (default 80, the §9.6 cap).
* The intervention/scoring loop operates on `--regions-per-image` (default 10,
  the §13.E0 spec). This keeps the smoke run cheap on L4 — each region is one
  GeoCLIP forward pass, and the §13.E0 acceptance criterion is "no script
  crashes / metrics file exists", not coverage.

Usage:
    python -m geoleaklens.experiments.run_e0_smoke \
        --manifest data/processed/manifests/wikimedia_smoke_mvp.parquet
"""
from __future__ import annotations

import argparse
import io
import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from PIL import Image

# Importing this module first registers SAMModal on the shared `app`; the
# GeoCLIP cls is registered by `modal_app` itself via the import side-effect.
from geoleaklens.segmentation.sam_modal import SAMModal, decode_masks
from geoleaklens.modal_app import GeoCLIPModal, app
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.scoring.parse_predictions import parse_geolocation_response
from geoleaklens.scoring.causal_scores import (
    assign_ranks_within_image,
    single_region_attribution,
)


REGION_COLUMNS = [
    "image_id",
    "region_id",
    "source",
    "label",
    "bbox_x1",
    "bbox_y1",
    "bbox_x2",
    "bbox_y2",
    "area_px",
    "area_frac",
    "mask_path",
    "stability_score",
    "predicted_iou",
    "ocr_text",
    "object_confidence",
    "parent_region_id",
    "dedup_group",
]


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _read_image_bytes(path: Path) -> bytes:
    with open(path, "rb") as f:
        return f.read()


def _save_mask_npz(path: Path, mask: np.ndarray) -> None:
    """Save a single bool mask as packed uint8 in compressed npz."""
    path.parent.mkdir(parents=True, exist_ok=True)
    packed = np.packbits(mask.astype(np.uint8).reshape(-1))
    np.savez_compressed(path, packed=packed, shape=np.array(mask.shape, dtype=np.int64))


def _pick_top_regions_by_area(
    masks: np.ndarray, meta: list[dict], k: int
) -> tuple[np.ndarray, list[dict]]:
    """Select up to k regions with the largest area_frac. Stable for ties."""
    if masks.shape[0] == 0 or k <= 0:
        return masks[:0], []
    order = sorted(
        range(len(meta)),
        key=lambda i: float(meta[i]["area_frac"]),
        reverse=True,
    )[:k]
    sel = sorted(order)  # preserve original mask ordering after pick
    return masks[sel], [meta[i] for i in sel]


def run(
    *,
    manifest_path: Path,
    out_predictions: Path,
    out_regions: Path,
    out_metrics: Path,
    masks_dir: Path,
    threshold_km: float = 25.0,
    sam_max_regions: int = 80,
    regions_per_image: int = 10,
    run_id: str = "E0_smoke",
) -> dict[str, Any]:
    """Run the §13.E0 pipeline. Returns a small summary dict."""
    df = pd.read_parquet(manifest_path)
    if df.empty:
        raise SystemExit(f"manifest is empty: {manifest_path}")

    out_predictions.parent.mkdir(parents=True, exist_ok=True)
    out_regions.parent.mkdir(parents=True, exist_ok=True)
    out_metrics.parent.mkdir(parents=True, exist_ok=True)

    region_rows: list[dict] = []
    score_rows: list[dict] = []
    n_images_with_masks = 0
    n_predictions_parsed = 0
    n_predictions_total = 0

    # Open predictions JSONL up-front so we stream rather than holding in memory.
    pred_fp = open(out_predictions, "w")

    try:
        with app.run():
            geoclip = GeoCLIPModal()
            sam = SAMModal()

            for _, row in df.iterrows():
                image_id = row["image_id"]
                image_path = Path(row["image_path"])
                if not image_path.exists():
                    print(f"[skip] missing image: {image_path}")
                    continue

                true_lat = float(row["lat"]) if pd.notna(row.get("lat")) else None
                true_lon = float(row["lon"]) if pd.notna(row.get("lon")) else None
                if true_lat is None or true_lon is None:
                    print(f"[skip] {image_id}: missing ground truth lat/lon")
                    continue

                t0 = time.time()
                img_bytes = _read_image_bytes(image_path)

                # --- 1. Original GeoCLIP prediction --------------------------
                orig_raw = geoclip.predict.remote(img_bytes)
                orig_pred = parse_geolocation_response(orig_raw)
                n_predictions_total += 1
                if orig_pred["parse_success"]:
                    n_predictions_parsed += 1
                pred_fp.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "image_id": image_id,
                            "image_variant_id": "original",
                            "model_name": "geoclip",
                            "model_type": "local",
                            "prompt_id": None,
                            "temperature": 0,
                            "raw_response": orig_pred["raw_response"],
                            "parsed": orig_pred["parsed"],
                            "parse_success": orig_pred["parse_success"],
                            "created_at": _now_iso(),
                            "error": orig_pred["error"],
                        }
                    )
                    + "\n"
                )

                # --- 2. SAM masks --------------------------------------------
                sam_out = sam.generate_masks.remote(
                    img_bytes,
                    min_area_frac=0.001,
                    max_area_frac=0.45,
                    dedup_iou_threshold=0.85,
                    max_regions=sam_max_regions,
                )
                n_masks = int(sam_out["n_masks"])
                masks_all = decode_masks(
                    sam_out["masks_packed_b64"],
                    n_masks,
                    int(sam_out["image_height"]),
                    int(sam_out["image_width"]),
                )
                if n_masks > 0:
                    n_images_with_masks += 1

                # --- 3. Pick the regions to intervene on ---------------------
                masks_sel, meta_sel = _pick_top_regions_by_area(
                    masks_all, sam_out["masks_meta"], regions_per_image
                )

                # Save §7.2 region rows for ALL kept SAM masks (not just the
                # ones we intervene on — the regions parquet is the search
                # space, not the action set).
                for i, meta in enumerate(sam_out["masks_meta"]):
                    region_id = f"{image_id}__sam_{i:03d}"
                    mask_rel = masks_dir / image_id / f"{region_id}.npz"
                    _save_mask_npz(mask_rel, masks_all[i])
                    region_rows.append(
                        {
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
                            "mask_path": str(mask_rel),
                            "stability_score": float(meta["stability_score"]),
                            "predicted_iou": float(meta["predicted_iou"]),
                            "ocr_text": None,
                            "object_confidence": None,
                            "parent_region_id": None,
                            "dedup_group": None,
                        }
                    )

                # --- 4. For each selected region: mean_mask + GeoCLIP + score
                pil_img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                for j, (mask_j, meta_j) in enumerate(zip(masks_sel, meta_sel)):
                    # Find the region_id that matches this meta_j by area+bbox.
                    sel_index = sam_out["masks_meta"].index(meta_j)
                    region_id = f"{image_id}__sam_{sel_index:03d}"

                    edited_img, _applied = mean_mask(pil_img, mask_j, mode="local")
                    buf = io.BytesIO()
                    edited_img.save(buf, format="JPEG", quality=92)
                    edit_bytes = buf.getvalue()

                    edit_raw = geoclip.predict.remote(edit_bytes)
                    edit_pred = parse_geolocation_response(edit_raw)
                    n_predictions_total += 1
                    if edit_pred["parse_success"]:
                        n_predictions_parsed += 1

                    pred_fp.write(
                        json.dumps(
                            {
                                "run_id": run_id,
                                "image_id": image_id,
                                "image_variant_id": f"{region_id}__mean_mask",
                                "model_name": "geoclip",
                                "model_type": "local",
                                "prompt_id": None,
                                "temperature": 0,
                                "raw_response": edit_pred["raw_response"],
                                "parsed": edit_pred["parsed"],
                                "parse_success": edit_pred["parse_success"],
                                "created_at": _now_iso(),
                                "error": edit_pred["error"],
                            }
                        )
                        + "\n"
                    )

                    score_rows.append(
                        single_region_attribution(
                            image_id=image_id,
                            region_id=region_id,
                            model_name="geoclip",
                            intervention_type="mean_mask",
                            threshold_km=threshold_km,
                            true_lat=true_lat,
                            true_lon=true_lon,
                            orig_lat=orig_pred["parsed"]["lat"],
                            orig_lon=orig_pred["parsed"]["lon"],
                            edit_lat=edit_pred["parsed"]["lat"],
                            edit_lon=edit_pred["parsed"]["lon"],
                            area_frac=float(meta_j["area_frac"]),
                            method="static_greedy",
                        )
                    )

                print(
                    f"[ok] {image_id}: {n_masks} sam masks → "
                    f"intervened on {len(meta_sel)} ({time.time() - t0:.1f}s)"
                )
    finally:
        pred_fp.close()

    # Persist the parquet outputs.
    regions_df = pd.DataFrame(region_rows, columns=REGION_COLUMNS)
    regions_df.to_parquet(out_regions, index=False)

    score_rows = assign_ranks_within_image(score_rows)
    scores_df = pd.DataFrame(score_rows)
    scores_df.to_parquet(out_metrics, index=False)

    summary = {
        "run_id": run_id,
        "n_images": int(len(df)),
        "n_images_with_masks": n_images_with_masks,
        "frac_images_with_masks": (
            n_images_with_masks / len(df) if len(df) else 0.0
        ),
        "n_predictions_total": n_predictions_total,
        "n_predictions_parsed": n_predictions_parsed,
        "parse_success_rate": (
            n_predictions_parsed / n_predictions_total
            if n_predictions_total
            else 0.0
        ),
        "n_score_rows": int(len(score_rows)),
        "regions_parquet": str(out_regions),
        "predictions_jsonl": str(out_predictions),
        "metrics_parquet": str(out_metrics),
    }
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E0 smoke-test runner.")
    ap.add_argument(
        "--manifest",
        default="data/processed/manifests/wikimedia_smoke_mvp.parquet",
    )
    ap.add_argument(
        "--out-predictions",
        default="data/processed/predictions/geoclip/E0_smoke.jsonl",
    )
    ap.add_argument(
        "--out-regions",
        default="data/processed/regions/wikimedia_smoke_sam.parquet",
    )
    ap.add_argument(
        "--out-metrics",
        default="data/processed/metrics/leakage_scores_E0_smoke.parquet",
    )
    ap.add_argument(
        "--masks-dir",
        default="data/interim/masks/wikimedia_smoke",
    )
    ap.add_argument("--threshold-km", type=float, default=25.0)
    ap.add_argument("--sam-max-regions", type=int, default=80)
    ap.add_argument("--regions-per-image", type=int, default=10)
    ap.add_argument("--run-id", default="E0_smoke")
    args = ap.parse_args()

    summary = run(
        manifest_path=Path(args.manifest),
        out_predictions=Path(args.out_predictions),
        out_regions=Path(args.out_regions),
        out_metrics=Path(args.out_metrics),
        masks_dir=Path(args.masks_dir),
        threshold_km=args.threshold_km,
        sam_max_regions=args.sam_max_regions,
        regions_per_image=args.regions_per_image,
        run_id=args.run_id,
    )

    # §13.E0 acceptance: at least 90% of images have masks; predictions parse;
    # metrics file exists; no script crashes. Print + exit non-zero if not met.
    print("\n=== E0 summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")

    failures = []
    if summary["frac_images_with_masks"] < 0.9:
        failures.append(
            f"only {summary['frac_images_with_masks']:.0%} images had masks (<90%)"
        )
    if summary["parse_success_rate"] < 0.9:
        failures.append(
            f"only {summary['parse_success_rate']:.0%} predictions parsed (<90%)"
        )
    if not Path(args.out_metrics).exists():
        failures.append(f"metrics file not written: {args.out_metrics}")

    if failures:
        print("\n=== E0 acceptance FAILED ===")
        for f in failures:
            print(f"  - {f}")
        raise SystemExit(1)
    print("\n=== E0 acceptance PASSED ===")


if __name__ == "__main__":
    main()
