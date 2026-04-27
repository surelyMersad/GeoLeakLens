"""§13.E0 steps 7-8: 10% budget redaction + qualitative contact sheet.

Reads the artifacts written by `run_e0_smoke` (regions parquet, scores parquet)
and:

  7. For each image, picks regions via §12.1 static greedy until ~10% of pixel
     area is covered, paints the union of those regions with mean_mask, saves
     the redacted JPEG, and re-runs GeoCLIP on the redacted image to record
     the post-redaction prediction.

  8. Renders a 2x5 contact sheet with original + redacted side-by-side per
     image, each pair labeled with original/redacted error in km.

The attribution scores from `run_e0_smoke` are reused (no re-scoring); §12.1
is the cheap-approximation path that consumes existing single-region scores.

Outputs:
    data/processed/redactions/E0_static_greedy_10pct/{image_id}.jpg
    data/processed/redactions/E0_static_greedy_10pct.parquet  # selection log
    data/processed/predictions/geoclip/E0_redacted.jsonl       # §7.3 rows
    outputs/figures/e0_contact_sheet.png
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.interventions.mask import mean_mask
from geoleaklens.modal_app import GeoCLIPModal, app
from geoleaklens.redaction.methods import StaticGreedy, selection_summary
from geoleaklens.scoring.parse_predictions import parse_geolocation_response
from geoleaklens.data.geo_utils import haversine_km
import geoleaklens.segmentation.sam_modal  # noqa: F401  (registers SAMModal on app)


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mask_npz(path: Path) -> np.ndarray:
    """Inverse of run_e0_smoke._save_mask_npz."""
    npz = np.load(path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def _union_masks(mask_paths: list[Path], shape: tuple[int, int]) -> np.ndarray:
    """OR all listed masks into a single bool array of `shape`. Empty -> all-False."""
    out = np.zeros(shape, dtype=bool)
    for p in mask_paths:
        m = _load_mask_npz(p)
        if m.shape != shape:
            raise ValueError(f"mask shape {m.shape} != expected {shape} ({p})")
        out |= m
    return out


def run_redaction(
    *,
    manifest_path: Path,
    regions_path: Path,
    scores_path: Path,
    redaction_dir: Path,
    redactions_log_path: Path,
    redacted_predictions_path: Path,
    contact_sheet_path: Path,
    target_area_frac: float = 0.10,
    run_id: str = "E0_static_greedy_10pct",
) -> dict:
    manifest = pd.read_parquet(manifest_path)
    regions = pd.read_parquet(regions_path)
    scores = pd.read_parquet(scores_path)

    redaction_dir.mkdir(parents=True, exist_ok=True)
    redactions_log_path.parent.mkdir(parents=True, exist_ok=True)
    redacted_predictions_path.parent.mkdir(parents=True, exist_ok=True)
    contact_sheet_path.parent.mkdir(parents=True, exist_ok=True)

    method = StaticGreedy()
    log_rows: list[dict] = []
    pred_rows: list[dict] = []

    pred_fp = open(redacted_predictions_path, "w")

    try:
        with app.run():
            geoclip = GeoCLIPModal()

            for _, row in manifest.iterrows():
                image_id = row["image_id"]
                image_path = Path(row["image_path"])
                if not image_path.exists():
                    print(f"[skip] missing image: {image_path}")
                    continue

                # 1. Static greedy region selection.
                selected = method.select_regions(
                    image_id=image_id,
                    regions=regions,
                    scores=scores,
                    budget={"area_frac": target_area_frac},
                )

                # 2. Build the union mask + paint with mean_mask.
                pil = Image.open(image_path).convert("RGB")
                w, h = pil.size

                if selected:
                    mask_paths = (
                        regions.loc[
                            regions["region_id"].isin(selected), "mask_path"
                        ]
                        .map(Path)
                        .tolist()
                    )
                    union = _union_masks(mask_paths, shape=(h, w))
                    edited, applied = mean_mask(pil, union, mode="local")
                else:
                    edited = pil
                    applied = np.zeros((h, w), dtype=bool)

                # 3. Save the redacted JPEG (§10.4: strip EXIF on save —
                #    Pillow's default JPEG save drops EXIF unless explicitly
                #    re-attached, so we're already safe).
                redacted_path = redaction_dir / f"{image_id}.jpg"
                edited.save(redacted_path, format="JPEG", quality=92)

                # 4. Re-run GeoCLIP on the redacted image so the contact sheet
                #    can label the post-redaction prediction.
                with open(redacted_path, "rb") as f:
                    edit_bytes = f.read()
                edit_raw = geoclip.predict.remote(edit_bytes)
                edit_pred = parse_geolocation_response(edit_raw)

                pred_rows.append(
                    {
                        "image_id": image_id,
                        "variant": run_id,
                        "lat": edit_pred["parsed"]["lat"],
                        "lon": edit_pred["parsed"]["lon"],
                        "parse_success": edit_pred["parse_success"],
                    }
                )
                pred_fp.write(
                    json.dumps(
                        {
                            "run_id": run_id,
                            "image_id": image_id,
                            "image_variant_id": run_id,
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

                summary = selection_summary(
                    selected_ids=selected,
                    regions=regions,
                    image_id=image_id,
                    target_area_frac=target_area_frac,
                )
                summary.update(
                    {
                        "method": method.name,
                        "intervention_type": method.intervention_type,
                        "redacted_image_path": str(redacted_path),
                        "applied_pixel_frac": float(applied.sum()) / float(h * w),
                    }
                )
                log_rows.append(summary)

                edit_lat = edit_pred["parsed"]["lat"]
                edit_lon = edit_pred["parsed"]["lon"]
                true_lat = float(row["lat"])
                true_lon = float(row["lon"])
                edit_err = haversine_km(edit_lat, edit_lon, true_lat, true_lon)
                print(
                    f"[ok] {image_id}: selected {len(selected)} regions, "
                    f"covered {summary['actual_area_frac']*100:.1f}% / target "
                    f"{target_area_frac*100:.0f}%, edit_err={edit_err:.2f} km"
                )
    finally:
        pred_fp.close()

    # --- Persist the selection log -----------------------------------------
    log_df = pd.DataFrame(log_rows)
    # selected_region_ids is a Python list — write as JSON string for parquet
    # compatibility (pyarrow handles list<string> natively but JSON is more
    # portable across readers).
    log_df["selected_region_ids"] = log_df["selected_region_ids"].map(json.dumps)
    log_df.to_parquet(redactions_log_path, index=False)

    # --- Step 8: contact sheet ---------------------------------------------
    _render_contact_sheet(
        manifest=manifest,
        log_df=pd.DataFrame(log_rows),  # un-jsonified copy still in memory
        pred_lookup={r["image_id"]: r for r in pred_rows},
        scores=scores,
        output_path=contact_sheet_path,
        target_area_frac=target_area_frac,
    )

    n = len(log_rows)
    summary = {
        "run_id": run_id,
        "n_images": int(n),
        "median_actual_area_frac": float(log_df["actual_area_frac"].median())
        if n
        else 0.0,
        "median_area_gap_pp": float(log_df["area_gap_pp"].median()) if n else 0.0,
        "redactions_log": str(redactions_log_path),
        "redacted_predictions": str(redacted_predictions_path),
        "contact_sheet": str(contact_sheet_path),
    }
    return summary


def _render_contact_sheet(
    *,
    manifest: pd.DataFrame,
    log_df: pd.DataFrame,
    pred_lookup: dict[str, dict],
    scores: pd.DataFrame,
    output_path: Path,
    target_area_frac: float,
) -> None:
    """Step 8: 2x5 grid, original | redacted side-by-side per image.

    Each pair gets a small overlay of the redaction mask outline on the
    original (so it's clear *where* we redacted), and labels showing the
    original prediction error vs. the post-redaction error.
    """
    images = log_df["image_id"].tolist()
    n = len(images)
    if n == 0:
        return

    # 2 rows × 5 cols × 2 sub-cols (orig | redacted) = 4 wide x 5 deep, but
    # easier to read as 5 rows × 2 cols. With 10 images that's 5x4 panels.
    # Just go 5x4 (5 rows of 2 image-pairs).
    n_cols_pairs = 2
    n_rows = (n + n_cols_pairs - 1) // n_cols_pairs

    fig, axes = plt.subplots(
        n_rows, n_cols_pairs * 2, figsize=(20, 4 * n_rows), constrained_layout=True
    )
    fig.suptitle(
        f"E0 contact sheet — §12.1 static greedy at {target_area_frac*100:.0f}% area budget, mean_mask",
        fontsize=14,
    )
    if n_rows == 1:
        axes = axes.reshape(1, -1)

    # Compute original GeoCLIP errors from the existing scores parquet —
    # original_error_km is duplicated across all rows for a given image_id.
    orig_err = (
        scores.groupby("image_id")["original_error_km"].first().to_dict()
    )

    for i, image_id in enumerate(images):
        r = i // n_cols_pairs
        c_pair = i % n_cols_pairs
        ax_orig = axes[r, c_pair * 2]
        ax_red = axes[r, c_pair * 2 + 1]

        manifest_row = manifest.loc[manifest["image_id"] == image_id].iloc[0]
        log_row = log_df.loc[log_df["image_id"] == image_id].iloc[0]
        pred = pred_lookup.get(image_id)

        original = np.asarray(
            Image.open(Path(manifest_row["image_path"])).convert("RGB")
        )
        redacted = np.asarray(
            Image.open(Path(log_row["redacted_image_path"])).convert("RGB")
        )

        ax_orig.imshow(original)
        ax_orig.set_axis_off()
        ax_orig.set_title(
            f"{manifest_row['city']}, {manifest_row['country']}\n"
            f"original  err={orig_err.get(image_id, float('nan')):.1f} km",
            fontsize=10,
        )

        true_lat = float(manifest_row["lat"])
        true_lon = float(manifest_row["lon"])
        if pred and pred.get("parse_success"):
            edit_err = haversine_km(
                pred.get("lat"), pred.get("lon"), true_lat, true_lon
            )
            edit_err_label = f"{edit_err:.1f} km"
        else:
            edit_err_label = "no parse"

        ax_red.imshow(redacted)
        ax_red.set_axis_off()
        ax_red.set_title(
            f"redacted ({log_row['actual_area_frac']*100:.1f}% area, "
            f"{log_row['n_selected']} regions)\n"
            f"err={edit_err_label}",
            fontsize=10,
        )

    # Hide any unused axes (in case we ever extend beyond 10 images).
    used = n
    for j in range(used, n_rows * n_cols_pairs):
        r = j // n_cols_pairs
        c_pair = j % n_cols_pairs
        axes[r, c_pair * 2].set_axis_off()
        axes[r, c_pair * 2 + 1].set_axis_off()

    fig.savefig(output_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E0 steps 7-8 runner.")
    ap.add_argument(
        "--manifest",
        default="data/processed/manifests/wikimedia_smoke_mvp.parquet",
    )
    ap.add_argument(
        "--regions",
        default="data/processed/regions/wikimedia_smoke_sam.parquet",
    )
    ap.add_argument(
        "--scores",
        default="data/processed/metrics/leakage_scores_E0_smoke.parquet",
    )
    ap.add_argument(
        "--redaction-dir",
        default="data/processed/redactions/E0_static_greedy_10pct",
    )
    ap.add_argument(
        "--redactions-log",
        default="data/processed/redactions/E0_static_greedy_10pct.parquet",
    )
    ap.add_argument(
        "--redacted-predictions",
        default="data/processed/predictions/geoclip/E0_redacted.jsonl",
    )
    ap.add_argument(
        "--contact-sheet",
        default="outputs/figures/e0_contact_sheet.png",
    )
    ap.add_argument("--target-area-frac", type=float, default=0.10)
    ap.add_argument("--run-id", default="E0_static_greedy_10pct")
    args = ap.parse_args()

    summary = run_redaction(
        manifest_path=Path(args.manifest),
        regions_path=Path(args.regions),
        scores_path=Path(args.scores),
        redaction_dir=Path(args.redaction_dir),
        redactions_log_path=Path(args.redactions_log),
        redacted_predictions_path=Path(args.redacted_predictions),
        contact_sheet_path=Path(args.contact_sheet),
        target_area_frac=args.target_area_frac,
        run_id=args.run_id,
    )

    print("\n=== E0 redaction summary ===")
    for k, v in summary.items():
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
