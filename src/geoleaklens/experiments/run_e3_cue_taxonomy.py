"""§13.E3 cue taxonomy runner — labels each SAM region and aggregates leakage
attribution by semantic bucket.

What this does
--------------
1. Load the conditional set (images where baseline GeoCLIP succeeds, per
   §13.E2 — re-uses the same 30-image subset as E2 by default).
2. Load cached SAM regions and the §10.7 min_across leakage scores from
   the E2 v1/v2 outputs. Both are required inputs.
3. For each image: run §9.3 OCR via EasyOCRModal, run §9.5 CLIP zero-shot
   labeling on every SAM bbox via CLIPLabelModal.
4. Combine labels via `label_regions_for_image` (§9.5 priority order:
   ocr_text > object > clip_zero_shot > fallback). Object detection is
   not yet wired — that's the next E3 chunk; CLIP zero-shot serves as the
   non-OCR label source for v0.
5. Attach §13.E3 buckets via `attach_buckets`.
6. Aggregate per-bucket statistics via `aggregate_by_bucket` and write
   `outputs/tables/cue_taxonomy.csv`.

Outputs
-------
  data/processed/regions/E3_im2gps3k_labeled.parquet  # §7.2 + §9.5 schema
  data/processed/predictions/ocr/E3_im2gps3k_ocr.jsonl
  outputs/tables/cue_taxonomy.csv

Plots (cue_taxonomy_bar.pdf etc.) are a separate runner — keep this
script focused on producing the labeled-regions parquet + table.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.modal_app import app
from geoleaklens.scoring.cue_taxonomy import aggregate_by_bucket, attach_buckets
from geoleaklens.segmentation.clip_label_modal import (
    DEFAULT_CLIP_PROMPTS,
    CLIPLabelModal,
)
from geoleaklens.segmentation.label_regions import label_regions_for_image
from geoleaklens.segmentation.ocr_modal import EasyOCRModal
from geoleaklens.segmentation.ocr_regions import detections_to_regions


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mask_npz(path: Path) -> np.ndarray:
    """Inverse of run_e0_smoke._save_mask_npz / run_e2 helpers."""
    npz = np.load(path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def _read_existing_jsonl_keyed(path: Path, key: str) -> dict[str, dict]:
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
            if key in row:
                out[row[key]] = row
    return out


def _run_ocr_phase(
    df: pd.DataFrame,
    ocr: EasyOCRModal,
    *,
    out_jsonl: Path,
    resume: bool,
) -> dict[str, list[dict]]:
    """Per-image OCR detections, keyed by image_id. Streams to JSONL."""
    cache = _read_existing_jsonl_keyed(out_jsonl, "image_id") if resume else {}
    out_jsonl.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume and cache else "w"
    out_per_image: dict[str, list[dict]] = {
        iid: row.get("detections", []) for iid, row in cache.items()
    }
    with open(out_jsonl, mode) as f:
        for i, row in df.iterrows():
            image_id = row["image_id"]
            if image_id in cache:
                continue
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            with open(image_path, "rb") as ip:
                img_bytes = ip.read()
            detections = ocr.detect.remote(img_bytes)
            out_per_image[image_id] = detections
            f.write(json.dumps({
                "image_id": image_id,
                "detections": detections,
                "n_detections": len(detections),
                "created_at": _now_iso(),
            }) + "\n")
            f.flush()
            if (i + 1) % 5 == 0:
                print(f"[ocr] {i + 1}/{len(df)} done")
    return out_per_image


def _run_clip_phase(
    df: pd.DataFrame,
    regions: pd.DataFrame,
    clip: CLIPLabelModal,
    *,
    prompts: tuple[str, ...] = DEFAULT_CLIP_PROMPTS,
) -> dict[str, dict[str, str]]:
    """Per-image CLIP top-1 label keyed by region_id.

    Returns {image_id -> {region_id -> top_label}}. We keep score on the
    side via the predictions log if you want to inspect later; for the
    label combiner we just need the top_label.
    """
    out: dict[str, dict[str, str]] = {}
    for i, row in df.iterrows():
        image_id = row["image_id"]
        image_path = Path(row["image_path"])
        if not image_path.exists():
            continue
        sub = regions[regions["image_id"] == image_id]
        if sub.empty:
            out[image_id] = {}
            continue
        bboxes = sub[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].values.tolist()
        with open(image_path, "rb") as ip:
            img_bytes = ip.read()
        per_region = clip.label_bboxes.remote(
            img_bytes, bboxes, list(prompts)
        )
        out[image_id] = {
            rid: lab.get("label")
            for rid, lab in zip(sub["region_id"].tolist(), per_region)
            if lab.get("label")
        }
        if (i + 1) % 5 == 0:
            print(f"[clip] {i + 1}/{len(df)} done")
    return out


def run(
    *,
    manifest_path: Path,
    regions_path: Path,
    scores_path: Path,
    masks_dir: Optional[Path],
    labeled_regions_path: Path,
    ocr_jsonl: Path,
    table_path: Path,
    intervention_type: str,
    top_k: int,
    resume: bool,
) -> dict:
    manifest = pd.read_parquet(manifest_path)
    regions = pd.read_parquet(regions_path)
    scores = pd.read_parquet(scores_path)

    # Restrict to images present in the regions parquet (the conditional set
    # already filtered). Keeps the OCR/CLIP loops on the same images E2
    # scored, so the leakage join is well-defined.
    image_ids = sorted(set(regions["image_id"]))
    df = manifest[manifest["image_id"].isin(image_ids)].reset_index(drop=True)
    print(f"[e3] {len(df)} images, {len(regions)} regions, {len(scores)} score rows")

    # --- 1. OCR phase + 2. CLIP phase under one Modal session ----------
    with app.run():
        ocr_modal = EasyOCRModal()
        ocr_per_image = _run_ocr_phase(
            df, ocr_modal, out_jsonl=ocr_jsonl, resume=resume,
        )

        clip_modal = CLIPLabelModal()
        clip_per_image = _run_clip_phase(df, regions, clip_modal)

    # --- 3. Build labeled-regions DataFrame ---------------------------
    labeled_rows: list[dict] = []
    for image_id in image_ids:
        sub_regions = regions[regions["image_id"] == image_id]
        if sub_regions.empty:
            continue
        # Load SAM masks from disk
        sam_masks: dict[str, np.ndarray] = {}
        for _, r in sub_regions.iterrows():
            try:
                sam_masks[r["region_id"]] = _load_mask_npz(Path(r["mask_path"]))
            except Exception as e:
                print(f"[skip-mask] {r['region_id']}: {e}")

        # Build OCR masks the same way (rasterize quads to the image shape).
        first_mask = next(iter(sam_masks.values()), None)
        if first_mask is None:
            continue
        h, w = first_mask.shape
        ocr_dets = ocr_per_image.get(image_id, [])
        ocr_region_rows = detections_to_regions(ocr_dets, image_id, (h, w))
        ocr_dets_with_id = [
            {"id": r["region_id"]} | {k: v for k, v in r.items() if k != "mask"}
            for r in ocr_region_rows
        ]
        ocr_masks = {r["region_id"]: r["mask"] for r in ocr_region_rows}

        labeled = label_regions_for_image(
            sam_masks=sam_masks,
            ocr_detections=ocr_dets_with_id,
            ocr_masks=ocr_masks,
            clip_zero_shot_labels=clip_per_image.get(image_id, {}),
        )
        labeled["image_id"] = image_id

        # Inner-join with the regions parquet so we carry area_frac, mask_path,
        # bbox, etc. forward.
        merged = sub_regions.merge(
            labeled, on=["image_id", "region_id"], how="inner",
        )
        labeled_rows.extend(merged.to_dict("records"))

    labeled_df = pd.DataFrame(labeled_rows)
    labeled_df = attach_buckets(labeled_df)
    labeled_regions_path.parent.mkdir(parents=True, exist_ok=True)
    # `labels` / `secondary_labels` / `secondary_buckets` / `overlapping_*`
    # are list-typed; pyarrow handles list<string> natively.
    labeled_df.to_parquet(labeled_regions_path, index=False)
    print(f"[e3] wrote {labeled_regions_path} ({len(labeled_df)} rows)")

    # --- 3b. Text-light subset (§13.E3 companion) ----------------------
    # An image is "text-light" when total OCR pixel coverage < 2% of the
    # image area. We compute this from the union of all OCR masks per
    # image (so overlapping detections don't double-count).
    text_frac_per_image: dict[str, float] = {}
    for image_id in image_ids:
        sub_regions = regions[regions["image_id"] == image_id]
        if sub_regions.empty:
            text_frac_per_image[image_id] = 0.0
            continue
        # Use the regions parquet's first mask for shape (cheap; we only
        # need (H, W)).
        try:
            ref = _load_mask_npz(Path(sub_regions.iloc[0]["mask_path"]))
        except Exception:
            text_frac_per_image[image_id] = 0.0
            continue
        h, w = ref.shape
        total_area = float(h * w) if h * w > 0 else 1.0
        ocr_dets = ocr_per_image.get(image_id, [])
        ocr_rows = detections_to_regions(ocr_dets, image_id, (h, w))
        if not ocr_rows:
            text_frac_per_image[image_id] = 0.0
            continue
        union = np.zeros((h, w), dtype=bool)
        for r in ocr_rows:
            union |= r["mask"]
        text_frac_per_image[image_id] = float(union.sum()) / total_area
    text_light_ids = {iid for iid, f in text_frac_per_image.items() if f < 0.02}
    print(
        f"[text-light] {len(text_light_ids)}/{len(image_ids)} images have "
        f"OCR coverage < 2% (median text frac = "
        f"{np.median(list(text_frac_per_image.values())):.4f})"
    )

    # --- 4. Aggregate (full + text-light) -----------------------------
    summary = aggregate_by_bucket(
        labeled_df, scores,
        intervention_type=intervention_type,
        top_k=top_k,
    )
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False, float_format="%.4f")

    summary_textlight = aggregate_by_bucket(
        labeled_df[labeled_df["image_id"].isin(text_light_ids)],
        scores,
        intervention_type=intervention_type,
        top_k=top_k,
    )
    textlight_path = table_path.with_name(
        table_path.stem + "_textlight" + table_path.suffix
    )
    summary_textlight.to_csv(textlight_path, index=False, float_format="%.4f")
    print(f"[e3] wrote text-light subset table to {textlight_path} "
          f"({len(text_light_ids)} images)")

    print("\n=== §13.E3 cue taxonomy summary ===")
    cols = ["bucket", "n_regions",
            "mean_continuous_attribution",
            "frac_top1_dominant",
            f"frac_top{top_k}_dominant",
            f"frac_top{top_k}_secondary_overlap"]
    show = summary[cols].copy()
    show["mean_continuous_attribution"] = show["mean_continuous_attribution"].map(
        "{:6.3f}".format
    )
    for c in ("frac_top1_dominant",
              f"frac_top{top_k}_dominant",
              f"frac_top{top_k}_secondary_overlap"):
        show[c] = show[c].map("{:5.1%}".format)
    # Drop buckets with zero regions to keep the printed table tight.
    nonempty = show[summary["n_regions"] > 0]
    print(nonempty.to_string(index=False))
    print(f"\nwrote {table_path}")
    return {
        "n_images": int(len(df)),
        "n_regions": int(len(labeled_df)),
        "labeled_regions_parquet": str(labeled_regions_path),
        "table": str(table_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E3 cue taxonomy runner.")
    ap.add_argument(
        "--manifest",
        default="data/processed/manifests/im2gps3k_test.parquet",
    )
    ap.add_argument(
        "--regions",
        default="data/processed/regions/E2_im2gps3k_sam.parquet",
    )
    ap.add_argument(
        "--scores",
        default="data/processed/metrics/leakage_scores_E2_im2gps3k_v1.parquet",
    )
    ap.add_argument(
        "--labeled-regions",
        default="data/processed/regions/E3_im2gps3k_labeled.parquet",
    )
    ap.add_argument(
        "--ocr-jsonl",
        default="data/processed/predictions/ocr/E3_im2gps3k_ocr.jsonl",
    )
    ap.add_argument(
        "--table",
        default="outputs/tables/cue_taxonomy.csv",
    )
    ap.add_argument("--intervention-type", default="min_across")
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--resume", action="store_true",
                    help="Skip OCR for images already in the OCR JSONL.")
    args = ap.parse_args()

    run(
        manifest_path=Path(args.manifest),
        regions_path=Path(args.regions),
        scores_path=Path(args.scores),
        masks_dir=None,  # masks are referenced by absolute path in the parquet
        labeled_regions_path=Path(args.labeled_regions),
        ocr_jsonl=Path(args.ocr_jsonl),
        table_path=Path(args.table),
        intervention_type=args.intervention_type,
        top_k=args.top_k,
        resume=args.resume,
    )


if __name__ == "__main__":
    main()
