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
from geoleaklens.interventions.blur import blur
from geoleaklens.interventions.inpaint import inpaint
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.modal_app import GeoCLIPModal, app
from geoleaklens.models.inpaint_modal import LaMaModal
from geoleaklens.redaction.baselines import LargestRegions, RandomRegions
from geoleaklens.redaction.methods import StaticGreedy
from geoleaklens.redaction.optimize import DynamicGreedy
from geoleaklens.scoring.causal_scores import (
    DEFAULT_INTERVENTION_SET,
    min_across_interventions,
    single_region_attribution,
)
from geoleaklens.scoring.parse_predictions import parse_geolocation_response
from geoleaklens.segmentation.sam_modal import SAMModal, decode_masks


def _apply_intervention(
    name: str,
    pil_image: Image.Image,
    mask: np.ndarray,
    *,
    lama_modal=None,
) -> Image.Image:
    """Dispatch on intervention name; matches §10 spec defaults.

    `lama_modal` is required when `name == "inpaint"`. The other two run
    locally with no Modal dep, so callers can pass None when only mean_mask
    or blur are needed.
    """
    if name == "mean_mask":
        edited, _ = mean_mask(pil_image, mask, mode="local")
        return edited
    if name == "blur":
        edited, _ = blur(pil_image, mask)
        return edited
    if name == "inpaint":
        if lama_modal is None:
            raise ValueError("inpaint requested but lama_modal is None")
        edited, _ = inpaint(pil_image, mask, lama_modal=lama_modal)
        return edited
    raise ValueError(f"unknown intervention type: {name!r}")


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
        # Use a single permutation and take the first max_images indices, so
        # the sample is *nested* across max_images values — n=30 ⊂ n=100 ⊂
        # n=500 with the same seed. Important for cache reuse on scale-up
        # runs and for "is this image already in the smaller set?" auditing.
        rng = np.random.default_rng(seed)
        order = rng.permutation(len(sub))[:max_images]
        sub = sub.iloc[sorted(order)].reset_index(drop=True)
        print(f"[e2] subsampled to {len(sub)} images (seed={seed}, nested)")
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
    intervention_types: list[str],
    lama_modal=None,
    threshold_km: float,
    region_cap: int,
    pred_jsonl: Path,
) -> pd.DataFrame:
    """Per-region single-attribution scores under each requested intervention.

    Emits one row of the §7.5 schema per (image, region, intervention).
    Caller can stitch §10.7 `min_across` rows in via `min_across_interventions(...)`.

    Concurrency
    -----------
    The per-image inner loop is fully batched:

      1. mean_mask / blur are local NumPy ops; produce all trial PIL images
         in memory.
      2. inpaint trial images are batched through `lama.inpaint.starmap(...)`
         so all K LaMa calls per image fan out across Modal containers.
      3. All K * |interventions| edited bytes are then batched through
         `geoclip.predict.map(...)` for parallel GeoCLIP scoring.

    For the n=100 / region_cap=15 / 3 interventions / image set, sequential
    scoring took ~5+ hours; the batched path cuts that to ~30-45 min.

    `lama_modal` is required if `"inpaint"` is in `intervention_types`.
    """
    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    if "inpaint" in intervention_types and lama_modal is None:
        raise ValueError("inpaint requested but lama_modal is None")

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
            w, h = pil.size

            picked = sub.sort_values("area_frac", ascending=False).head(region_cap)

            # Phase A: load all the masks for this image once.
            region_records = []
            for _, r in picked.iterrows():
                try:
                    mask = _load_mask_npz(Path(r["mask_path"]))
                except Exception as e:
                    print(f"[attr-skip-mask] {r['region_id']}: {e}")
                    continue
                if mask.shape != (h, w):
                    continue
                region_records.append((r, mask))
            if not region_records:
                continue

            # Phase B: inpaint trial images, batched via LaMa `.starmap()`.
            # Build (img_bytes, mask_png_bytes) pairs once, fan out, collect
            # in input order. Local mean_mask / blur trials happen in-process.
            need_inpaint = "inpaint" in intervention_types
            inpaint_results: list[Optional[bytes]] = []
            if need_inpaint:
                from geoleaklens.interventions.mask import dilate_mask
                inpaint_inputs: list[tuple[bytes, bytes]] = []
                for _, mask in region_records:
                    applied = dilate_mask(mask.astype(bool), 7)
                    img_buf = io.BytesIO()
                    pil.save(img_buf, format="JPEG", quality=92)
                    mask_pil = Image.fromarray(
                        (applied.astype(np.uint8) * 255), mode="L"
                    )
                    mask_buf = io.BytesIO()
                    mask_pil.save(mask_buf, format="PNG")
                    inpaint_inputs.append(
                        (img_buf.getvalue(), mask_buf.getvalue())
                    )
                try:
                    raw_inpaints = list(lama_modal.inpaint.starmap(inpaint_inputs))
                    inpaint_results = list(raw_inpaints)
                except Exception as e:
                    print(f"[attr-skip-lama-batch] {image_id}: {e}")
                    inpaint_results = [None] * len(region_records)

            # Phase C: build all (region, intervention) edited PIL images.
            #   Build trial bytes for batched GeoCLIP scoring; remember the
            #   (region_id, intervention) keying so we can re-thread the
            #   batched results back to score rows.
            trial_jobs: list[tuple[str, str, bytes]] = []  # (region_id, intervention, jpeg_bytes)
            for ridx, (r, mask) in enumerate(region_records):
                for intervention_type in intervention_types:
                    try:
                        if intervention_type == "mean_mask":
                            edited, _ = mean_mask(pil, mask, mode="local")
                        elif intervention_type == "blur":
                            edited, _ = blur(pil, mask)
                        elif intervention_type == "inpaint":
                            inpainted_bytes = (
                                inpaint_results[ridx]
                                if ridx < len(inpaint_results)
                                else None
                            )
                            if inpainted_bytes is None:
                                continue
                            edited = Image.open(
                                io.BytesIO(inpainted_bytes)
                            ).convert("RGB")
                            if edited.size != pil.size:
                                edited = edited.crop((0, 0, pil.size[0], pil.size[1]))
                        else:
                            continue
                    except Exception as e:
                        print(
                            f"[attr-skip] {r['region_id']} / "
                            f"{intervention_type}: {type(e).__name__}: {e}"
                        )
                        continue
                    buf = io.BytesIO()
                    edited.save(buf, format="JPEG", quality=92)
                    trial_jobs.append(
                        (r["region_id"], intervention_type, buf.getvalue())
                    )

            # Phase D: batched GeoCLIP scoring of all trials for this image.
            if not trial_jobs:
                print(f"[attribution] {image_id}: no trials produced ({i+1}/{len(df)})")
                continue
            bytes_list = [job[2] for job in trial_jobs]
            try:
                if len(bytes_list) == 1:
                    raw_results = [geoclip.predict.remote(bytes_list[0])]
                else:
                    raw_results = list(geoclip.predict.map(bytes_list))
            except Exception as e:
                print(f"[attr-skip-geoclip-batch] {image_id}: {e}")
                continue

            for (region_id, intervention_type, _), edit_pred in zip(
                trial_jobs, raw_results
            ):
                edit_parsed = parse_geolocation_response(edit_pred)
                pred_fp.write(json.dumps({
                    "image_id": image_id,
                    "image_variant_id": f"{region_id}__{intervention_type}",
                    "model_name": "geoclip",
                    "intervention_type": intervention_type,
                    "raw_response": edit_parsed["raw_response"],
                    "parsed": edit_parsed["parsed"],
                    "parse_success": edit_parsed["parse_success"],
                    "created_at": _now_iso(),
                }) + "\n")
                pred_fp.flush()
                # Look up area_frac for this region.
                area_frac = float(
                    picked.loc[picked["region_id"] == region_id, "area_frac"].iloc[0]
                )
                score_rows.append(single_region_attribution(
                    image_id=image_id,
                    region_id=region_id,
                    model_name="geoclip",
                    intervention_type=intervention_type,
                    threshold_km=threshold_km,
                    true_lat=true_lat,
                    true_lon=true_lon,
                    orig_lat=orig_lat, orig_lon=orig_lon,
                    edit_lat=edit_parsed["parsed"].get("lat"),
                    edit_lon=edit_parsed["parsed"].get("lon"),
                    area_frac=area_frac,
                    method="static_greedy",
                ))
            print(
                f"[attribution] {image_id}: {len(region_records)} regions × "
                f"{len(intervention_types)} interventions, "
                f"{len(trial_jobs)} GeoCLIP-batched ({i+1}/{len(df)})"
            )
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


def _run_dynamic_greedy_sweep(
    df: pd.DataFrame,
    regions: pd.DataFrame,
    scores: pd.DataFrame,
    geoclip: GeoCLIPModal,
    *,
    method: DynamicGreedy,
    edit_intervention_type: str,
    budgets: list[float],
    threshold_km: float,
    pred_jsonl: Path,
    lama_modal=None,
) -> pd.DataFrame:
    """Per-image §12.2 dynamic greedy sweep with snapshots at each budget.

    Returns rows in the same shape as `_run_redaction_sweep` (image_id /
    method / target_area_frac / actual_area_frac / n_selected /
    edited_error_km / edited_success_25km) so the aggregator can concat.

    DynamicGreedy is fundamentally different from the score-based methods —
    each pick requires a Modal-side GeoCLIP call on the trial-edited image.
    The runner provides the model + intervention as callbacks; the
    DynamicGreedy class itself stays Modal-agnostic (and unit-testable).
    """
    pred_jsonl.parent.mkdir(parents=True, exist_ok=True)
    rows: list[dict] = []
    pred_fp = open(pred_jsonl, "w")

    if edit_intervention_type == "inpaint" and lama_modal is None:
        raise ValueError("inpaint edit requested but lama_modal is None")

    try:
        for i, manifest_row in df.iterrows():
            image_id = manifest_row["image_id"]
            image_path = Path(manifest_row["image_path"])
            if not image_path.exists():
                print(f"[skip-dyn] missing image: {image_path}")
                continue
            true_lat = float(manifest_row["lat"])
            true_lon = float(manifest_row["lon"])

            with open(image_path, "rb") as f:
                img_bytes = f.read()
            orig_pred = geoclip.predict.remote(img_bytes)
            orig_parsed = parse_geolocation_response(orig_pred)
            orig_lat = orig_parsed["parsed"].get("lat")
            orig_lon = orig_parsed["parsed"].get("lon")
            original_error_km = haversine_km(orig_lat, orig_lon, true_lat, true_lon)
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Closures over geoclip / lama_modal so DynamicGreedy stays
            # decoupled from Modal types.
            def _apply(image, mask):
                return _apply_intervention(
                    edit_intervention_type, image, mask, lama_modal=lama_modal
                )

            def _score_batch(images: list[Image.Image]):
                """Fan out K candidate scorings via Modal `.map()` so they run
                concurrently across GeoCLIPModal containers. Falls back to a
                single sequential call when batch size is 1 — `.map()` works
                fine there too but `.remote()` is one less round-trip.
                """
                if not images:
                    return []
                bytes_list: list[bytes] = []
                for img in images:
                    buf = io.BytesIO()
                    img.save(buf, format="JPEG", quality=92)
                    bytes_list.append(buf.getvalue())
                if len(bytes_list) == 1:
                    raw = geoclip.predict.remote(bytes_list[0])
                    parsed = parse_geolocation_response(raw)
                    return [(
                        parsed["parsed"].get("lat"),
                        parsed["parsed"].get("lon"),
                    )]
                # `.map(iterable)` returns a generator yielding results in
                # input order. Each is the @modal.method's return value.
                results = list(geoclip.predict.map(bytes_list))
                out: list[tuple] = []
                for raw in results:
                    parsed = parse_geolocation_response(raw)
                    out.append(
                        (parsed["parsed"].get("lat"),
                         parsed["parsed"].get("lon"))
                    )
                return out

            def _load(path):
                return _load_mask_npz(Path(path))

            try:
                result = method.run_image(
                    image_id=image_id,
                    regions=regions,
                    scores=scores,
                    budgets=budgets,
                    original_image=pil_image,
                    true_lat=true_lat,
                    true_lon=true_lon,
                    original_error_km=original_error_km,
                    load_mask=_load,
                    apply_intervention=_apply,
                    score_images=_score_batch,
                )
            except Exception as e:
                print(
                    f"[dyn-skip] {image_id}: {type(e).__name__}: {e}"
                )
                continue

            for budget in budgets:
                snap = result.snapshots[float(budget)]
                rows.append({
                    "image_id": image_id,
                    "method": method.name,
                    "target_area_frac": float(budget),
                    "actual_area_frac": float(snap.actual_area_frac),
                    "n_selected": len(snap.selected_region_ids),
                    "edited_error_km": float(snap.edited_error_km),
                    "edited_success_25km": bool(
                        success_at(snap.edited_error_km, threshold_km)
                    ),
                })
                pred_fp.write(json.dumps({
                    "image_id": image_id,
                    "image_variant_id": (
                        f"{method.name}__b{int(budget*100)}__{edit_intervention_type}"
                    ),
                    "method": method.name,
                    "target_area_frac": float(budget),
                    "model_name": "geoclip",
                    "parsed": {
                        "lat": snap.edited_lat,
                        "lon": snap.edited_lon,
                    },
                    "n_picks": result.n_picks,
                    "stopped_early": result.stopped_early,
                    "selected_region_ids": list(snap.selected_region_ids),
                    "created_at": _now_iso(),
                }) + "\n")
                pred_fp.flush()
            print(
                f"[redact-dyn] {image_id}: {result.n_picks} picks, "
                f"stopped_early={result.stopped_early} ({i+1}/{len(df)})"
            )
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
    intervention_types: list[str],  # NEW — which interventions to score under
    ranking_intervention_type: str,  # NEW — which scores StaticGreedy ranks by
    include_dynamic_greedy: bool,
    dynamic_greedy_top_k: int,
    dynamic_greedy_edit_intervention: str,
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
        needs_lama = "inpaint" in intervention_types
        with app.run():
            geoclip = GeoCLIPModal()
            lama_modal = LaMaModal() if needs_lama else None
            scores = _run_attribution_phase(
                df, regions, geoclip,
                intervention_types=intervention_types,
                lama_modal=lama_modal,
                threshold_km=threshold_km,
                region_cap=region_cap_for_attribution,
                pred_jsonl=pred_attribution_jsonl,
            )

        # §10.7 — materialize min_across rows alongside the per-intervention
        # rows. Static/dynamic greedy can then filter by intervention_type
        # to switch between the conservative headline and per-intervention
        # diagnostics without touching the runner.
        if len(intervention_types) > 1:
            min_rows = min_across_interventions(
                scores, intervention_types=tuple(intervention_types)
            )
            if not min_rows.empty:
                scores = pd.concat([scores, min_rows], ignore_index=True)
                print(f"[min_across] added {len(min_rows)} aggregated rows")

        scores_path.parent.mkdir(parents=True, exist_ok=True)
        scores.to_parquet(scores_path, index=False)
        print(f"[attribution] wrote {scores_path} ({len(scores)} rows)")

    # --- 3. Redaction sweep -------------------------------------------
    # StaticGreedy ranks by `ranking_intervention_type` (default "min_across").
    # The baselines ignore scores entirely, so their intervention_type label
    # is purely cosmetic — left at "mean_mask" because that's the §13.E2
    # primary edit type per the spec.
    methods = [
        StaticGreedy(intervention_type=ranking_intervention_type),
        LargestRegions(intervention_type="mean_mask"),
        RandomRegions(intervention_type="mean_mask", seed=seed),
    ]
    if redactions_path.exists() and skip_redaction:
        redactions = pd.read_parquet(redactions_path)
        print(f"[redact] reusing {len(redactions)} rows from {redactions_path}")
    else:
        # `dynamic_greedy_edit_intervention == "inpaint"` requires a live
        # LaMa cls. Spin it up alongside GeoCLIP only when we need it so the
        # common path (mean_mask edits) doesn't pay the cold-start tax.
        needs_lama_for_dyn = (
            include_dynamic_greedy
            and dynamic_greedy_edit_intervention == "inpaint"
        )
        with app.run():
            geoclip = GeoCLIPModal()
            redactions = _run_redaction_sweep(
                df, regions, scores, geoclip,
                methods=methods,
                budgets=budgets,
                threshold_km=threshold_km,
                pred_jsonl=pred_redaction_jsonl,
            )
            if include_dynamic_greedy:
                lama_for_dyn = LaMaModal() if needs_lama_for_dyn else None
                dyn_method = DynamicGreedy(
                    intervention_type=ranking_intervention_type,
                    top_k=dynamic_greedy_top_k,
                )
                # Append dynamic-greedy rows to the predictions JSONL so the
                # downstream tooling sees one stream per E2 run.
                dyn_pred_jsonl = pred_redaction_jsonl.with_name(
                    pred_redaction_jsonl.stem + "__dynamic.jsonl"
                )
                dyn_rows = _run_dynamic_greedy_sweep(
                    df, regions, scores, geoclip,
                    method=dyn_method,
                    edit_intervention_type=dynamic_greedy_edit_intervention,
                    budgets=budgets,
                    threshold_km=threshold_km,
                    pred_jsonl=dyn_pred_jsonl,
                    lama_modal=lama_for_dyn,
                )
                redactions = pd.concat([redactions, dyn_rows], ignore_index=True)
                print(f"[redact-dyn] appended {len(dyn_rows)} rows; "
                      f"predictions at {dyn_pred_jsonl}")
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
    ap.add_argument(
        "--intervention-types",
        default=",".join(DEFAULT_INTERVENTION_SET),
        help="Comma-separated interventions to score regions under "
             "(e.g. mean_mask,blur,inpaint). 'inpaint' requires LaMa.",
    )
    ap.add_argument(
        "--ranking-intervention-type",
        default="min_across",
        help="Which intervention's scores StaticGreedy ranks by. "
             "Default 'min_across' is the §10.7 conservative headline.",
    )
    ap.add_argument(
        "--include-dynamic-greedy",
        action="store_true",
        help="Add §12.2 dynamic greedy alongside the score-based methods. "
             "Expensive: top_k * n_picks GeoCLIP calls per image. "
             "Off by default — opt in when you want the spec's headline measure.",
    )
    ap.add_argument(
        "--dynamic-greedy-top-k",
        type=int,
        default=30,
        help="Initial candidate shortlist size for dynamic greedy (§12.2 default).",
    )
    ap.add_argument(
        "--dynamic-greedy-edit-intervention",
        default="mean_mask",
        choices=["mean_mask", "blur", "inpaint"],
        help="Intervention applied during dynamic-greedy selection. "
             "Default 'mean_mask' matches the existing redaction sweep.",
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
        intervention_types=[x.strip() for x in args.intervention_types.split(",")],
        ranking_intervention_type=args.ranking_intervention_type,
        include_dynamic_greedy=args.include_dynamic_greedy,
        dynamic_greedy_top_k=args.dynamic_greedy_top_k,
        dynamic_greedy_edit_intervention=args.dynamic_greedy_edit_intervention,
        skip_sam=args.skip_sam,
        skip_attribution=args.skip_attribution,
        skip_redaction=args.skip_redaction,
    )


if __name__ == "__main__":
    main()
