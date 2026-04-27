"""§12.11 Shapley spot-check runner.

Validates that static greedy's region ranking roughly agrees with the
joint-causal Shapley ranking. Closes the §17 MVP criterion 6 gap.

Procedure (matches §12.11 spec):
  1. Pick the top-K regions per image by static-greedy score_per_area
     under the ranking intervention (default min_across).
  2. For each image, sample N random orderings of those K regions.
  3. Compute Shapley(r) for each region using the cached unique-set
     edits (see redaction/shapley.py).
  4. Compare: top-1 agreement, top-K Jaccard, mean Spearman rank corr
     against static-greedy's ranking.

Cost
----
The naive cost is K × N × n_images. The cached implementation reduces
to "unique region sets seen across all (ordering × prefix) pairs" per
image, which for K=10 / N=200 is bounded by 2^K = 1024 per image but
typically lower because orderings repeat prefixes.

Modal `.map()` is used for the per-image batch of GeoCLIP scorings —
same pattern as DynamicGreedy.

Usage
-----
For a v0 spot-check:
    python -m geoleaklens.experiments.run_shapley_validation \
        --max-images 10 --top-k 5 --n-orderings 50

For the full §12.11 spec:
    python -m geoleaklens.experiments.run_shapley_validation \
        --max-images 50 --top-k 10 --n-orderings 200
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.data.geo_utils import clip_error_km, haversine_km
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.modal_app import GeoCLIPModal, app
from geoleaklens.redaction.shapley import (
    ShapleyImageResult,
    shapley_spot_check_image,
    static_vs_shapley_agreement,
)
from geoleaklens.scoring.parse_predictions import parse_geolocation_response


def _load_mask_npz(path: Path) -> np.ndarray:
    npz = np.load(path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def run(
    *,
    manifest_path: Path,
    regions_path: Path,
    scores_path: Path,
    out_parquet: Path,
    out_table: Path,
    intervention_type: str,
    top_k: int,
    n_orderings: int,
    max_images: Optional[int],
    seed: int,
    threshold_km: float,
) -> dict:
    manifest = pd.read_parquet(manifest_path).set_index("image_id")
    regions = pd.read_parquet(regions_path)
    scores = pd.read_parquet(scores_path)

    # Pick image_ids present in scores at the configured intervention_type.
    eligible_ids = scores.loc[
        scores["intervention_type"] == intervention_type, "image_id"
    ].unique().tolist()
    if max_images:
        eligible_ids = eligible_ids[:max_images]
    print(
        f"[shapley] running spot-check on {len(eligible_ids)} images, "
        f"top_k={top_k}, n_orderings={n_orderings}"
    )

    shapley_rows: list[dict] = []
    shapley_results: list[ShapleyImageResult] = []

    with app.run():
        geoclip = GeoCLIPModal()

        def _score_batch(images: list[Image.Image]):
            if not images:
                return []
            bytes_list: list[bytes] = []
            for img in images:
                buf = io.BytesIO()
                img.save(buf, format="JPEG", quality=92)
                bytes_list.append(buf.getvalue())
            if len(bytes_list) == 1:
                raw = geoclip.predict.remote(bytes_list[0])
                p = parse_geolocation_response(raw)
                return [(p["parsed"].get("lat"), p["parsed"].get("lon"))]
            results = list(geoclip.predict.map(bytes_list))
            return [
                (parse_geolocation_response(r)["parsed"].get("lat"),
                 parse_geolocation_response(r)["parsed"].get("lon"))
                for r in results
            ]

        for i, image_id in enumerate(eligible_ids):
            if image_id not in manifest.index:
                continue
            row = manifest.loc[image_id]
            image_path = Path(row["image_path"])
            if not image_path.exists():
                continue
            true_lat = float(row["lat"])
            true_lon = float(row["lon"])

            # Pick top-K regions for this image by score_per_area under
            # the configured intervention (matches StaticGreedy ranking).
            sub = (
                scores[(scores["image_id"] == image_id)
                       & (scores["intervention_type"] == intervention_type)]
                .merge(
                    regions[regions["image_id"] == image_id]
                    [["region_id", "mask_path"]],
                    on="region_id", how="inner",
                )
                .sort_values("score_per_area", ascending=False)
                .head(top_k)
            )
            if sub.empty:
                continue

            region_ids = sub["region_id"].tolist()
            region_masks = {
                r["region_id"]: _load_mask_npz(Path(r["mask_path"]))
                for _, r in sub.iterrows()
            }

            with open(image_path, "rb") as f:
                img_bytes = f.read()
            pil = Image.open(io.BytesIO(img_bytes)).convert("RGB")

            # Original error: ground-truth haversine of the cached GeoCLIP
            # baseline prediction. We pull this from the scores parquet's
            # original_error_km field (any row for this image works — they
            # all share the unedited prediction).
            orig_err_row = scores[
                (scores["image_id"] == image_id)
                & (scores["intervention_type"] == intervention_type)
            ].iloc[0]
            original_error_km = float(orig_err_row["original_error_km"])

            def _apply(image, mask):
                edited, _ = mean_mask(image, mask, mode="local")
                return edited

            def _err_for_pred(lat, lon):
                if lat is None or lon is None:
                    return clip_error_km(float("inf"))
                return clip_error_km(haversine_km(lat, lon, true_lat, true_lon))

            res = shapley_spot_check_image(
                image_id=image_id,
                region_ids=region_ids,
                region_masks=region_masks,
                original_image=pil,
                original_error_km=original_error_km,
                apply_intervention=_apply,
                score_image=_score_batch,
                error_for_pred=_err_for_pred,
                n_orderings=n_orderings,
                seed=seed,
            )
            shapley_results.append(res)
            for rid, val in res.shapley_values.items():
                shapley_rows.append({
                    "image_id": image_id,
                    "region_id": rid,
                    "shapley_value": float(val),
                    "n_orderings": int(n_orderings),
                    "n_unique_sets_scored": int(res.n_unique_sets_scored),
                })
            print(
                f"[shapley] {image_id}: K={top_k}, "
                f"unique sets scored={res.n_unique_sets_scored} "
                f"({i + 1}/{len(eligible_ids)})"
            )

    # Persist per-region Shapley values.
    shapley_df = pd.DataFrame(shapley_rows)
    out_parquet.parent.mkdir(parents=True, exist_ok=True)
    shapley_df.to_parquet(out_parquet, index=False)
    print(f"[shapley] wrote {out_parquet} ({len(shapley_df)} rows)")

    # Aggregate static-vs-Shapley agreement.
    summary = static_vs_shapley_agreement(
        scores, shapley_results,
        intervention_type=intervention_type,
        top_k=top_k,
    )
    out_table.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([summary]).to_csv(out_table, index=False, float_format="%.4f")
    print(f"[shapley] wrote {out_table}")

    print("\n=== §12.11 Shapley validation summary ===")
    for k, v in summary.items():
        if isinstance(v, float):
            print(f"  {k}: {v:.3f}")
        else:
            print(f"  {k}: {v}")
    print()
    print("§12.11 acceptance:")
    if isinstance(summary["top1_agreement"], float) and isinstance(summary["mean_spearman"], float):
        if summary["top1_agreement"] >= 0.7 and summary["mean_spearman"] >= 0.7:
            print("  PASS — static greedy is a defensible fast approximation.")
        else:
            print("  FAIL — static greedy disagrees with Shapley; only "
                  "dynamic greedy / Shapley results support the causal claim.")
    return summary


def main() -> None:
    ap = argparse.ArgumentParser(description="§12.11 Shapley spot-check runner.")
    ap.add_argument(
        "--manifest", default="data/processed/manifests/im2gps3k_test.parquet"
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
        "--out-parquet",
        default="data/processed/metrics/shapley_validation.parquet",
    )
    ap.add_argument(
        "--out-table", default="outputs/tables/shapley_validation.csv"
    )
    ap.add_argument("--intervention-type", default="min_across")
    ap.add_argument("--top-k", type=int, default=10,
                    help="§12.11 default: 10 candidate regions per image.")
    ap.add_argument("--n-orderings", type=int, default=200,
                    help="§12.11 default: 200 random orderings.")
    ap.add_argument("--max-images", type=int, default=50,
                    help="§12.11 default: 50 images.")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threshold-km", type=float, default=25.0)
    args = ap.parse_args()

    run(
        manifest_path=Path(args.manifest),
        regions_path=Path(args.regions),
        scores_path=Path(args.scores),
        out_parquet=Path(args.out_parquet),
        out_table=Path(args.out_table),
        intervention_type=args.intervention_type,
        top_k=args.top_k,
        n_orderings=args.n_orderings,
        max_images=args.max_images,
        seed=args.seed,
        threshold_km=args.threshold_km,
    )


if __name__ == "__main__":
    main()
