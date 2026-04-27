"""§13.E5 model transfer runner — do GeoCLIP-derived redactions transfer to GPT-4o?

The §13.E5 question:
  "Are causal regions model-specific, or do they remove generally
   location-informative evidence?"

If GeoCLIP-picked regions also break GPT-4o's accuracy on the same image,
the regions are model-agnostic — they remove *underlying* location
evidence, not GeoCLIP-specific artifacts. That's the §11.3 paper claim
that semantic redaction outperforms perturbation-based defenses.

Procedure
---------
1. Source = GeoCLIP. Use cached dynamic_greedy selections from E2 v2.
2. Target = GPT-4o. Run on baseline (unredacted) and on the redacted
   image at each requested budget.
3. Restrict the transfer measurement to images where GPT-4o committed
   on baseline — only those have a meaningful "before/after" Acc@25km
   to compare.

Scope of v0
-----------
- 30-image conditional set from E2 v2 (cached).
- Source = GeoCLIP dynamic_greedy (cached selections).
- Target = GPT-4o, geo_neutral_v1 prompt (matches E1).
- Budgets: configurable via --budgets, default 0.10 (single point).
- Cost-bounded: 30 baseline + 30×|budgets| redacted calls.

Outputs
-------
  data/processed/predictions/openai/E5_im2gps3k_baseline.jsonl
  data/processed/predictions/openai/E5_im2gps3k_redacted_b{B}.jsonl
  outputs/tables/model_transfer_matrix.csv
"""
from __future__ import annotations

import argparse
import io
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

try:
    from dotenv import load_dotenv  # type: ignore
    load_dotenv()
except ImportError:
    pass

from geoleaklens.data.geo_utils import haversine_km, success_at
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.models.openai_wrapper import OpenAIVLMWrapper
from geoleaklens.models.prompts import GEO_NEUTRAL_V1
from geoleaklens.scoring.parse_predictions import parse_geolocation_response


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_mask_npz(path: Path) -> np.ndarray:
    npz = np.load(path)
    packed = npz["packed"]
    shape = tuple(int(x) for x in npz["shape"])
    bits = np.unpackbits(packed)[: int(np.prod(shape))]
    return bits.astype(bool).reshape(shape)


def _load_dynamic_selections(path: Path) -> dict[tuple[str, float], list[str]]:
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


def _load_predictions_jsonl(path: Path) -> dict[str, dict]:
    out: dict[str, dict] = {}
    if not path.exists():
        return out
    with open(path) as f:
        for line in f:
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if "image_id" in row:
                out[row["image_id"]] = row
    return out


def _build_redacted_bytes(
    image_id: str,
    selected_ids: list[str],
    regions: pd.DataFrame,
    pil_image: Image.Image,
) -> bytes:
    if not selected_ids:
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    sel = regions[regions["region_id"].isin(selected_ids)]
    w, h = pil_image.size
    union = np.zeros((h, w), dtype=bool)
    for _, row in sel.iterrows():
        m = _load_mask_npz(Path(row["mask_path"]))
        if m.shape == (h, w):
            union |= m
    if not union.any():
        buf = io.BytesIO()
        pil_image.save(buf, format="JPEG", quality=92)
        return buf.getvalue()
    edited, _ = mean_mask(pil_image, union, mode="local")
    buf = io.BytesIO()
    edited.save(buf, format="JPEG", quality=92)
    return buf.getvalue()


def _query_gpt4o(
    image_bytes: bytes,
    wrapper: OpenAIVLMWrapper,
) -> dict:
    raw = wrapper.predict(
        image_bytes=image_bytes,
        prompt=GEO_NEUTRAL_V1,
        prompt_id="geo_neutral_v1",
    )
    parsed = parse_geolocation_response(raw["raw_response"])
    return {
        "lat": parsed["parsed"].get("lat"),
        "lon": parsed["parsed"].get("lon"),
        "parse_success": parsed["parse_success"],
        "raw_response": parsed["raw_response"],
        "cache_hit": raw.get("cache_hit", False),
        "error": parsed.get("error") or raw.get("error"),
    }


def run(
    *,
    manifest_path: Path,
    regions_path: Path,
    dynamic_jsonl: Path,
    e2_redactions_path: Path,
    baseline_jsonl: Path,
    redacted_jsonl_template: str,
    table_path: Path,
    budgets: list[float],
    threshold_km: float,
    openai_model: str,
) -> dict:
    manifest = pd.read_parquet(manifest_path).set_index("image_id")
    regions = pd.read_parquet(regions_path)
    dynamic_selections = _load_dynamic_selections(dynamic_jsonl)
    e2_redactions = pd.read_parquet(e2_redactions_path)

    image_ids = sorted(set(regions["image_id"]))
    print(f"[e5] target eval on {len(image_ids)} conditional-set images")

    wrapper = OpenAIVLMWrapper(model=openai_model)

    # --- 1. Baseline GPT-4o on the 30 conditional-set images -----------
    baseline_jsonl.parent.mkdir(parents=True, exist_ok=True)
    cached_baseline = _load_predictions_jsonl(baseline_jsonl)
    baseline_predictions: dict[str, dict] = dict(cached_baseline)
    mode = "a" if cached_baseline else "w"
    with open(baseline_jsonl, mode) as f:
        for i, image_id in enumerate(image_ids):
            if image_id in cached_baseline:
                continue
            if image_id not in manifest.index:
                continue
            image_path = Path(manifest.loc[image_id, "image_path"])
            if not image_path.exists():
                continue
            with open(image_path, "rb") as ip:
                img_bytes = ip.read()
            result = _query_gpt4o(img_bytes, wrapper)
            entry = {
                "image_id": image_id,
                "variant": "baseline",
                "model_name": openai_model,
                "prompt_id": "geo_neutral_v1",
                "lat": result["lat"], "lon": result["lon"],
                "parse_success": result["parse_success"],
                "raw_response": result["raw_response"],
                "created_at": _now_iso(),
                "error": result["error"],
                "cache_hit": result["cache_hit"],
            }
            f.write(json.dumps(entry) + "\n")
            f.flush()
            baseline_predictions[image_id] = entry
            if (i + 1) % 5 == 0:
                print(f"[baseline] {i + 1}/{len(image_ids)} done")

    # --- 2. Redacted GPT-4o per budget --------------------------------
    redacted_predictions: dict[float, dict[str, dict]] = {}
    for budget in budgets:
        path = Path(redacted_jsonl_template.format(budget=int(budget * 100)))
        cached = _load_predictions_jsonl(path)
        redacted_predictions[budget] = dict(cached)
        path.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if cached else "w"
        with open(path, mode) as f:
            for i, image_id in enumerate(image_ids):
                if image_id in cached:
                    continue
                if image_id not in manifest.index:
                    continue
                image_path = Path(manifest.loc[image_id, "image_path"])
                if not image_path.exists():
                    continue
                pil = Image.open(image_path).convert("RGB")
                selected = dynamic_selections.get((image_id, float(budget)), [])
                edit_bytes = _build_redacted_bytes(
                    image_id, selected, regions, pil
                )
                result = _query_gpt4o(edit_bytes, wrapper)
                entry = {
                    "image_id": image_id,
                    "variant": f"dynamic_greedy_b{int(budget*100)}",
                    "method": "dynamic_greedy",
                    "target_area_frac": float(budget),
                    "model_name": openai_model,
                    "prompt_id": "geo_neutral_v1",
                    "lat": result["lat"], "lon": result["lon"],
                    "parse_success": result["parse_success"],
                    "raw_response": result["raw_response"],
                    "n_selected": len(selected),
                    "created_at": _now_iso(),
                    "error": result["error"],
                    "cache_hit": result["cache_hit"],
                }
                f.write(json.dumps(entry) + "\n")
                f.flush()
                redacted_predictions[budget][image_id] = entry
                if (i + 1) % 5 == 0:
                    print(f"[redacted b={int(budget*100)}%] {i + 1}/{len(image_ids)} done")

    # --- 3. Build transfer matrix -------------------------------------
    rows: list[dict] = []
    for budget in budgets:
        # Compute target=GPT-4o accuracy at this budget restricted to
        # images where GPT-4o committed on baseline.
        committed_ids = {
            iid for iid, p in baseline_predictions.items()
            if p.get("lat") is not None
        }
        n_committed = len(committed_ids)
        n_with_eval = 0
        n_target_success = 0
        n_target_baseline_success = 0
        target_baseline_acc = 0
        for iid in committed_ids:
            base = baseline_predictions[iid]
            true_lat = float(manifest.loc[iid, "lat"])
            true_lon = float(manifest.loc[iid, "lon"])
            err_base = haversine_km(base["lat"], base["lon"], true_lat, true_lon)
            if success_at(err_base, threshold_km):
                n_target_baseline_success += 1
            red = redacted_predictions[budget].get(iid)
            if red is None:
                continue
            n_with_eval += 1
            if red.get("lat") is None:
                continue
            err_red = haversine_km(red["lat"], red["lon"], true_lat, true_lon)
            if success_at(err_red, threshold_km):
                n_target_success += 1
        target_baseline_acc = (
            n_target_baseline_success / n_committed if n_committed else 0.0
        )

        # Cached source (GeoCLIP) Acc@25km drop at this budget — read from
        # the E2 v2 redactions parquet. Note that this is across the full
        # 30 conditional-set images, not restricted to committed_ids.
        src_at_budget = e2_redactions[
            (e2_redactions["method"] == "dynamic_greedy")
            & (e2_redactions["target_area_frac"] == budget)
        ]
        src_baseline_acc = 1.0  # baseline was 100% by construction (conditional)
        src_after_acc = float(src_at_budget["edited_success_25km"].mean()) if not src_at_budget.empty else float("nan")

        # Target Acc restricted to GPT-4o-committed subset.
        tgt_after_acc = (
            n_target_success / n_committed if n_committed else 0.0
        )

        rows.append({
            "source_model": "geoclip",
            "target_model": openai_model,
            "method": "dynamic_greedy",
            "target_area_frac": float(budget),
            "n_total": int(len(image_ids)),
            "n_target_committed_baseline": int(n_committed),
            "src_baseline_acc25km": src_baseline_acc,
            "src_after_redaction_acc25km": src_after_acc,
            "src_drop": (src_baseline_acc - src_after_acc) if src_after_acc == src_after_acc else float("nan"),
            "tgt_baseline_acc25km": target_baseline_acc,
            "tgt_after_redaction_acc25km": tgt_after_acc,
            "tgt_drop": target_baseline_acc - tgt_after_acc,
        })

    summary = pd.DataFrame(rows)
    table_path.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(table_path, index=False, float_format="%.4f")
    print(f"[e5] wrote {table_path}")

    print("\n=== §13.E5 model transfer matrix ===")
    show = summary.copy()
    for c in ("src_baseline_acc25km", "src_after_redaction_acc25km", "src_drop",
              "tgt_baseline_acc25km", "tgt_after_redaction_acc25km", "tgt_drop",
              "target_area_frac"):
        show[c] = show[c].map("{:5.1%}".format)
    print(show.to_string(index=False))

    return {
        "n_images": int(len(image_ids)),
        "budgets": budgets,
        "table": str(table_path),
    }


def main() -> None:
    ap = argparse.ArgumentParser(description="§13.E5 model transfer runner.")
    ap.add_argument("--manifest", default="data/processed/manifests/im2gps3k_test.parquet")
    ap.add_argument("--regions", default="data/processed/regions/E2_im2gps3k_sam.parquet")
    ap.add_argument(
        "--dynamic-jsonl",
        default="data/processed/predictions/geoclip/E2_im2gps3k_redacted_v2__dynamic.jsonl",
    )
    ap.add_argument(
        "--e2-redactions",
        default="data/processed/redactions/E2_im2gps3k_v2.parquet",
    )
    ap.add_argument(
        "--baseline-jsonl",
        default="data/processed/predictions/openai/E5_im2gps3k_baseline.jsonl",
    )
    ap.add_argument(
        "--redacted-jsonl-template",
        default="data/processed/predictions/openai/E5_im2gps3k_redacted_b{budget}.jsonl",
    )
    ap.add_argument(
        "--table", default="outputs/tables/model_transfer_matrix.csv"
    )
    ap.add_argument(
        "--budgets", default="0.10",
        help="Comma-separated dynamic_greedy budgets to evaluate transfer at.",
    )
    ap.add_argument("--threshold-km", type=float, default=25.0)
    ap.add_argument("--openai-model", default="gpt-4o")
    args = ap.parse_args()

    run(
        manifest_path=Path(args.manifest),
        regions_path=Path(args.regions),
        dynamic_jsonl=Path(args.dynamic_jsonl),
        e2_redactions_path=Path(args.e2_redactions),
        baseline_jsonl=Path(args.baseline_jsonl),
        redacted_jsonl_template=args.redacted_jsonl_template,
        table_path=Path(args.table),
        budgets=[float(x) for x in args.budgets.split(",")],
        threshold_km=args.threshold_km,
        openai_model=args.openai_model,
    )


if __name__ == "__main__":
    main()
