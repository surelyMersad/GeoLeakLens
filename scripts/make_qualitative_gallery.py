"""§17 MVP criterion 10 — 10 qualitative examples (mix of successes/failures).

Uses cached E2 v2 dynamic_greedy redactions. Picks 5 "successes" (images
where the redaction broke Acc@25km) and 5 "failures" (where the
prediction stayed within 25km despite redaction). Side-by-side panels:

    [original photo]      [redacted photo]
    GeoCLIP: lat/lon       GeoCLIP: lat/lon
    err = X km             err = Y km                   ← labeled
                                                          success
                                                          / failure

Output: outputs/figures/e2_qualitative_gallery.png
"""
from __future__ import annotations

import io
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image

from geoleaklens.data.geo_utils import haversine_km, success_at
from geoleaklens.interventions.mask import mean_mask
from geoleaklens.redaction.optimize import DynamicGreedy
from geoleaklens.redaction.methods import StaticGreedy
from geoleaklens.redaction.baselines import LargestRegions, RandomRegions


ROOT = Path(__file__).resolve().parents[1]
THRESHOLD_KM = 25.0
BUDGET = 0.05  # the budget where the bootstrap CI excludes zero
OUTPUT_PATH = ROOT / "outputs/figures/e2_qualitative_gallery.png"


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
            try:
                row = json.loads(line)
            except ValueError:
                continue
            iid = row.get("image_id")
            budget = row.get("target_area_frac")
            sel = row.get("selected_region_ids")
            if iid and budget is not None and sel is not None:
                out[(iid, float(budget))] = list(sel)
    return out


def _build_redacted_image(
    image_path: Path,
    selected_ids: list[str],
    regions: pd.DataFrame,
) -> tuple[Image.Image, Image.Image]:
    pil = Image.open(image_path).convert("RGB")
    if not selected_ids:
        return pil, pil.copy()
    sel = regions[regions["region_id"].isin(selected_ids)]
    w, h = pil.size
    union = np.zeros((h, w), dtype=bool)
    for _, row in sel.iterrows():
        m = _load_mask_npz(Path(row["mask_path"]))
        if m.shape == (h, w):
            union |= m
    if not union.any():
        return pil, pil.copy()
    edited, _ = mean_mask(pil, union, mode="local")
    return pil, edited


def main() -> None:
    manifest = pd.read_parquet(ROOT / "data/processed/manifests/im2gps3k_test.parquet").set_index("image_id")
    regions = pd.read_parquet(ROOT / "data/processed/regions/E2_im2gps3k_sam.parquet")
    redactions = pd.read_parquet(ROOT / "data/processed/redactions/E2_im2gps3k_v2.parquet")
    dynamic_selections = _load_dynamic_selections(
        ROOT / "data/processed/predictions/geoclip/E2_im2gps3k_redacted_v2__dynamic.jsonl"
    )

    # Pull baseline GeoCLIP predictions (E1) so we can render orig lat/lon.
    e1_path = ROOT / "data/processed/predictions/geoclip/E1_im2gps3k.jsonl"
    e1_pred: dict[str, dict] = {}
    if e1_path.exists():
        with open(e1_path) as f:
            for line in f:
                r = json.loads(line)
                e1_pred[r["image_id"]] = r["parsed"]

    # Filter to dynamic_greedy at the headline budget.
    sub = redactions[
        (redactions["method"] == "dynamic_greedy")
        & np.isclose(redactions["target_area_frac"], BUDGET)
    ].copy()
    if sub.empty:
        raise SystemExit(f"no dynamic_greedy rows at budget {BUDGET}")

    # Successes: edited_success_25km == False (redaction broke it). Sort
    # by edited_error_km desc to surface the most-impactful examples.
    successes = (
        sub[~sub["edited_success_25km"].astype(bool)]
        .sort_values("edited_error_km", ascending=False)
        .head(5)
    )
    # Failures: edited_success_25km == True (redaction didn't break it).
    # Sort by actual_area_frac desc so we show "we tried hard but it
    # held" cases.
    failures = (
        sub[sub["edited_success_25km"].astype(bool)]
        .sort_values("actual_area_frac", ascending=False)
        .head(5)
    )
    print(f"successes available: {len(successes)}, failures: {len(failures)}")

    def _row_panel(ax_pair, row, label):
        image_id = row["image_id"]
        if image_id not in manifest.index:
            for ax in ax_pair:
                ax.set_axis_off()
            return
        manifest_row = manifest.loc[image_id]
        image_path = Path(manifest_row["image_path"])
        if not image_path.exists():
            for ax in ax_pair:
                ax.set_axis_off()
            return

        true_lat = float(manifest_row["lat"])
        true_lon = float(manifest_row["lon"])
        selected = dynamic_selections.get((image_id, BUDGET), [])
        orig_pil, edit_pil = _build_redacted_image(image_path, selected, regions)

        # Original prediction.
        op = e1_pred.get(image_id, {})
        orig_lat = op.get("lat")
        orig_lon = op.get("lon")
        if orig_lat is not None and orig_lon is not None:
            orig_err = haversine_km(orig_lat, orig_lon, true_lat, true_lon)
            orig_label = f"GeoCLIP ({orig_lat:.2f}, {orig_lon:.2f})\nerr = {orig_err:.1f} km"
        else:
            orig_label = "GeoCLIP: no parse"

        edit_err = float(row["edited_error_km"])
        edit_label = (
            f"after dynamic_greedy @ {BUDGET*100:.0f}% area "
            f"(actual {row['actual_area_frac']*100:.1f}%)\n"
            f"err = {edit_err:.1f} km"
        )

        place = f"{manifest_row['city']}, {manifest_row['country']}"
        success_color = "tab:green" if label == "SUCCESS" else "tab:red"

        ax_pair[0].imshow(np.asarray(orig_pil))
        ax_pair[0].set_axis_off()
        ax_pair[0].set_title(
            f"{place}  •  truth ({true_lat:.2f}, {true_lon:.2f})\n{orig_label}",
            fontsize=8,
        )
        ax_pair[1].imshow(np.asarray(edit_pil))
        ax_pair[1].set_axis_off()
        ax_pair[1].set_title(
            f"[{label}]  {edit_label}",
            fontsize=8, color=success_color,
        )

    n_rows = 5  # 5 success rows on top, 5 failure rows below — but show in 5×2 grid? No, do 10x2.
    fig, axes = plt.subplots(10, 2, figsize=(13, 28), constrained_layout=True)
    fig.suptitle(
        "§17 MVP qualitative gallery — dynamic_greedy @ 5% budget on Im2GPS3k\n"
        "(GREEN = redaction broke Acc@25km; RED = prediction held)",
        fontsize=12,
    )
    rows_to_show = list(successes.iterrows()) + list(failures.iterrows())
    labels = ["SUCCESS"] * len(successes) + ["FAILURE"] * len(failures)
    for i in range(10):
        if i < len(rows_to_show):
            _, row = rows_to_show[i]
            _row_panel(axes[i], row, labels[i])
        else:
            for ax in axes[i]:
                ax.set_axis_off()

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
