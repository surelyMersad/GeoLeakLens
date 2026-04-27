import pandas as pd

from geoleaklens.redaction.baselines import LargestRegions, RandomRegions


def _regions(image_id: str, sizes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "image_id": image_id,
                "region_id": f"r{i}",
                "area_frac": s,
            }
            for i, s in enumerate(sizes)
        ]
    )


# ----- §12.4 LargestRegions ------------------------------------------------

def test_largest_picks_biggest_first():
    regs = _regions("img", [0.05, 0.20, 0.01, 0.12, 0.03])
    sel = LargestRegions().select_regions(
        "img", regs, None, budget={"area_frac": 0.40}
    )
    # Sorted desc: [0.20, 0.12, 0.05, 0.03, 0.01]
    # Greedy fit at budget 0.40: 0.20 → 0.32 → 0.37 → 0.40 (cap reached).
    assert sel[:3] == ["r1", "r3", "r0"]
    # Total area should not overshoot.
    total = sum(regs.set_index("region_id").loc[sel, "area_frac"])
    assert total <= 0.40


def test_largest_skips_oversized_regions():
    regs = _regions("img", [0.45, 0.05, 0.08])
    sel = LargestRegions().select_regions(
        "img", regs, None, budget={"area_frac": 0.10}
    )
    # 0.45 too big → skip; 0.08 fits → take; 0.05 won't fit (would be 0.13).
    assert sel == ["r2"]


def test_largest_zero_budget_returns_empty():
    regs = _regions("img", [0.05, 0.10])
    sel = LargestRegions().select_regions(
        "img", regs, None, budget={"area_frac": 0.0}
    )
    assert sel == []


def test_largest_filters_to_image():
    regs = pd.DataFrame(
        [
            {"image_id": "a", "region_id": "ra", "area_frac": 0.05},
            {"image_id": "b", "region_id": "rb", "area_frac": 0.30},
        ]
    )
    sel = LargestRegions().select_regions(
        "a", regs, None, budget={"area_frac": 0.10}
    )
    assert sel == ["ra"]


# ----- §12.3 RandomRegions -------------------------------------------------

def test_random_respects_area_budget_and_tolerance():
    regs = _regions("img", [0.04, 0.04, 0.04, 0.04, 0.04, 0.04])  # 6×4%
    sel = RandomRegions(seed=0).select_regions(
        "img", regs, None, budget={"area_frac": 0.10}
    )
    # 10% target with ±0.5pp tolerance → cap = 10.5%. Pick at most 2 regions
    # (8% accumulated; 12% would overshoot 10.5%).
    total = sum(regs.set_index("region_id").loc[sel, "area_frac"])
    assert total <= 0.105 + 1e-9
    assert len(sel) <= 2


def test_random_different_seeds_can_pick_different_regions():
    regs = _regions("img", [0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    sel1 = set(RandomRegions(seed=1).select_regions(
        "img", regs, None, budget={"area_frac": 0.10}
    ))
    seen_other = False
    for s in range(2, 30):
        sel = set(RandomRegions(seed=s).select_regions(
            "img", regs, None, budget={"area_frac": 0.10}
        ))
        if sel != sel1:
            seen_other = True
            break
    assert seen_other, "RandomRegions must produce seed-dependent picks"


def test_random_same_seed_is_reproducible():
    regs = _regions("img", [0.04, 0.04, 0.04, 0.04, 0.04, 0.04])
    sel1 = RandomRegions(seed=42).select_regions(
        "img", regs, None, budget={"area_frac": 0.10}
    )
    sel2 = RandomRegions(seed=42).select_regions(
        "img", regs, None, budget={"area_frac": 0.10}
    )
    assert sel1 == sel2


def test_random_zero_budget_returns_empty():
    regs = _regions("img", [0.05, 0.10])
    sel = RandomRegions(seed=0).select_regions(
        "img", regs, None, budget={"area_frac": 0.0}
    )
    assert sel == []


def test_random_skips_oversized_regions():
    """A region larger than the budget+tolerance must be skipped, not picked."""
    regs = _regions("img", [0.50, 0.02, 0.03])  # one big, two small
    # Budget 0.10 ± 0.5pp → cap 0.105. The 0.50 region must be skipped on
    # every seed.
    for seed in range(20):
        sel = RandomRegions(seed=seed).select_regions(
            "img", regs, None, budget={"area_frac": 0.10}
        )
        total = sum(regs.set_index("region_id").loc[sel, "area_frac"])
        assert total <= 0.105 + 1e-9
        assert "r0" not in sel  # the 0.50 region's id


def test_random_filters_to_image():
    regs = pd.DataFrame(
        [
            {"image_id": "a", "region_id": "ra", "area_frac": 0.05},
            {"image_id": "b", "region_id": "rb", "area_frac": 0.05},
        ]
    )
    sel = RandomRegions(seed=0).select_regions(
        "a", regs, None, budget={"area_frac": 0.10}
    )
    assert sel == ["ra"]
