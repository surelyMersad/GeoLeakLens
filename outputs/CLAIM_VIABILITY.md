# GeoLeakLens — Claim Viability Note (§17 MVP)

**State as of branch `e0-smoke-pipeline`, all results on Im2GPS3k at n=100 conditional set unless noted.**

This is the §17 MVP criterion 11 deliverable: whether each paper claim looks viable from the v3 evidence (n=100 scale-up). Written for the paper writer; experiment runners and source code in this repo are the authoritative artifacts.

---

## §1.3 Main paper claims — viability assessment

### Claim 1 — "Single-image geolocation leakage is concentrated on a small number of semantic regions" (sparsity)

**Viable. Statistically supported at every budget tested (n=100).**

| Comparison | Δ Acc@25km drop | 95% paired-bootstrap CI | §17 gate |
|---|---:|---:|---|
| dynamic_greedy vs random @ 1% budget | **+22.0pp** | [+14.0, +30.0] | **PASS** |
| dynamic_greedy vs random @ 2% budget | **+23.0pp** | [+15.0, +32.0] | **PASS** |
| dynamic_greedy vs random @ 5% budget | **+22.0pp** | [+13.0, +32.0] | **PASS** |
| dynamic_greedy vs random @ 10% budget | **+14.0pp** | [+7.0, +22.0] | **PASS** |
| dynamic_greedy vs random @ 15% budget | **+10.0pp** | [+3.0, +18.0] | **PASS** |
| dynamic_greedy vs random @ 20% budget | **+9.0pp** | [+1.0, +18.0] | **PASS** |

**Every budget passes §17's "≥5pp drop AND CI excludes zero" gate.** Effect size at the canonical 5% budget grew from +13.3pp at n=30 to +22.0pp at n=100 — the bigger sample punched through the gallery-snap noise floor.

Important framing point: **static_greedy and largest do NOT statistically beat random** at any budget (paired-difference CI includes zero throughout). Only **dynamic_greedy** does. This empirically confirms §11.3's framing that single-region attribution is a "fast approximation, not the headline" — only the joint-causal dynamic-greedy method clears the bar.

| Budget | dynamic_greedy Acc@25km after | random | largest | static |
|---|--:|--:|--:|--:|
| 1% | **69.0** | 91.0 | 93.0 | 94.0 |
| 5% | **67.0** | 89.0 | 87.0 | 86.0 |
| 10% | **66.0** | 80.0 | 80.0 | 83.0 |
| 20% | **66.0** | 75.0 | 78.0 | 77.0 |

Sources: `outputs/tables/e2_sparsity_v3.csv`, `outputs/tables/e2_sparsity_v3_bootstrap_paired.csv`.

### Claim 2 — "The leaky regions are not just obvious identifiers (text, faces) — environmental cues carry substantial leakage"

**Viable. Strongly supported.**

Top-5 dominant-bucket distribution on n=100 conditional set (`min_across` ranking):

| Bucket | n_regions | mean Δ | top-5 % |
|---|--:|--:|--:|
| storefront_commercial | 9 | +0.576 | 0.2 |
| road_marking | 19 | +0.258 | 1.8 |
| license_plate_vehicle | 132 | +0.135 | 9.8 |
| **explicit_text_signage** | 154 | **+0.123** | 9.0 |
| vegetation | 143 | +0.071 | 6.8 |
| utility_pole_wires | 187 | +0.053 | 11.4 |
| sidewalk_curb | 204 | +0.045 | 15.8 |
| sky_skyline | 70 | +0.048 | 7.2 |
| building_architecture | 258 | +0.031 | 14.8 |
| face_person | 196 | +0.017 | 16.2 |
| terrain_mountain_water | 92 | -0.002 | 7.0 |

**Environmental top-5 share: ~65%** (target ≥30%). Buckets in {building, sidewalk, utility_pole, vegetation, sky, terrain, storefront, road_marking} sum to ~65% of top-5 dominants.

**Important nuance vs the n=30 v0+ result:** at n=100, **explicit_text_signage now has positive mean attribution (+0.123)** instead of negative. Text *does* carry leakage signal at scale — just less per-region than environmental cues, and dominated in count by environmental buckets. The paper framing should be:

> "Environmental cues dominate the top-K count of leaky regions. Text and face buckets do carry positive but smaller per-region signal; the paper's 'non-obvious cues' claim rests on the count-share, not on text being non-leaky."

Source: `outputs/tables/cue_taxonomy_v3.csv`, `outputs/figures/cue_taxonomy_bar_v3.png`.

**Caveat — text-light subset:** 96/100 images have <2% OCR coverage (median text fraction = 0.0000). The §13.E3 spec wants a contrast between text-rich and text-light; at this distribution there's effectively no text-rich subset to contrast against, so the "even more strongly without text" framing is uninformative. The headline claim doesn't need it.

**Caveat — secondary-overlap panel** is still 0% across buckets because §9.4 object detection isn't wired. This is a presentation-only gap; the headline "where leakage lives" claim is the dominant-bucket bar chart and that one is fine.

### Claim 3 — "Causally-targeted redaction yields a better privacy-utility frontier than naive baselines"

**Viable. Strict Pareto dominance at n=100.**

§13.E4 Panel A:

| Method @ 5% budget | mean CLIP sim | Acc@25km drop | mean error increase (km) |
|---|--:|--:|--:|
| **dynamic_greedy** | **0.964** | **33.0pp** | **+1029** |
| largest | 0.930 | 13.0pp | +316 |
| random | 0.900 | 11.0pp | +276 |
| static_greedy | 0.907 | 14.0pp | +387 |

dynamic_greedy delivers **~3× the privacy gain at HIGHER CLIP similarity than any baseline** — strict Pareto dominance, not just on the frontier. This is unchanged in direction from the n=30 result and the effect size has held / grown.

Note: **at n=100 the mean-error-increase metric finally differentiates** (median is still 0 because <50% of images move past 25 km). The §13.E4 spec's primary "median error increase" metric is still uninformative on Im2GPS3k at this scale, but mean error increase tells the same story (1029 km for dynamic vs 276–387 km for baselines at 5% budget).

Source: `outputs/tables/redaction_method_comparison_v3.csv`, `outputs/figures/privacy_utility_panelA_v3.png`.

---

## Adjacent findings worth foregrounding

### GPT-4o has bifurcated commit behavior + research framing makes it WORSE

Two §13.E1/E7 findings about closed-VLM threat models:

1. **GPT-4o refuses 54% of random Im2GPS3k images** (`geo_neutral_v1` prompt) — returns valid JSON with `null` lat/lon and confidence < 0.2.
2. On the **46 it commits to**, it's **dramatically accurate**: median 1.3 km, Acc@25km 89%. Far better than GeoCLIP's 32.2% on the same dataset.
3. **Research-study framing (`geo_json_v1`) makes refusals STRONGER, not weaker.** On 17 images sampled, 8 went from committed→refused, 0 went the other way. The refusal mode shifts from "JSON with nulls" to "empty content" — a stricter refusal entirely.
4. **Refusal correlates with image identifiability.** On the GeoCLIP-easy E5 conditional set (median GeoCLIP error = 1.97 km — iconic landmarks), GPT-4o refuses **96% of the time** (vs 54% on random sample). The two adversaries are inversely correlated on commit behavior.

**Recommended paper framing:** "Closed VLMs exhibit a refusal–accuracy tradeoff: under casual prompts they refuse on ~half of geolocation queries but answer the rest with high precision. Research-style framing increases the refusal rate further, not lower. A determined adversary using either GeoCLIP or a VLM independently catches a different population of leaked images."

### Static greedy is moderate-not-strict approximation of joint causal effect

§12.11 Shapley spot-check at n=30 / K=5 / N=100: top-1 agreement 63%, mean Spearman 0.42 between static-greedy ranking and Shapley ranking. Below the §12.11 acceptance bar of (≥0.7, ≥0.7), but above chance. The bootstrap-CI evidence on E2 v3 reinforces this: only dynamic_greedy beats random with paired-CI excluding zero; static greedy doesn't (CI includes zero at every budget).

### `min_across_interventions` is empirically justified

§13.E6 ablation: mean_mask + blur agree (Spearman 0.57); inpaint is the outlier (0.19/0.29 vs the others). The three interventions don't agree well enough to trust any single one — the §10.7 conservative `min_across` is the right choice for headline numbers.

---

## §17 MVP gate — final accounting at n=100

| § | Criterion | Status |
|---|---|---|
| 1 | 500-image manifest per dataset | ⚠️ Im2GPS3k 2996 ✓, 100-image conditional set; GSV-Cities not ingested |
| 2 | Baseline geo table for GeoCLIP + 1 VLM | ✅ |
| 3 | ≥90% region masks | ✅ 100% |
| 4 | Per-region attribution scores | ✅ 5856 rows (3 interventions + min_across, n=100) |
| 5 | Dynamic-greedy joint-causal results | ✅ E2 v3 |
| 6 | Shapley spot-check 50 images | ⚠️ at v0 scope (n=30 K=5 N=100); below acceptance threshold but directionally correct |
| **7** | **Sparsity curve with paired-bootstrap CIs** | **✅ §17 STAT-SIG GATE PASSES at every budget (1%, 2%, 5%, 10%, 15%, 20%)** |
| 8 | Cue taxonomy + text-light subset | ✅ taxonomy strong; text-light contrast uninformative because 96/100 already text-light |
| 9 | Privacy-utility Pareto Panel A | ✅ strict Pareto dominance |
| 10 | 10 qualitative examples | ✅ `outputs/figures/e2_qualitative_gallery_v3.png` |
| 11 | Written claim-viability note | ✅ this file |

**Three core paper claims are viable with n=100 data, with statistical significance at every budget for Claim 1, and effect sizes that grew vs the n=30 pilot.**

## Out of scope / deferred

- §9.4 object detection (would populate E3 secondary-overlap panel; headline claim doesn't need it)
- GSV-Cities ingest (would give §18 multi-dataset criterion; one-dataset MVP holds first)
- §12.10 GeoShield baselines (Panel B in E4)
- §13.E5 model transfer at the right intersection conditional set
- §13.E8 inferential-privacy pilot (Path A: cut)
- §12.11 Shapley v1 at K=10 / N=200 / n=50 (current is K=5 / N=100 / n=30)

---

## Output file index

| File | What |
|---|---|
| `outputs/tables/e2_sparsity_v3.csv` | Sparsity table per (method, budget), n=100 |
| `outputs/tables/e2_sparsity_v3_bootstrap_marginal.csv` | Marginal bootstrap CIs per cell |
| `outputs/tables/e2_sparsity_v3_bootstrap_paired.csv` | Paired-difference bootstrap CIs vs random |
| `outputs/figures/e2_sparsity_curve_v3.png` | Sparsity curve plot |
| `outputs/tables/cue_taxonomy_v3.csv` | E3 bucket aggregation |
| `outputs/tables/cue_taxonomy_v3_textlight.csv` | E3 text-light subset (96/100 images) |
| `outputs/figures/cue_taxonomy_bar_v3.png` | E3 bar chart |
| `outputs/tables/redaction_method_comparison_v3.csv` | E4 Pareto data |
| `outputs/figures/privacy_utility_panelA_v3.png` | E4 Pareto plot |
| `outputs/figures/e2_qualitative_gallery_v3.png` | 10-image §17 gallery (5 success / 5 failure) |
| `outputs/tables/baseline_geolocation_accuracy.csv` | E1 baseline (GeoCLIP + GPT-4o) |
| `outputs/tables/intervention_ablation.csv` | E6 mean_mask vs blur vs inpaint agreement |
| `outputs/tables/prompt_sensitivity.csv` | E7 partial (n=17 — clear signal, killed early) |
| `outputs/tables/shapley_validation.csv` | §12.11 spot-check (n=30 K=5 N=100) |
