# experiments.md — Implementation spec for GeoLeakLens

## Project

**Working title:** GeoLeakLens: Causal Redaction of Location-Leaking Visual Evidence for Inferential Image Privacy

This document is intended for a coding agent implementing the experiments for the paper. Treat it as the source of truth for repository structure, data schemas, experimental protocol, metrics, baselines, plots, and acceptance checks.

---

## 1. Paper context the coding agent must understand

### 1.1 Core idea

We are studying **inferential image privacy**: private information that can be inferred from visual context, even when explicit metadata and obvious identifiers are removed.

The paper starts with **geolocation** because it has objective ground truth and measurable error in kilometers, but the broader motivation is inferential privacy: VLMs can infer private attributes from benign images.

The core hypothesis:

> Image geolocation leakage is often caused by a sparse set of semantically meaningful visual cues. If we identify those cues with counterfactual interventions, we can redact a small amount of image content and reduce private inference much more efficiently than standard privacy edits.

### 1.2 What makes this different from GeoShield

GeoShield is the closest prior work. It protects geolocation privacy using **imperceptible adversarial perturbations**. It includes feature disentanglement, exposure-element identification, and scale-adaptive perturbation optimization.

Our paper is different. We are not primarily adding adversarial noise. We are asking:

> Which semantic visual regions actually caused the model to infer the private attribute?

Then we remove/edit those regions.

So the method must output:

1. The original model geolocation result.
2. A semantic region map.
3. A causal leakage score for each region.
4. A targeted redacted image.
5. A privacy-utility comparison against baselines.

If the implementation only blurs signs or runs SAM + blur, it is **not enough**. The experiments must prove causal, semantic, interpretable redaction.

### 1.3 Main paper claims

Every experiment should support at least one of these:

**Claim 1 — Leakage sparsity**  
A small number of semantic regions accounts for a large fraction of geolocation leakage.

**Claim 2 — Non-obvious cues**  
Many high-leakage regions are not faces, license plates, or visible text, but environmental cues such as architecture, road markings, vegetation, storefront style, utility poles, mountains, skyline, and road/sidewalk texture.

**Claim 3 — Better privacy-utility tradeoff**  
Targeted causal semantic redaction reduces geolocation leakage more efficiently than standard privacy edits and adversarial perturbation baselines.

### 1.4 Ethical guardrails

This is a defensive privacy project.

Implementation constraints:

- Use only public, licensed, benchmark-style geotagged images.
- Do not build or expose an app that accepts arbitrary user images and returns precise addresses.
- Do not evaluate on private social media images unless explicit consent and IRB-like review exist.
- Public demo should show leakage level, cue map, and redacted image; not "here is your exact address."
- Strip EXIF metadata from all images before model queries.
- Ensure filenames and paths do not contain location information that could leak into prompts or logs.

---

## 2. Key references and why they matter

Use these for related-work positioning and baselines.

### 2.1 GeoShield

- Paper: https://arxiv.org/abs/2508.03209
- Official-ish code: https://github.com/thinwayliu/Geoshield
- Why it matters: closest geoprivacy defense baseline.
- GeoShield is perturbation-based. It generates imperceptible adversarial perturbations to protect against VLM geolocation.
- Our comparison must show that GeoLeakLens is causal, semantic, interpretable, and robust by removing underlying evidence rather than adding invisible perturbation.

### 2.2 GSV-Cities

- GitHub: https://github.com/amaralibey/gsv-cities
- Kaggle: https://www.kaggle.com/datasets/amaralibey/gsv-cities
- Why it matters: large public street-view geolocation dataset with many cities and place IDs.
- Good default main dataset for MVP.
- Use a stratified subset first, then scale.

### 2.3 GPTGeoChat

- Paper page: https://aclanthology.org/2024.emnlp-main.957/
- arXiv: https://arxiv.org/abs/2407.04952
- GitHub: https://github.com/ethanm88/GPTGeoChat
- Why it matters: frames VLM geolocation as an immediate privacy risk and provides moderation/geolocation-conversation context.
- Use as related work, not necessarily as the main dataset.

### 2.4 GeoCLIP

- Paper: https://proceedings.neurips.cc/paper_files/paper/2023/hash/1b57aaddf85ab01a2445a79c9edc1f4b-Abstract-Conference.html
- GitHub: https://github.com/VicenteVivan/geo-clip
- Why it matters: local image-to-GPS model; useful as an open geolocation adversary and possible saliency baseline.

### 2.5 SAM / SAM2

- SAM GitHub: https://github.com/facebookresearch/segment-anything
- SAM2 GitHub: https://github.com/facebookresearch/sam2
- Why it matters: segmentation engine for semantic regions.

### 2.6 Private attribute inference from images

- NeurIPS 2024 paper: https://proceedings.neurips.cc/paper_files/paper/2024/hash/bb97e9a7c811904c9b01f51fde66edcf-Abstract-Conference.html
- arXiv: https://arxiv.org/abs/2404.10618
- Code/data: https://github.com/eth-sri/llmprivacy
- Why it matters: broader inferential privacy motivation and optional pilot.

---

## 3. Repository structure

Implement this structure.

```text
geoleaklens/
  README.md
  experiments.md

  configs/
    default.yaml
    dataset_gsv_cities.yaml
    models.yaml
    redaction.yaml
    experiment_matrix.yaml

  data/
    raw/
      gsv_cities/
      im2gps3k/
      gptgeochat/
      private_attr_pilot/
    interim/
      resized/
      stripped_exif/
      masks/
      interventions/
    processed/
      manifests/
      predictions/
      regions/
      redactions/
      metrics/
    external/
      geoshield/
      geoclip/

  src/
    geoleaklens/
      __init__.py

      data/
        build_manifest.py
        sample_dataset.py
        strip_exif.py
        image_io.py
        geo_utils.py
        normalize_locations.py

      models/
        base.py
        geoclip_wrapper.py
        vlm_api_wrapper.py
        open_vlm_wrapper.py
        geocoder.py
        clip_wrapper.py
        caption_wrapper.py
        aesthetic_wrapper.py

      segmentation/
        sam_regions.py
        ocr_regions.py
        object_regions.py
        merge_regions.py
        label_regions.py
        visualize_regions.py

      interventions/
        blur.py
        mask.py
        inpaint.py
        replace.py
        apply_intervention.py

      scoring/
        parse_predictions.py
        geolocation_metrics.py
        causal_scores.py
        utility_metrics.py
        bootstrap.py

      redaction/
        optimize.py
        methods.py
        baselines.py

      experiments/
        run_baseline_geolocation.py
        run_region_interventions.py
        run_redaction_methods.py
        run_transfer_eval.py
        run_ablation_interventions.py
        run_prompt_sensitivity.py
        run_private_attr_pilot.py

      plots/
        plot_threat_table.py
        plot_sparsity_curve.py
        plot_cue_taxonomy.py
        plot_privacy_utility.py
        plot_transfer.py
        make_qualitative_figures.py

  scripts/
    00_download_or_link_data.sh
    01_build_manifest.sh
    02_sample_mvp.sh
    03_run_segmentation.sh
    04_run_baseline_geolocation.sh
    05_run_interventions.sh
    06_score_regions.sh
    07_run_redactions.sh
    08_evaluate_methods.sh
    09_make_plots.sh
    10_run_all_mvp.sh

  notebooks/
    sanity_check_masks.ipynb
    inspect_predictions.ipynb
    qualitative_gallery.ipynb

  outputs/
    figures/
    tables/
    logs/
    demo_examples/

  tests/
    test_geo_utils.py
    test_prediction_parser.py
    test_causal_scores.py
    test_region_merge.py
```

---

## 4. Dependencies

Use Python 3.10 or 3.11.

Core dependencies:

```text
torch
torchvision
transformers
accelerate
Pillow
opencv-python
numpy
pandas
pyarrow
scikit-learn
scipy
matplotlib
tqdm
pydantic
omegaconf
hydra-core
rich
pytest
shapely
geopy
pycountry
country_converter
ftfy
regex
sentence-transformers
open_clip_torch
```

Segmentation/OCR/object detection:

```text
segment-anything or sam2
easyocr or paddleocr
ultralytics
groundingdino-py or GroundingDINO repo
```

Inpainting options:

```text
lama-cleaner or simple-lama-inpainting
diffusers   # optional; only for higher-quality inpainting
```

Closed VLM APIs are optional and should be behind wrappers/env vars:

```text
OPENAI_API_KEY
ANTHROPIC_API_KEY
GOOGLE_API_KEY
```

Never hardcode keys.

### 4.1 Cost and time estimation

Cost matters: VLM API queries dominate. Plan budgets per experiment before running.

Order-of-magnitude estimates (assume 1024 px images, ~1k input tokens per query):

| Item | Per call | 500 imgs × 80 regions × 3 interventions | 5000 × 80 × 3 |
|---|---|---|---|
| GPT-4o vision | ~$0.005 | ~$600 | ~$6,000 |
| Gemini 1.5 vision | ~$0.003 | ~$360 | ~$3,600 |
| Claude vision | ~$0.005 | ~$600 | ~$6,000 |
| Qwen-VL local | ~free (GPU time) | depends on hardware | depends on hardware |
| GeoCLIP local | ~free (GPU time) | minutes | hours |

Implications:
- For E1/E2/E3 MVP: GeoCLIP + 1 VLM at smaller VLM N (e.g., 100 images) keeps cost in low hundreds of dollars.
- For E5 (transfer) and E6 (intervention ablation), use 100–200 image subsets — they don't need full N.
- For E4 (Pareto), full N is more important; this is the headline.
- Caching by `(model_name, image_sha256, prompt_id, region_set_hash, intervention_type)` is non-negotiable. Single-region scoring queries should each be cached so re-runs cost nothing.

Add a cost row to each experiment's planning section before running, with the cap and an early-stop trigger (e.g., abort if median cost-per-image exceeds estimate by 2×).

---

## 5. Configuration

Use YAML configs. Every experiment must be reproducible from a config file.

### 5.1 `configs/default.yaml`

```yaml
project:
  name: geoleaklens
  seed: 42
  output_dir: outputs

data:
  manifest_path: data/processed/manifests/gsv_mvp.parquet
  image_root: data/interim/stripped_exif
  cache_dir: data/processed
  max_image_side: 1024
  strip_exif: true

evaluation:
  thresholds_km: [1, 25, 200, 750, 2500]
  main_threshold_km: 25
  n_bootstrap: 1000
  bootstrap_seed: 123
  include_only_original_success_for_causal_region_analysis: true
  include_all_images_for_final_redaction_eval: true

segmentation:
  engine: sam2
  sam_checkpoint: null
  min_area_frac: 0.001
  max_area_frac: 0.45
  dedup_iou_threshold: 0.85
  mask_dilation_px: 5
  add_grid_regions: true
  grid_sizes: [4, 8]
  add_ocr_regions: true
  add_object_regions: true

interventions:
  types: ["blur", "mean_mask", "inpaint"]
  primary_type: "mean_mask"   # predictable failure mode (less info); inpaint is secondary because it can introduce new content
  headline_intervention: "min_across_types"  # for sparsity headline, take min causal effect across mean_mask/blur/inpaint per region
  blur_sigma_frac: 0.035
  mean_mask_use_local_color: true
  inpaint_engine: lama
  mask_dilation_px: 7

models:
  geolocation_adversaries:
    - name: geoclip
      type: local
      enabled: true
    - name: gpt4o
      type: api
      enabled: false
    - name: gemini
      type: api
      enabled: false
    - name: qwen_vl
      type: open_vlm
      enabled: false

  utility:
    clip_model: ViT-L-14
    clip_pretrained: openai
    caption_model: Salesforce/blip-image-captioning-base
    aesthetic_model: null

redaction:
  budgets_area_frac: [0.01, 0.02, 0.05, 0.1, 0.15, 0.2]
  utility_clip_min: 0.85
  score_normalization: leakage_per_area      # primary; leakage_per_utility_cost biases against scenic regions and is reserved for E4 Pareto only
  pareto_score_normalization: leakage_per_utility_cost  # only used for §13.E4 / §15.4
  greedy_recompute: true                     # dynamic greedy is the primary method (§12.2); static greedy (§12.1) is a fast approximation
  random_baseline_repeats: 10
  primary_metric: median_error_increase_km   # see §11.2; Acc@25km kept as secondary view
  conditioning:
    region_analysis: original_success_at_25km   # E2/E3
    final_redaction_eval: all_images            # E4

logging:
  level: INFO
  save_intermediate_images: true
  save_api_raw_responses: true
```

### 5.2 `configs/experiment_matrix.yaml`

```yaml
experiments:
  mvp:
    datasets: ["gsv_cities", "im2gps3k"]   # co-primary; see §6
    n_images_per_dataset: 500
    models: ["geoclip", "gpt4o"]            # GeoCLIP for cheap region scoring; VLM for headline result. See C3.
    vlm_subset_n: 100                       # smaller N for VLM to control cost (§4.1)
    interventions: ["mean_mask", "blur", "inpaint"]
    primary_intervention: "mean_mask"       # see §10.4
    methods: ["geoleaklens_dynamic_greedy", "geoleaklens_static_greedy"]   # dynamic primary, static fast-approx
    baselines: ["random_regions", "largest_regions", "ocr_only", "grid_random", "full_blur"]
    outputs: ["fig_sparsity", "fig_cue_taxonomy", "fig_privacy_utility", "qual_gallery", "shapley_validation_subset"]

  full_geolocation:
    datasets: ["gsv_cities", "im2gps3k"]
    n_images: 5000
    models: ["geoclip", "gpt4o", "gemini", "qwen_vl"]
    interventions: ["mean_mask", "blur", "inpaint"]
    primary_intervention: "mean_mask"
    baselines: ["random_regions", "largest_regions", "ocr_only", "face_plate_blur", "saliency", "geoshield_imperceptible", "geoshield_visible_eps"]
    outputs: ["all"]

  transfer:
    dataset: gsv_cities
    n_images: 1000
    source_models: ["geoclip", "gpt4o"]
    target_models: ["geoclip", "gpt4o", "gemini", "qwen_vl"]

  private_attr_pilot:
    dataset: private_attr_pilot
    n_images: 300
    attributes: ["institution_type", "socioeconomic_proxy", "sensitive_setting"]
```

---

## 6. Data

### 6.1 Main datasets: GSV-Cities + Im2GPS3k (co-primary)

Use **both** datasets as primary. Results on two datasets are substantially stronger than on one, and they cover complementary angles: GSV-Cities for stratified-by-city analysis (its place IDs are the right unit for that), Im2GPS3k for direct comparison with the standard geolocation literature.

**GSV-Cities**

Why:

- Public street-view style data.
- Many cities, place IDs and metadata.
- Enough visual cues for geolocation.
- Good for measuring whether cues are sparse and for stratified-by-city aggregation.

Implementation:

1. Download manually or via Kaggle to `data/raw/gsv_cities/`.
2. Build a manifest with image path, city, country if available, place ID, latitude, longitude if available.
3. If GSV-Cities metadata lacks lat/lon in your local copy, use the metadata fields available. For images without exact coordinates, fall back to city centroid coordinates with a documented uncertainty radius and exclude from sub-25km analyses.
4. Ensure no location information appears in image filename passed to model wrappers.

**Im2GPS3k**

Why:

- Standard geolocation benchmark; results comparable to prior work (GeoCLIP, PIGEON, etc.).
- Diverse imagery (Flickr) — tests generalization beyond curated street-view.
- Per-image lat/lon ground truth.

Implementation:

1. Obtain via the GeoCLIP repo's data scripts or directly from the Im2GPS authors.
2. Build a parallel manifest under `data/raw/im2gps3k/`.
3. Strip EXIF (Im2GPS3k images may retain it).
4. Use as a parallel evaluation track for E1, E2, E4. Report results separately and combined.

### 6.2 Secondary dataset options

Use secondary datasets only after the co-primary results are in.

Potential options:

1. **Mapillary Street-Level Sequences**
   - Public street-level sequences.
   - More diverse but larger and heavier to process.
   - Check license and access rules.

2. **GPTGeoChat**
   - Useful for privacy framing and prompt/moderation context.
   - It is conversation-based, not necessarily ideal as the main causal-intervention dataset.

### 6.3 Optional inferential privacy pilot dataset

Use the ETH-SRI private attribute inference dataset/code if accessible.

Alternative: construct a small consented/public dataset with non-personal scene-level attributes:

- institution type: school / hospital / government / commercial / residential
- setting type: home / workplace / transit / clinic / protest / religious site
- neighborhood proxy: high-income visual proxy / low-income visual proxy, but handle with care
- travel context: airport / hotel / tourist landmark / local street

Do not infer or publish real individuals' sensitive attributes. Use public scenes and aggregate labels.

---

## 7. Data schemas

Use Parquet for structured tables and JSONL for raw model outputs.

### 7.1 Image manifest schema

File:

```text
data/processed/manifests/{dataset_name}_{split}.parquet
```

Columns:

```text
image_id: str                         # stable UUID/hash
dataset: str                          # gsv_cities / im2gps3k / etc
split: str                            # train/dev/test or mvp/full
image_path: str                       # path to stripped image
original_image_path: str              # path to raw image
width: int
height: int
sha256: str
source_url: str | null
license: str | null

lat: float | null
lon: float | null
country: str | null
country_iso: str | null
region: str | null
city: str | null
place_id: str | null

contains_people_flag: bool | null
contains_faces_flag: bool | null
contains_license_plate_flag: bool | null
contains_ocr_text_flag: bool | null

is_allowed_for_public_demo: bool
notes: str | null
```

How `contains_*_flag` fields are populated:

- `contains_people_flag` / `contains_faces_flag`: populated from object detector (YOLO/COCO classes `person`, `face` if available) at manifest-build time. `null` if detector not run.
- `contains_license_plate_flag`: populated from a license-plate detector (e.g., a fine-tuned YOLO) or from OCR boxes with plate-shaped aspect ratio + alphanumeric content. `null` if not run.
- `contains_ocr_text_flag`: populated from OCR pass — true if any box has confidence ≥ 0.5 and ≥3 alphanumeric characters.
- `is_allowed_for_public_demo`: manually curated for demo subset; for non-demo images, default to `false`.

These flags drive E3 stratification (text-heavy vs text-light) and §16.4 sanity checks. Run the detectors once at manifest-build time and cache.

### 7.2 Region schema

File:

```text
data/processed/regions/{dataset}_{segmentation_engine}.parquet
```

Columns:

```text
image_id: str
region_id: str
source: str                    # sam / ocr / object / grid / merged
label: str | null              # semantic label
bbox_x1: int
bbox_y1: int
bbox_x2: int
bbox_y2: int
area_px: int
area_frac: float
mask_path: str                 # npy/png binary mask
stability_score: float | null
predicted_iou: float | null
ocr_text: str | null
object_confidence: float | null
parent_region_id: str | null
dedup_group: str | null
```

### 7.3 Prediction schema

Raw JSONL:

```text
data/processed/predictions/{model_name}/{run_id}.jsonl
```

Each line:

```json
{
  "run_id": "baseline_geoclip_2026_04_26",
  "image_id": "abc123",
  "image_variant_id": "original",
  "model_name": "geoclip",
  "model_type": "local",
  "prompt_id": "geo_json_v1",
  "temperature": 0,
  "raw_response": "...",
  "parsed": {
    "lat": 37.7749,
    "lon": -122.4194,
    "country": "United States",
    "region": "California",
    "city": "San Francisco",
    "confidence": 0.64,
    "evidence": ["street signs", "architecture"]
  },
  "parse_success": true,
  "created_at": "ISO_TIMESTAMP",
  "error": null
}
```

### 7.4 Intervention schema

```text
data/processed/interventions/{intervention_run_id}.parquet
```

Columns:

```text
image_id: str
region_id: str
intervention_type: str          # blur / mean_mask / inpaint / replace
image_variant_id: str           # e.g. img123_region456_inpaint
variant_path: str
mask_path: str
mask_area_frac: float
clip_similarity_to_original: float | null
caption_similarity_to_original: float | null
aesthetic_delta: float | null
artifact_flag: bool | null
```

### 7.5 Leakage score schema

Stores both single-region attribution scores (the heuristic) and joint-causal scores (from dynamic-greedy / Shapley). See §11.3 for the terminology distinction.

```text
data/processed/metrics/leakage_scores_{run_id}.parquet
```

(The legacy filename `causal_scores_{run_id}.parquet` is also accepted by tooling for backward compatibility, but new runs should use `leakage_scores_*`.)

Columns:

```text
image_id: str
region_id: str
model_name: str
threshold_km: float
original_error_km: float
edited_error_km: float
original_success: bool
edited_success: bool
binary_attribution: float        # single-region: original_success - edited_success (was binary_causal_leakage)
continuous_attribution: float    # single-region: log1p(edited_err) - log1p(original_err) (was continuous_causal_leakage)
binary_joint_causal: float | null      # joint effect from dynamic-greedy / Shapley method (§12.2, §12.11)
continuous_joint_causal: float | null  # joint effect from dynamic-greedy / Shapley method
method: str                      # static_greedy | dynamic_greedy | shapley | random | etc — how this score was computed
intervention_type: str           # mean_mask | blur | inpaint | min_across — see §10.7
area_frac: float
utility_cost: float
score_raw: float
score_per_area: float             # PRIMARY ranking (§11.4)
score_per_utility: float          # used only for E4 Pareto ranking
rank_within_image: int
```

---

## 8. Model interfaces

All models should implement a common interface.

### 8.1 Base geolocation model interface

```python
class GeolocationModel:
    name: str

    def predict(self, image_path: str, prompt: str | None = None) -> dict:
        '''
        Returns:
        {
          "lat": float | None,
          "lon": float | None,
          "country": str | None,
          "region": str | None,
          "city": str | None,
          "confidence": float | None,
          "evidence": list[str],
          "raw_response": str,
          "parse_success": bool,
          "error": str | None,
        }
        '''
```

### 8.2 Geolocation prompt for VLMs

Use this only on benchmark images.

Prompt ID: `geo_json_v1`

```text
You are evaluating a public benchmark image for a geolocation privacy research study.

Task: infer the likely location where the image was taken using only visual evidence in the image. Do not use file names, metadata, or external reverse image search.

Return valid JSON only, with this schema:
{
  "country": string or null,
  "region": string or null,
  "city": string or null,
  "latitude": number or null,
  "longitude": number or null,
  "confidence": number between 0 and 1,
  "visual_evidence": [string, ...],
  "uncertainty_notes": string
}

If the image does not contain enough visual evidence, return null for latitude and longitude and set confidence below 0.2.
```

Prompt ID: `geo_neutral_v1` — neutral control without research framing. Some VLMs (especially safety-tuned ones) refuse or hedge under the "privacy research study" framing in `geo_json_v1`. The control prompt isolates that effect.

```text
Where do you think this photo was taken? Return your best guess as JSON only:
{
  "country": string or null,
  "region": string or null,
  "city": string or null,
  "latitude": number or null,
  "longitude": number or null,
  "confidence": number between 0 and 1,
  "visual_evidence": [string, ...]
}

If you can't tell, return null for latitude and longitude and set confidence below 0.2.
```

Use `geo_neutral_v1` as the **default for E1/E2/E4 headline numbers** (it's what a real adversary would use). Use `geo_json_v1` and `geo_cautious_v1` / `geo_evidence_first_v1` for E7 prompt-sensitivity only.

Important:

- Never include image path, original dataset city, GPS, or filename in the prompt.
- Temperature should be 0.
- Query each image once for MVP; optionally query 3 times for prompt/noise robustness.

### 8.3 Prediction parsing

Implement strict JSON parsing with fallback regex extraction.

Rules:

- If no JSON parse, set `parse_success = false`.
- If lat/lon invalid or outside ranges, set null.
- If lat/lon null, geodesic error = infinity for threshold accuracy.
- Keep raw response for debugging, but do not expose raw exact-location outputs in public demo.

### 8.4 Local geolocation model: GeoCLIP

Use GeoCLIP as the default reproducible adversary.

Expected wrapper:

```python
GeoCLIPWrapper.predict(image_path) -> {
  "lat": float,
  "lon": float,
  "confidence": float | None,
  "raw_response": serialized_topk,
  "topk": [{"lat": ..., "lon": ..., "score": ...}, ...]
}
```

If GeoCLIP returns top-k candidates, store all candidates in raw response.

### 8.5 Open VLM wrappers

Optional open VLMs:

- Qwen2.5-VL or latest stable Qwen-VL model available locally.
- InternVL.
- LLaVA-style model.

Keep memory requirements in config. Make wrappers robust to missing GPUs.

### 8.6 Closed VLM wrappers

Optional API models:

- GPT-4o / GPT-4.1 vision.
- Gemini vision model.
- Claude vision model.

API wrapper requirements:

- Caching by `(model_name, image_sha256, prompt_id)`.
- Retry with exponential backoff.
- Rate-limit control.
- Redact dataset location metadata from prompt.
- Save raw response only in private local logs.

---

## 9. Segmentation and semantic region extraction

### 9.1 Goal

Produce human-interpretable candidate regions.

Candidate sources:

1. SAM/SAM2 masks.
2. OCR text boxes.
3. Object detector boxes.
4. Grid patches.
5. Optional panoptic segmentation.

### 9.2 SAM/SAM2 automatic masks

Run automatic mask generation.

Filtering:

```text
min_area_frac = 0.001
max_area_frac = 0.45
deduplicate masks with IoU > 0.85
remove tiny speckles
dilate masks by 5 px before intervention
```

Save masks as compressed `.npz` or binary PNG.

### 9.3 OCR regions

Use EasyOCR or PaddleOCR.

OCR regions are important because text/signage is an obvious geolocation cue and a baseline.

For each OCR box:

- Save bbox.
- Save text.
- Save OCR confidence.
- Convert quadrilateral to mask.
- Add label: `ocr_text`.

Do not display OCR text in public demo if it contains private info.

### 9.4 Object regions

Use GroundingDINO, YOLO-World, or a COCO detector.

Prompt/category list should include likely geolocation cues:

```text
street sign
storefront sign
license plate
road sign
traffic light
utility pole
power line
road marking
sidewalk
curb
building
storefront
mountain
tree
palm tree
vegetation
skyline
bridge
bus stop
train station
subway sign
school sign
hospital sign
religious building
flag
license plate
car
bus
taxi
```

### 9.5 Region labeling

Each region should get a **set of labels** (dominant + secondaries), not a single label. Single-label assignment understates Claim 2: a sign on a building façade gets labeled `sign`, and the building is reported as a separate region — readers see "70% of top regions are text" without knowing the text was on a façade the model also relied on.

**Schema** (extends §7.2 region schema):

```text
labels: list[str]              # all applicable labels, ranked
dominant_label: str            # top of labels list, used for primary taxonomy bucket
secondary_labels: list[str]    # labels[1:]
overlapping_region_ids: list[str]  # other regions whose mask IoU > 0.3 with this one
```

**Label assignment** (each region can satisfy multiple):

1. `ocr_text` — if region overlaps OCR box (IoU > 0.3).
2. Object detector label — for each detector class with IoU > 0.3, add the class name.
3. CLIP zero-shot label — top-1 zero-shot label.
4. Background fallbacks — `sam_unknown` or `grid_patch` if no other label applies.

**Dominant label priority** (which one is treated as "the" label for taxonomy bar charts):

1. `ocr_text` if present (text dominates the model's signal in most cases).
2. Object detector label with highest IoU.
3. CLIP zero-shot top-1.
4. `sam_unknown` / `grid_patch`.

For the cue-taxonomy figure (§13.E3, §15.3), report:

- Bars by **dominant label** (the headline view).
- A second panel: "% of top-k regions whose `secondary_labels` contains category X" — this surfaces overlap (e.g., text on building, sign on storefront).
- A third small-multiples panel: qualitative examples per dominant bucket so readers see what bucket assignment means visually.

Optional CLIP label prompts:

```text
a street sign
a storefront
a building facade
a road marking
a sidewalk
a curb
a utility pole
vegetation
mountains
a skyline
a vehicle
a license plate
a person
the sky
a window
a shop sign
a flag
a transit sign
```

### 9.6 Region merging

Avoid duplicate regions.

Algorithm:

1. Load all candidate masks.
2. Sort by priority: OCR > object > SAM > grid (only for deduplication tie-breaking; labels themselves are multi-label per §9.5).
3. Deduplicate high-overlap masks.
4. Keep overlapping masks if they have different semantic labels and IoU < 0.85.
5. Save all masks and labels.
6. For each image, cap number of regions to `max_regions_per_image = 80` for MVP.
7. Ensure grid fallback gives coverage even if segmentation fails.
8. After merging, write `data/processed/regions/{dataset}_region_distribution.json` with per-image statistics: `n_regions`, `n_regions_by_source` (sam/ocr/object/grid), `mean_area_frac`, `coverage_frac`. Report the distribution (median, p10, p90) in the paper so readers know the search space across which causal/leakage scores are computed.

---

## 10. Interventions

### 10.1 Required intervention types

Implement at least:

1. `blur`
2. `mean_mask`
3. `inpaint`

Optional:

4. `replace`
5. `crop`
6. `downsample_region`

### 10.2 Blur

Gaussian blur inside region mask.

Parameters:

```yaml
blur_sigma_frac: 0.035
mask_dilation_px: 7
```

Sigma should scale with image size:

```python
sigma = blur_sigma_frac * max(width, height)
```

### 10.3 Mean mask

Replace masked pixels with local mean color or global image mean.

Use for fast/cheap baseline.

### 10.4 Inpainting

Use LaMa/simple-lama. **Secondary intervention, not primary** for the headline sparsity result.

Why secondary: inpainting fills regions with plausible-looking content (a removed façade gets a generic façade back; a removed sign may come back as something or nothing). The model's response then depends on what the inpainter produced, not just on what was removed — failure modes are unpredictable. `mean_mask` and `blur` lose information predictably and are the safer choice for the headline.

Use inpainting for:

- Showing real-world redaction is feasible (qualitative gallery).
- E6 intervention ablation (does the causal ranking agree across blur / mean_mask / inpaint?).
- Min-across-interventions per region for §10.7 (most conservative headline).

Rules:

- Dilate mask before inpainting.
- Save artifact flag if inpainting fails.
- Strip EXIF after saving.
- Check that output dimensions match input.

### 10.5 Intervention artifact checks

For each edited image:

- Verify file exists.
- Verify dimensions unchanged.
- Compute CLIP similarity to original.
- Compute pixel edited fraction.
- Compute LPIPS if available, optional.
- Save a contact sheet for random samples.

### 10.6 Multiple-region edits

For redaction methods, apply intervention to a set of regions.

If masks overlap, union them before applying intervention.

### 10.7 Min-across-interventions for the headline

For each (image, region) pair, run **all** of `mean_mask`, `blur`, and `inpaint`. Compute the per-region leakage score under each, and take the **min** across intervention types as the conservative headline score:

```python
score_min = min(score_mean_mask, score_blur, score_inpaint)
```

This guards against attributing leakage drop to inpainting artifacts: a region only counts as high-leakage if **all three** interventions agree it is. This is the conservative version of §13.E6 (intervention ablation) used for headline numbers; E6 itself reports the agreement structure across intervention types.

For sparsity (E2) and Pareto (E4) headline results, use `score_min`. For per-intervention diagnostic plots, use the per-intervention scores.

---

## 11. Metrics

### 11.1 Geodesic distance

Implement Haversine distance.

```python
def haversine_km(lat1, lon1, lat2, lon2) -> float:
    ...
```

If prediction lat/lon is missing:

```text
error_km = inf
success@threshold = False
```

### 11.2 Geodesic error and threshold accuracy

**Primary metric: median geodesic error and median error increase under redaction.** Threshold accuracy at 25km is too coarse for the headline: it treats a 1km→24km shift as zero leakage and a 20km→26km shift as full leakage. The continuous score in §11.3 captures intervention effects faithfully and is what `score_per_*` rankings are built on. Acc@25km is kept as a familiar secondary view because the geolocation community uses it.

For threshold `tau_km`:

```python
success_tau = error_km <= tau_km
```

Reporting order (primary → secondary):

```text
Median geodesic error km                # PRIMARY for sparsity, Pareto
Median error increase km (paired)       # PRIMARY for redaction comparisons
Continuous causal score (log-error diff) # see §11.3
Acc@25km                                # secondary headline (familiarity)
Acc@1km, Acc@200km, Acc@750km, Acc@2500km   # secondary curve view
Mean clipped error km                   # tertiary (sensitive to single capped values; reportable but not headline)
Parse success rate
```

Clip infinite errors to Earth half-circumference:

```text
max_error_km = 20015
```

Median naturally handles inf via clipping. Mean is reportable but should not lead — a single capped value can dominate a small bootstrap sample.

For the "fraction of leakage explained by top-k%" headline number used in the paper: use **continuous error reduction**, not binary success drop.

### 11.3 Leakage attribution and causal scores — terminology

We distinguish two things that the literature sometimes conflates:

- **Single-region intervention effect** (a.k.a. "leakage attribution") — measured by removing one region in isolation. This is what §12.1 (static greedy) is built on. It's a useful heuristic, but it ignores interaction effects (suppression, redundancy) and **does not constitute a causal decomposition of joint leakage**. We deliberately avoid the word "causal" for this quantity.
- **Joint causal effect** — measured by the dynamic-greedy method (§12.2) or the Shapley-style spot-check (§12.11). These are what we call "causal" in the paper's headline.

Field naming in schema (§7.5) uses the prefix `attribution_*` for single-region-effect fields and `causal_*` for joint-effect fields:

```python
# Single-region intervention effect (leakage attribution)
orig_error = d(A(x), true_location)
edit_error = d(A(edit(x, r)), true_location)

orig_success = orig_error <= tau
edit_success = edit_error <= tau

binary_attribution    = int(orig_success) - int(edit_success)
continuous_attribution = log1p(edit_error_clipped) - log1p(orig_error_clipped)
```

For joint causal effect on a region set R = {r_1, ..., r_k}:

```python
# Joint causal effect under dynamic greedy / Shapley
edit_error_R = d(A(edit(x, R)), true_location)
binary_joint_causal    = int(orig_success) - int(edit_error_R <= tau)
continuous_joint_causal = log1p(edit_error_R_clipped) - log1p(orig_error_clipped)
```

Notes:

- Continuous attribution is more informative than binary; use it for ranking regions.
- Binary attribution is easy to explain and tracks Acc@25km drop.
- The headline "causal" claims in the paper rest on §12.2 dynamic greedy or §12.11 Shapley, **not** on summed single-region attributions.

Update §7.5 schema field names accordingly:

```text
binary_attribution: float           # was binary_causal_leakage
continuous_attribution: float       # was continuous_causal_leakage
binary_joint_causal: float | null   # populated only for dynamic/Shapley methods
continuous_joint_causal: float | null
```

### 11.4 Normalized scores

Compute:

```python
score_per_area    = continuous_attribution / max(area_frac, 0.005)
score_per_utility = continuous_attribution / max(utility_cost, 0.01)
```

Where:

```python
utility_cost = 1 - clip_similarity_to_original
```

**Primary ranking: `score_per_area`.**

```text
score_per_area
```

**`score_per_utility` is reserved for §13.E4 Pareto ranking only**, not for selecting top regions for E2 (sparsity) or E3 (cue taxonomy). Reason: CLIP similarity drops much more when you remove large/scenic content (façades, vegetation, sky) than when you remove a small text region. Worked example — at our defaults:

- Sign region: CLIP_sim ≈ 0.99 → utility_cost ≈ 0.01 (capped) → multiplier ×100.
- Building region: CLIP_sim ≈ 0.85 → utility_cost ≈ 0.15 → multiplier ×6.7.

So `score_per_utility` would systematically pre-rank text/sign regions at the top of the cue taxonomy, biasing the result against Claim 2 ("non-obvious environmental cues"). Using `score_per_area` as primary avoids encoding this bias into the selection criterion.

For the **cue-taxonomy figure (§13.E3)**, also report **unnormalized `continuous_attribution`** aggregated by cue category — i.e., where leakage actually lives, before any normalization. Normalization is a downstream selection concern; it should not shape the description.

For the **privacy-utility Pareto (§13.E4)**, `score_per_utility` is appropriate because the experiment is asking "given a utility budget, where does redaction help most?"

### 11.5 Utility metrics

Compute for each redacted image:

1. CLIP image-image similarity.
2. Caption similarity:
   - Generate caption for original and edited image.
   - Embed captions with SentenceTransformer.
   - Cosine similarity.
3. Percent pixels edited.
4. Aesthetic score, optional.
5. Human preference, optional.

### 11.6 Aggregate statistics

Use **paired bootstrap over image IDs** for all method comparisons (we evaluate methods on the same images, so paired sampling is correct).

For each metric:

- median (primary; per §11.2)
- standard error
- 95% bootstrap confidence interval
- paired difference vs GeoLeakLens
- p-value via paired bootstrap sign test

Bootstrap settings:

```yaml
n_bootstrap: 1000
seed: 123
unit: image_id
paired: true                      # all method comparisons are paired
stratify_by: country_iso          # use country if ≥30 examples per country, else don't stratify
stratify_min_per_stratum: 30
report:
  - bootstrap_ci_method: percentile  # default
  - also: bca                        # bias-corrected accelerated, if compute permits — more accurate for skewed distributions
```

Stratification rule (explicit):

- If a stratum (country_iso) has ≥30 images: include it as a stratum.
- If not: collapse small strata into "other" or drop stratification entirely for that comparison.
- Note in the table caption which stratification was used.

For non-overlapping-CI claims in §17 acceptance criteria: use **paired bootstrap of the difference**, not separate CIs of each method. Two methods can have overlapping marginal CIs but a paired difference whose CI excludes zero — that's still a valid significant difference.

---

## 12. Redaction methods

Implement these methods with a common interface.

```python
class RedactionMethod:
    name: str

    def select_regions(self, image_id: str, regions: pd.DataFrame, scores: pd.DataFrame, budget: dict) -> list[str]:
        ...
```

### 12.1 GeoLeakLens static greedy (fast approximation)

**Status: fast approximation, not the headline method.** Use this for quick scans during development and for region-level diagnostic plots. The headline method is dynamic greedy (§12.2). Validate static-vs-dynamic agreement via the Shapley spot-check (§12.11).

Why "fast approximation, not causal": this method scores each region independently (single-region intervention effect; see §11.3) and greedily unions the top-ranked. It ignores interaction effects:

- **Suppression**: two regions with low individual leakage might jointly carry a lot.
- **Saturation**: two regions might be redundant clues, so removing both helps no more than removing one.

That makes it a **feature attribution heuristic**, not a causal decomposition. It's still useful when scoring needs to be cheap (e.g., cue taxonomy across many images) — just don't claim it's causal.

Algorithm:

1. For each image, load region attribution scores (`continuous_attribution`).
2. Sort by `score_per_area` descending (per §11.4 — primary ranking).
3. Add regions until area budget reached.
4. Apply intervention to union mask (per §10.6).

Budgets:

```text
1%, 2%, 5%, 10%, 15%, 20% image area
```

### 12.2 GeoLeakLens dynamic greedy — PRIMARY METHOD

**Status: primary method.** This is the version that supports the paper's "causal" framing because it accounts for interaction effects (selecting one region changes the marginal value of every other region). Static greedy (§12.1) is a fast approximation; the Shapley spot-check (§12.11) validates that approximation.

Algorithm:

1. Start with no redaction. Cache `original_error = d(A(x), true_loc)`.
2. For each candidate region r, compute the single-region score `continuous_attribution(r)` under the primary intervention type (per §10.4: `mean_mask`).
3. Take the top-K candidates by `score_per_area`. K = 30 by default (caps cost; bigger K barely changes selection in pilot).
4. Among the top-K, pick the region r* that, when applied to the current edited image, maximizes joint causal effect:
   ```python
   r* = argmax_r [ d(A(edit(x_current, S ∪ {r})), true_loc) - d(A(x_current), true_loc) ]
   ```
5. Apply r*: `S ← S ∪ {r*}`, `x_current ← edit(x, S)`.
6. Recompute scores for the remaining candidates on `x_current`.
7. Repeat until area budget reached or privacy target met.

Cost control:

- **Caching**: every `(x, region_set, intervention_type, model)` query is cached by hash. Repeat runs are nearly free.
- **Top-K shortlist**: only re-score the top K=30 single-region candidates after each step, not all 80.
- **Stop early** when an additional region produces less than `delta_continuous = 0.05` (≈5% relative log-error movement) for two iterations in a row.
- For VLM adversaries, use `vlm_subset_n` (§5.2) and run dynamic greedy on that smaller subset; use static greedy on the full set.

Affordability: dynamic greedy at K=30, budget=20%, ~10 selection iterations per image, on 100 images × 1 VLM ≈ 30k VLM calls per (intervention × model) — well within the §4.1 cost budget.

`config: redaction.greedy_recompute = true` enables this method; set to `false` to fall back to static greedy.

### 12.3 Random regions baseline

Area-matched random selection. The protocol must match GeoLeakLens's edited area exactly per image — otherwise the comparison can systematically over- or under-shoot, biasing the headline.

Protocol:

1. For each image, compute the **target area** = the area GeoLeakLens uses at this budget on this image. This may be slightly less than the budget (e.g., 9.7% if no available region combination hits exactly 10%).
2. Permute the candidate regions uniformly at random (without replacement).
3. Walk the permutation, accumulating area. Stop **before** the next region would exceed `target_area + 0.5pp`.
4. If the accumulated area is below `target_area - 0.5pp`, fill the remaining budget with grid patches (smallest first) until within ±0.5pp of target.
5. Apply the primary intervention (mean_mask, per §10.4) to the union of selected regions.
6. Repeat with **10 seeds** per image. Report mean and paired-bootstrap CI across seeds, then aggregate across images via the §11.6 paired bootstrap.

Acceptance check: per-image edited-area-difference between random and GeoLeakLens should have median |Δarea| < 0.5pp. If it doesn't, the protocol is broken — fix before reporting.

Logging: per-image record `seed`, `selected_region_ids`, `actual_edited_area_frac`, and `target_area_frac` so the area-match can be audited.

### 12.4 Largest regions baseline

Select largest regions until area budget.

Controls for simply removing lots of content.

### 12.5 OCR-only baseline

Select OCR regions first.

If OCR area is below budget, either:

- stop at OCR-only area, or
- fill remaining budget with random regions.

Report both if possible:

```text
ocr_only
ocr_plus_random_fill
```

### 12.6 Face/license blur baseline

Use face detector and license plate detector if available.

This tests standard privacy anonymization.

If no detector is available, implement person/vehicle/license plate proxies and document limitations.

### 12.7 Full blur/downsample baseline

Apply full-image degradation tuned to match utility or privacy budget.

Variants:

- full Gaussian blur
- JPEG compression
- downsample then upsample

This is a privacy-utility lower bound.

### 12.8 Grid occlusion baseline

Divide image into grid patches and score patches instead of semantic regions.

This tests whether semantic regions matter.

### 12.9 Saliency baseline

Use gradient/saliency if using a differentiable local model such as GeoCLIP.

Options:

- Grad-CAM style saliency.
- CLIP relevance.
- Occlusion over grid patches.

### 12.10 GeoShield baselines (cross-paradigm — handle with care)

GeoShield adds **imperceptible adversarial perturbations**. GeoLeakLens **removes visible content**. They solve different problems and any utility metric correlated with perceptual similarity (CLIP-sim, LPIPS, caption similarity) will, by construction, put GeoShield near the top — its perturbations are designed to be invisible. Putting them on the same Pareto axis without caveats rigs the utility comparison against us.

We therefore treat GeoShield as **two** baselines, on a **separate panel** of the Pareto plot (or with a perceptibility annotation):

**12.10.a `geoshield_imperceptible`** (the published method):

1. Clone GeoShield repo into `data/external/geoshield/`.
2. Follow its README.
3. Run on the same image subset using the published ε.
4. Evaluate protected images using the same model wrappers and metrics.
5. Compute utility metrics using the same code.
6. **Also compute a perceptibility score**: human "is this image edited?" detection rate on a 50-image subset, plus LPIPS to original.

**12.10.b `geoshield_visible_eps`** (fair head-to-head):

1. Run GeoShield with a much larger ε so the perturbation budget is *visible* (e.g., L∞ ε = 16/255 or whichever budget brings LPIPS into the same range as our redaction).
2. This is the fair head-to-head against GeoLeakLens: both methods are now visibly modifying the image.
3. Report side-by-side privacy reduction and CLIP/caption similarity.

**Paper framing**: "GeoShield trades imperceptibility for protection robustness; GeoLeakLens trades visible content removal for protection robustness against post-processing detection. The methods address different threat surfaces. We compare them on dimensions that are common (privacy reduction, transferability across models, robustness to JPEG/resize) and report perceptibility as a third axis."

If GeoShield cannot be reproduced:

- Include a "not reproduced" note in logs.
- Still keep the code hook.
- Do not claim superiority over GeoShield without reproduced numbers.
- Compare conceptually in paper and mark empirical comparison as future/optional.

### 12.11 Shapley-style validation (joint causal spot-check)

To validate that static greedy (§12.1) and dynamic greedy (§12.2) ranks roughly agree — i.e., that single-region attribution is a defensible fast approximation to joint causal effect — run a Shapley-style spot-check on a small subset.

Subset:

```text
50 images × top-10 candidate regions per image
```

For each (image, region-set-of-size-k), compute joint causal effect under `mean_mask` intervention. Approximate Shapley values for each region by sampling 200 random orderings of the top-10 regions and averaging marginal contributions. Cost: 50 × 10 × 200 = 100k cached queries per (intervention, model) — within budget.

Compare:

- **Top-1 agreement**: how often does static greedy's top-1 region equal Shapley's top-1?
- **Top-5 Jaccard**: between static-greedy's top-5 and Shapley's top-5.
- **Spearman rank correlation** of region scores.

Acceptance:

- If top-1 agreement ≥ 0.7 and Spearman ≥ 0.7: static greedy is a defensible fast approximation; report this in the paper as validation of the broader analysis.
- If they disagree substantially: only Shapley / dynamic greedy results support Claim 1; flag this and run dynamic greedy on the larger subset for E2.

Output:

```text
tables/shapley_validation.csv
figures/shapley_vs_static_rank_agreement.pdf
```

---

## 13. Experiments to run

### E0 — Smoke test

Goal: ensure pipeline works end-to-end.

Dataset:

```text
20 images from 4 cities
```

Models:

```text
GeoCLIP only
```

Run:

1. Build manifest.
2. Strip EXIF.
3. Segment images.
4. Run original geolocation.
5. Generate interventions for max 10 regions/image.
6. Score causal leakage.
7. Produce one redacted image per image at 10% budget.
8. Generate qualitative contact sheet.

Acceptance:

- No script crashes.
- At least 90% of images have masks.
- Predictions parse.
- Metrics file exists.
- Qualitative masks look reasonable.

---

### E1 — Threat establishment

Question:

> Can models infer location before redaction?

Dataset:

```text
MVP: 500 images per dataset (GSV-Cities + Im2GPS3k = 1,000 total), stratified.
Full: 5,000 images per dataset if compute permits.
```

Models:

```text
MVP: GeoCLIP (full N) + 1 VLM (e.g., GPT-4o or Qwen-VL) on a vlm_subset_n=100 subset.
Full: GeoCLIP + 2-3 VLMs on full N (cost permitting; see §4.1).
```

The VLM-in-MVP step is non-negotiable: the paper's threat model is VLMs, not retrieval models. GeoCLIP scores are useful as a cheap surrogate for region selection, but at least one VLM in the headline is needed to know whether the threat is real for the threat model the paper claims.

Outputs:

- `tables/baseline_geolocation.csv`
- `figures/baseline_geolocation_bar.pdf`

Metrics (in §11.2 priority order):

```text
Median geodesic error km    # PRIMARY
Acc@25km                     # secondary headline
Acc@1km, Acc@200km, Acc@750km, Acc@2500km   # secondary curve
Parse success rate
```

Report per (model × dataset × prompt_id) cell. Use `geo_neutral_v1` (per §8.2) as the default prompt for the headline number.

Important:

- Region-level analysis (E2, E3) should focus on images where original prediction succeeds at the main threshold, because if the model already fails, removing regions cannot reduce success. Be explicit in figure captions that these analyses are **conditional on baseline success**.
- Final redaction evaluation (E4) should include **all images**, not only originally successful ones.
- Add a complementary table: on originally-failed images, does redaction increase the error further, decrease it, or leave it unchanged? This is informational, not a primary claim, but it tells the reader whether redaction is also harmless on the harder distribution.

---

### E2 — Leakage sparsity

Question:

> Does redacting top-ranked semantic regions reduce geolocation accuracy faster than random or obvious-identifier baselines?

Dataset:

**Primary (conditional)**: only images where original prediction succeeds at Acc@25km for the source model. Be explicit in the figure caption and paper text that the sparsity claim is **conditional on baseline success** — this is the right frame because regions can't reduce success if there was no success to begin with, and readers should know.

**Companion (unconditional)**: same plots on all images. This shows whether the result generalizes beyond the conditional set; expect smaller effect sizes here, but the shape of the curve should still favor the targeted methods.

Intervention:

- **Primary**: `mean_mask` (per §10.4 — predictable failure mode).
- **Headline**: `min_across_interventions` (per §10.7 — most conservative).
- Per-intervention diagnostic: also report `inpaint` and `blur` curves separately.

Methods (in priority):

- **GeoLeakLens dynamic greedy** (§12.2) — primary headline method.
- GeoLeakLens static greedy (§12.1) — fast approximation, included as a comparison row.
- Random regions (§12.3, area-matched).
- Largest regions (§12.4).
- OCR-only (§12.5).
- Face/license blur (§12.6).
- Grid random (§12.8).
- Saliency baseline (§12.9) — Grad-CAM on GeoCLIP.

Budgets:

```text
1%, 2%, 5%, 10%, 15%, 20%
```

Output figures (in priority order):

```text
figures/sparsity_median_error_vs_area.pdf      # PRIMARY: median geodesic error increase vs edited area
figures/sparsity_continuous_score_vs_area.pdf  # PRIMARY companion: continuous attribution sum
figures/sparsity_acc25_vs_area.pdf             # secondary: Acc@25km drop vs edited area
figures/sparsity_unconditional_median_error.pdf  # unconditional companion
```

Plot conventions:

- x-axis: edited area fraction (must be exactly area-matched across methods per §12.3).
- y-axis: median error increase (primary) or Acc@25km drop (secondary).
- Lines: one per method.
- Shaded bands: paired-bootstrap 95% CI per §11.6.
- Annotation: the area at which the dynamic-greedy curve crosses 50% of the maximum effect — this is the "fraction of leakage explained by top-X% area" headline number.

Expected paper result (conditional set):

> The top semantic regions selected by dynamic greedy should degrade median geodesic error much faster than random or obvious-identifier baselines, with a sparse fraction of image area accounting for a large fraction of leakage.

Expected paper result (unconditional companion):

> The same ranking still degrades model performance on the full image set, with smaller magnitudes — sparsity is a real property of the conditional distribution, not an artifact of conditioning.

---

### E3 — Cue taxonomy

Question:

> Which semantic cue types carry geolocation leakage?

Dataset:

```text
Images where original prediction succeeds and at least one region has positive single-region attribution.
```

Compute (per region):

- `dominant_label` and `secondary_labels` (multi-label per §9.5)
- `continuous_attribution` (single-region; primary aggregation field)
- `score_per_area` (used to define "top-k regions")
- `rank_within_image`
- `area_fraction`
- `is_ocr_overlapping`, `is_face_or_plate_overlapping`, `is_environmental` (derived from labels)

Aggregate by **dominant** semantic bucket (primary view) and by **secondary-label overlap** (companion view).

**Primary metric: unnormalized `continuous_attribution`** aggregated per bucket. Don't normalize by area or utility cost when describing where leakage lives — normalization is a downstream selection concern (per §11.4).

Outputs:

```text
tables/cue_taxonomy.csv                    # primary
tables/cue_taxonomy_textlight.csv          # text-light subset (see below)
figures/cue_taxonomy_bar.pdf               # primary, dominant labels
figures/cue_taxonomy_overlap.pdf           # secondary-label overlap panel
figures/cue_taxonomy_top_examples.pdf      # qualitative gallery per bucket
figures/cue_taxonomy_textlight_bar.pdf     # text-light subset comparison
```

Metrics per bucket:

```text
mean continuous_attribution                # PRIMARY: unnormalized leakage by bucket
median continuous_attribution
% of top-1 regions whose dominant_label is this bucket
% of top-5 regions whose dominant_label is this bucket
% of top-5 regions whose secondary_labels include this bucket    # overlap view
mean area_frac (informational, not a normalization)
n_regions in bucket
```

Semantic buckets:

```text
explicit_text_signage
face_person
license_plate_vehicle
building_architecture
storefront_commercial
road_marking
sidewalk_curb
vegetation
terrain_mountain_water
sky_skyline
utility_pole_wires
transit_infrastructure
other
unknown
```

**Text-heavy vs text-light split** (planned figure, not a fallback):

Stratify the analysis by `contains_ocr_text_flag` (per §7.1) and report cue taxonomy separately on:

- All images (headline).
- Text-light subset (`contains_ocr_text_flag == False` or text covers <2% of pixels). This subset specifically tests Claim 2 — if environmental cues dominate even when text is absent, that's the strongest evidence for non-obvious leakage channels.

Expected paper result:

> Conditional on baseline success, a non-trivial fraction (≥30% target) of top-5 regions are environmental (not explicit text/face/plate). On the text-light subset, environmental cues dominate the top-5 by construction — the leakage shifts to architecture, road markings, vegetation, and skyline.

If the headline result is dominated by `explicit_text_signage`, that is **still a result** — it means our paper's "non-obvious cues" claim must be stated more carefully. The text-light subset then becomes the primary demonstration of Claim 2.

---

### E4 — Privacy-utility Pareto

Question:

> Does GeoLeakLens give a better privacy-utility frontier than visible-redaction baselines, and how does it compare to perturbation-based defenses on shared dimensions?

Dataset:

```text
All evaluation images, not only originally successful ones.
```

Methods, on **two separate panels** (do not put them on the same Pareto axis without caveats):

**Panel A — Visible redaction methods (head-to-head):**

- GeoLeakLens dynamic greedy (PRIMARY).
- GeoLeakLens static greedy.
- Random regions (area-matched per §12.3).
- OCR-only.
- Face/license blur.
- Full blur/downsample (lower-bound reference).
- Largest regions.
- Saliency/grid baseline.
- **`geoshield_visible_eps`** (GeoShield run with ε large enough to be visible — fair head-to-head; see §12.10.b).

**Panel B — Cross-paradigm reference (separate panel):**

- `geoshield_imperceptible` (GeoShield as published; included for context).
- All Panel A methods replotted for reference.
- Annotated with **perceptibility axis**: human "is this image edited?" detection rate (50-image subset) and LPIPS to original.

For each method and budget:

Compute privacy:

```text
Median error increase km    # PRIMARY (per §11.2)
Acc@25km drop
Acc@1km drop
Continuous attribution sum (paired, total log-error movement)
```

Compute utility:

```text
CLIP similarity
caption similarity
percent pixels edited
aesthetic score (optional)
LPIPS (only used for Panel B perceptibility comparison)
```

For the privacy-utility ranking on Panel A only, use `score_per_utility` from §11.4 (this is the one experiment where utility-aware ranking is justified).

Output:

```text
figures/privacy_utility_panelA_clip_median_err.pdf       # PRIMARY: visible-redaction head-to-head
figures/privacy_utility_panelA_caption_median_err.pdf
figures/privacy_utility_panelB_perceptibility.pdf        # cross-paradigm with GeoShield
tables/redaction_method_comparison.csv
tables/perceptibility_panel_b.csv
```

Plot conventions:

- x-axis: utility preserved (CLIP similarity).
- y-axis: privacy protected (median error increase, primary; Acc@25km reduction as secondary panel).
- Each method is a curve over budgets.
- GeoLeakLens should be on or near the **Panel A** Pareto frontier.

Acceptance:

- Baselines must be **area-matched** per §12.3 protocol; verify median |Δarea| < 0.5pp before reporting.
- Do not compare GeoLeakLens at 20% area to OCR at 2% area without normalization.
- Do **not** put `geoshield_imperceptible` on Panel A — that comparison is rigged on utility (§12.10).
- In the paper, frame Panel B as cross-paradigm; do not claim GeoLeakLens "beats GeoShield" without referencing perceptibility.

---

### E5 — Model transfer

Question:

> Are causal regions model-specific, or do they remove generally location-informative evidence?

Procedure:

1. Use source model A to compute causal scores.
2. Redact top regions according to model A.
3. Evaluate edited image on target model B.

Matrix:

```text
source: GeoCLIP, GPT-4o if available
target: GeoCLIP, GPT-4o, Gemini, Qwen-VL if available
```

Output:

```text
tables/model_transfer_matrix.csv
figures/model_transfer_heatmap.pdf
```

Metric:

```text
Acc@25km reduction on target model
```

Expected result:

> Semantic redaction should transfer better than perturbation-based or model-specific saliency because it removes underlying evidence.

---

### E6 — Intervention-type ablation

Question:

> Are causal scores stable across blur, mask, and inpainting?

For a subset:

```text
500 images or 100 images for closed VLMs
```

Compute top-k causal regions under:

- blur
- mean mask
- inpaint

Metrics:

```text
top1 agreement
top5 Jaccard
Spearman correlation of region ranks
redaction performance after applying each intervention
```

Output:

```text
tables/intervention_ablation.csv
figures/intervention_rank_agreement.pdf
```

Expected:

- Some variation is expected.
- If inpainting and blur agree on high-level cues, causal result is stronger.

---

### E7 — Prompt sensitivity

Question:

> Are VLM geolocation and causal scores sensitive to prompt phrasing?

Use 3 prompts:

1. `geo_json_v1`: neutral benchmark prompt.
2. `geo_cautious_v1`: encourages uncertainty.
3. `geo_evidence_first_v1`: asks for visual evidence before prediction.

Do not optimize prompts to maximize doxing. Keep this for robustness.

Metrics:

```text
prediction parse rate
Acc@25km
top-region agreement
redaction performance
```

Output:

```text
tables/prompt_sensitivity.csv
```

---

### E8 — Broader inferential privacy pilot (descoped — concrete protocol or cut)

Question:

> Does the causal redaction framework extend beyond geolocation?

This is the riskiest experiment in the suite (vague labels, ethically loaded categories, harder-to-measure ground truth). Two acceptable paths — pick one before starting:

**Path A — Cut from this paper.** Move to a follow-up paper where it gets the space and care it needs. This is the recommended default.

**Path B — Tightly scoped pilot, with the following non-negotiable constraints:**

Drop the `socioeconomic_visual_proxy` label entirely. Encoding "high-resource / low-resource" visual proxies is ethically loaded, hard to validate, and likely to backfire under reviewer scrutiny. Stick to scene-type labels with objective ground truth.

Use only:

```text
institution_type: school / hospital / religious_site / government / commercial / residential / transit / other
```

Dataset (Path B):

- 300 images from a public, license-cleared source (e.g., Wikimedia Commons category-tagged photos, OpenImages with `Place` label, or curated subset of GSV-Cities matched to OSM POI categories).
- Each image labeled by ≥2 annotators with the institution_type taxonomy above. Inter-annotator agreement ≥ Cohen's κ = 0.6 required. Discard images where annotators disagree.
- Document the labeling protocol in `data/raw/private_attr_pilot/labeling_protocol.md`.
- Do not infer or publish any individual's attributes. Aggregate only.

Procedure (Path B):

1. Query VLM for institution_type prediction (closed-set classification, not free text).
2. Score original accuracy.
3. Segment and intervene on regions per the geolocation pipeline.
4. Compute single-region attribution for attribute prediction confidence.
5. Redact top regions per dynamic-greedy on attribute confidence.
6. Measure attribute prediction accuracy drop and utility preservation.

Metrics:

```text
attribute accuracy (closed-set, top-1)
confidence drop on the original-correct class
prediction entropy increase
top cue categories (multi-label per §9.5)
CLIP utility preservation
```

Output:

```text
tables/private_attr_pilot.csv
figures/private_attr_examples.pdf
```

Important language:

- Call this a preliminary extension to a single attribute (institution_type).
- Do not claim the method solves inferential privacy broadly.
- Acknowledge in limitations that this generalization is one data point and additional attributes / datasets are needed for stronger claims.

---

### E9 — Small human evaluation

Question:

> Do humans agree with the automated privacy-utility metrics? Specifically: (a) can humans still identify location from redacted images, and (b) do humans rate the utility-preserving claim as plausible?

Privacy-utility in E4 is decided entirely by automated metrics (CLIP-sim, caption-sim, model error). These correlate with perception but imperfectly. A small human study is cheap and high-impact for a privacy paper.

Dataset:

```text
50 images, sampled stratified across cities/regions and across original-success / original-fail.
3 redaction methods × 50 images = 150 method-images.
3 raters per method-image.
Total: 450 ratings.
```

Methods compared (use the 10% area budget, dynamic-greedy version):

- GeoLeakLens (mean_mask)
- OCR-only (area-matched)
- Face/license plate blur

Tasks per rater per image (no prior knowledge of which method was applied):

1. **Privacy task**: "What city, country, or region is this photo taken in? You may answer at any granularity (country / region / city). If you cannot tell, say so." Rater confidence on 1–5 scale.
2. **Naturalness task**: "Does this image look natural, or has it been edited? Rate 1 (clearly edited) to 5 (clearly natural)."
3. **Utility task**: Free-text describe the scene in one sentence. (Used for caption-similarity validation.)

Recruitment:

- MTurk or lab volunteers; 3 raters per condition.
- Pay above local minimum; document compensation in the paper.
- Filter out raters with <80% accuracy on a calibration set (3 unedited control images with known location).

Metrics:

```text
% raters who correctly identify country
% raters who correctly identify region (if applicable)
% raters who correctly identify city
Mean rater confidence (privacy task)
Mean naturalness score
Sentence-embedding similarity of rater descriptions across conditions (caption-sim validation)
Inter-rater agreement (Krippendorff's alpha or similar)
```

Output:

```text
tables/human_eval.csv
figures/human_eval_privacy_naturalness.pdf
```

Acceptance:

- Inter-rater agreement κ ≥ 0.4 on city-level identification (humans should at least loosely agree on which images are unidentifiable).
- Report results regardless of direction. If humans can still identify city from GeoLeakLens-redacted images at higher rates than expected, that is a result and a limitation.

This experiment is small (~$200–500 on MTurk) and provides a human-grounded validation row in the paper's main table — disproportionately valuable for reviewer credibility.

---

## 14. Scripts and expected commands

### 14.1 Build manifest

```bash
python -m geoleaklens.data.build_manifest \
  --dataset gsv_cities \
  --raw-root data/raw/gsv_cities \
  --out data/processed/manifests/gsv_full.parquet
```

### 14.2 Sample MVP subset

```bash
python -m geoleaklens.data.sample_dataset \
  --manifest data/processed/manifests/gsv_full.parquet \
  --n 500 \
  --stratify city \
  --seed 42 \
  --out data/processed/manifests/gsv_mvp.parquet
```

### 14.3 Strip EXIF and resize

```bash
python -m geoleaklens.data.strip_exif \
  --manifest data/processed/manifests/gsv_mvp.parquet \
  --out-root data/interim/stripped_exif \
  --max-side 1024 \
  --out-manifest data/processed/manifests/gsv_mvp_stripped.parquet
```

### 14.4 Run segmentation

```bash
python -m geoleaklens.segmentation.sam_regions \
  --config configs/default.yaml \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --out data/processed/regions/gsv_mvp_regions.parquet
```

### 14.5 Run baseline geolocation

```bash
python -m geoleaklens.experiments.run_baseline_geolocation \
  --config configs/default.yaml \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --model geoclip \
  --out data/processed/predictions/geoclip/baseline_gsv_mvp.jsonl
```

### 14.6 Generate region interventions

```bash
python -m geoleaklens.experiments.run_region_interventions \
  --config configs/default.yaml \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --regions data/processed/regions/gsv_mvp_regions.parquet \
  --intervention inpaint \
  --out data/processed/interventions/gsv_mvp_inpaint.parquet
```

### 14.7 Geolocate edited variants

```bash
python -m geoleaklens.experiments.run_baseline_geolocation \
  --config configs/default.yaml \
  --variant-manifest data/processed/interventions/gsv_mvp_inpaint.parquet \
  --model geoclip \
  --out data/processed/predictions/geoclip/interventions_gsv_mvp_inpaint.jsonl
```

### 14.8 Score causal regions

```bash
python -m geoleaklens.scoring.causal_scores \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --regions data/processed/regions/gsv_mvp_regions.parquet \
  --baseline-preds data/processed/predictions/geoclip/baseline_gsv_mvp.jsonl \
  --edited-preds data/processed/predictions/geoclip/interventions_gsv_mvp_inpaint.jsonl \
  --out data/processed/metrics/causal_scores_gsv_mvp_geoclip.parquet
```

### 14.9 Run redaction methods

```bash
python -m geoleaklens.experiments.run_redaction_methods \
  --config configs/default.yaml \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --regions data/processed/regions/gsv_mvp_regions.parquet \
  --scores data/processed/metrics/causal_scores_gsv_mvp_geoclip.parquet \
  --methods geoleaklens,random_regions,largest_regions,ocr_only,full_blur \
  --out data/processed/redactions/gsv_mvp_redactions.parquet
```

### 14.10 Evaluate methods

```bash
python -m geoleaklens.experiments.run_baseline_geolocation \
  --config configs/default.yaml \
  --variant-manifest data/processed/redactions/gsv_mvp_redactions.parquet \
  --model geoclip \
  --out data/processed/predictions/geoclip/redactions_gsv_mvp.jsonl

python -m geoleaklens.experiments.run_redaction_methods \
  --evaluate-only \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --redactions data/processed/redactions/gsv_mvp_redactions.parquet \
  --predictions data/processed/predictions/geoclip/redactions_gsv_mvp.jsonl \
  --out data/processed/metrics/redaction_eval_gsv_mvp.parquet
```

### 14.11 Make plots

```bash
python -m geoleaklens.plots.plot_sparsity_curve \
  --metrics data/processed/metrics/redaction_eval_gsv_mvp.parquet \
  --out outputs/figures/sparsity_curve.pdf

python -m geoleaklens.plots.plot_cue_taxonomy \
  --scores data/processed/metrics/causal_scores_gsv_mvp_geoclip.parquet \
  --regions data/processed/regions/gsv_mvp_regions.parquet \
  --out outputs/figures/cue_taxonomy.pdf

python -m geoleaklens.plots.plot_privacy_utility \
  --metrics data/processed/metrics/redaction_eval_gsv_mvp.parquet \
  --out outputs/figures/privacy_utility.pdf

python -m geoleaklens.plots.make_qualitative_figures \
  --manifest data/processed/manifests/gsv_mvp_stripped.parquet \
  --regions data/processed/regions/gsv_mvp_regions.parquet \
  --scores data/processed/metrics/causal_scores_gsv_mvp_geoclip.parquet \
  --redactions data/processed/redactions/gsv_mvp_redactions.parquet \
  --out outputs/demo_examples/
```

---

## 15. Plot specifications

### 15.1 Figure 1: whole paper figure

Create a 4-panel figure:

1. Original image.
2. Model prediction and error.
3. Causal heatmap/region overlays.
4. Redacted image and new prediction.

Must include:

- top 3 causal region labels
- original Acc@25km status
- edited Acc@25km status
- edited area percentage
- CLIP similarity

### 15.2 Sparsity curve

```text
x-axis: edited image area %
y-axis: Acc@25km
lines: GeoLeakLens, random, largest, OCR-only, face/license, full blur
include 95% CI
```

A second version:

```text
y-axis: median geodesic error km
```

### 15.3 Cue taxonomy bar chart

```text
x-axis: semantic cue category
y-axis: mean causal leakage or % top-5 regions
error bars: bootstrap CI
```

Sort categories descending.

### 15.4 Privacy-utility Pareto

```text
x-axis: CLIP similarity to original
y-axis: Acc@25km reduction or median error increase
points/curves: methods across budgets
```

Higher is better for privacy; farther right is better for utility.

### 15.5 Transfer heatmap

Rows:

```text
source model used for causal scores
```

Columns:

```text
target model used for final evaluation
```

Cell:

```text
Acc@25km reduction
```

---

## 16. Sanity checks and failure modes

### 16.1 Data leakage checks

Before any model query:

- Confirm EXIF stripped.
- Confirm prompt does not include path, city, country, coordinates, dataset ID, or filename.
- Confirm image filename sent to API is generic if API sees filenames.

### 16.2 Mask sanity checks

Generate contact sheets:

```text
outputs/debug/masks_random_100.pdf
```

Check:

- masks align with objects/regions
- OCR boxes are correct
- masks are not mostly empty
- large masks do not cover whole image unless intended

### 16.3 Prediction sanity checks

Create:

```text
outputs/debug/prediction_samples.jsonl
```

Check:

- parse success rate
- invalid coordinates
- null predictions
- obvious impossible predictions
- model refusing too often

### 16.4 Intervention sanity checks

Create contact sheets:

```text
original | mask | blur | mean_mask | inpaint
```

Check:

- inpainting does not create new signs/text.
- blur is strong enough.
- masks not shifted.
- EXIF not restored.

### 16.5 Leakage score sanity checks

For a few images:

- Top region removal should visibly remove plausible clue.
- Random region removal should usually matter less.
- If original prediction failed, single-region attribution should not be interpreted as joint causal leakage reduction (per §11.3 terminology).
- Static-greedy top-1 should match dynamic-greedy top-1 on most images. Where they disagree, inspect — it's usually an interaction effect (suppression or saturation) and these are the most informative qualitative examples.

### 16.6 Common failure modes

1. **Original model is too weak.**  
   If baseline geolocation accuracy is near zero, causal leakage cannot be measured. Use easier dataset, stronger model, or threshold Acc@750km.

2. **Original model is too strong globally.**  
   If removing small regions does not change output, use larger budgets or images with localized cues.

3. **Inpainting artifacts dominate.**  
   Compare blur/mask/inpaint agreement.

4. **Causal score picks huge regions.**  
   Use score per area or per utility cost.

5. **Model uses prior/dataset bias.**  
   Test on geographically diverse data and check predictions after heavy redaction.

6. **OCR dominates everything.**  
   This is still a result, but the paper needs non-obvious cue analysis. Separate text-heavy vs text-light images.

7. **Segmentation misses road/building cues.**  
   Add grid patches and panoptic/object detectors.

---

## 17. MVP acceptance criteria

The MVP is successful if we can produce:

1. A manifest of 500 public geotagged images **per dataset** (GSV-Cities + Im2GPS3k).
2. Baseline geolocation table for GeoCLIP (full N) and at least one VLM (vlm_subset_n=100).
3. Semantic region masks for at least 90% of images.
4. Single-region attribution scores for all (image, region) pairs.
5. Dynamic-greedy joint-causal results on the conditional set (originally-successful images).
6. Shapley spot-check (50 images) confirming static-vs-dynamic agreement, OR clear documentation that they disagree.
7. Sparsity curve showing GeoLeakLens vs random/obvious baselines, with paired-bootstrap CIs.
8. Cue taxonomy chart with multi-label aggregation, plus text-light subset version.
9. Privacy-utility Pareto chart (Panel A: visible-redaction head-to-head).
10. 10 qualitative examples (mix of successes and failures).
11. A written note identifying whether each claim appears viable.

Minimum evidence for viability (with statistical power thresholds):

- **Baseline strength**: at least one model achieves median geodesic error ≤ 200 km on at least one of the two primary datasets, AND Acc@25km ≥ 0.20 on the same set. If neither holds, the threat is not strong enough to support causal-region analysis on this distribution — pivot to easier subsets or stronger models.
- **Sparsity, statistically significant**: at the 5% area budget, GeoLeakLens dynamic-greedy beats area-matched random by ≥ 5pp Acc@25km drop (and ≥ 100km median error increase) with **paired-bootstrap 95% CI of the difference excluding zero** (per §11.6). Marginal CIs being non-overlapping is too strict; use the paired-difference CI.
- **Non-obvious cues**: at least one of:
  - ≥ 30% of top-5 regions on the all-images cue taxonomy have dominant_label outside `{explicit_text_signage, face_person, license_plate_vehicle}`, OR
  - On the text-light subset (per §13.E3), environmental cues account for ≥ 50% of top-5 regions.
- **Utility**: GeoLeakLens at 10% area budget retains CLIP similarity ≥ 0.85 (i.e., utility_clip_min from §5.1), beating full-image blur/downsample at the same Acc@25km drop level.
- **Intervention robustness**: min-across-interventions headline curves are within 25% of mean_mask-only curves at 10% budget. If the gap is large, results are intervention-dependent and the paper must say so.

---

## 18. Full paper acceptance criteria

The full experiment suite is successful if:

1. Results hold on at least 2 datasets or 2 distinct subsets.
2. Results hold across at least 2 geolocation adversaries.
3. GeoLeakLens beats area-matched random, OCR-only, face/license blur, and largest-region baselines.
4. If feasible, GeoLeakLens is compared against GeoShield or a documented attempt is made.
5. Causal cue categories are stable enough to support Claim 2.
6. Model-transfer experiment shows semantic redaction is not purely model-specific.
7. Intervention ablation shows results are not just editing artifacts.
8. Bootstrap confidence intervals are reported.
9. Qualitative examples include both successes and failures.

---

## 19. Notes for the coding agent

### 19.1 Prioritize implementation order

Do this order:

1. Manifest + image preprocessing (both GSV-Cities and Im2GPS3k).
2. GeoCLIP wrapper.
3. Geolocation metrics (median primary, Acc@thresholds secondary).
4. SAM region extraction + multi-label labeling (§9.5).
5. Mean-mask + blur interventions.
6. Single-region attribution scoring (static).
7. Static greedy GeoLeakLens redaction.
8. Random (with §12.3 area-match protocol)/largest/OCR baselines.
9. Sparsity and privacy-utility plots — first end-to-end pass on GeoCLIP only, all conditional set, before adding VLM.
10. **VLM API wrapper (one VLM, e.g., GPT-4o or Qwen-VL) — with caching from day 1.**
11. Re-run E1/E2 headline on VLM at vlm_subset_n=100. Validates threat model before investing more.
12. **Dynamic greedy (§12.2) — primary causal method.**
13. Shapley spot-check (§12.11) on 50 images.
14. Inpainting — for qualitative gallery and intervention ablation only.
15. Min-across-interventions (§10.7) headline run.
16. Transfer (E5) and intervention ablation (E6).
17. Human eval (E9).
18. GeoShield baselines (§12.10) — both visible and imperceptible.
19. Prompt sensitivity (E7).
20. Private-attribute pilot (E8) — only if Path B was chosen, otherwise cut.

Do not start with closed VLM APIs or fancy inpainting. Get the local GeoCLIP pipeline working first, then add one VLM with caching, then dynamic greedy, then everything else.

### 19.2 Use caching everywhere

Cache expensive steps by image hash and config hash:

- segmentation
- OCR
- model predictions
- edited variants
- utility metrics

### 19.3 Make every output auditable

Every table row should be traceable back to:

```text
image_id
region_id
model_name
prompt_id
intervention_type
config hash
```

### 19.4 Do not hide bad results

Save failure cases and invalid outputs. The paper needs credible limitations.

### 19.5 Make the code resumable

Every script should skip existing outputs unless `--overwrite` is passed.

### 19.6 Keep public demo safe

The demo should not output precise coordinates. It should output:

```text
Low / medium / high geolocation leakage
Top visual cue categories
Suggested redacted image
Before/after coarse prediction, e.g. country or broad region only
```

---

## 20. Suggested paper tables from experiment outputs

### Table 1 — Baseline geolocation risk

Columns:

```text
model
dataset
n_images
Acc@1km
Acc@25km
Acc@200km
Acc@750km
median_error_km
parse_success_rate
```

### Table 2 — Redaction method comparison

Columns:

```text
method
area_budget
Acc@25km
Acc@25km_drop
median_error_km
CLIP_similarity
caption_similarity
pixels_edited
```

### Table 3 — Cue taxonomy

Columns:

```text
cue_category
mean_causal_leakage
median_causal_leakage
percent_top1_regions
percent_top5_regions
mean_area_frac
```

### Table 4 — Transfer

Columns:

```text
source_model
target_model
method
area_budget
Acc@25km_drop
CLIP_similarity
```

### Table 5 — Ablations

Columns:

```text
ablation_type
setting
top5_jaccard
spearman_rank_corr
Acc@25km_drop
utility
```

---

## 21. Suggested final claims after experiments

Only make claims supported by results.

Strong version:

> We find that geolocation leakage is sparse: redacting the top 5% of causal semantic regions reduces Acc@25km by X%, compared with Y% for area-matched random regions.

> We find that Z% of top causal cues are environmental rather than explicit identifiers, showing that standard privacy redaction misses important leakage channels.

> GeoLeakLens improves the privacy-utility Pareto frontier over OCR-only, face/license blurring, random masking, and full-image degradation.

Weak version if results are mixed:

> We introduce a causal measurement framework for geolocation leakage and show that for a subset of images successfully geolocated by modern models, semantic interventions identify interpretable leakage cues and can improve over random redaction.

Do not overclaim.

---

## 22. Checklist before handing results to paper writer

- [ ] **Pre-registered hypotheses committed before running E2/E3/E4**. Write expected results (direction, rough magnitude, threshold for each acceptance criterion in §17) to `outputs/preregistration.md` and git-commit before the first headline run. This makes Claim 2 in particular more credible — that claim could go either way and reviewers will assume it didn't if there's no record.
- [ ] Dataset licenses checked (GSV-Cities + Im2GPS3k).
- [ ] EXIF stripped (verified on a 50-image audit subset).
- [ ] Manifest complete with `contains_*_flag` fields populated per §7.1.
- [ ] Baseline predictions cached for all (model, dataset, prompt) cells.
- [ ] Parse success reported per cell.
- [ ] Single-region attribution scores computed for all (image, region) pairs.
- [ ] Dynamic-greedy joint-causal scores computed (primary headline method).
- [ ] Shapley spot-check (§12.11) results reported.
- [ ] Area-matched baselines implemented per §12.3 protocol; median |Δarea| audit < 0.5pp.
- [ ] Utility metrics (CLIP, caption, LPIPS for Panel B) computed.
- [ ] Paired-bootstrap CIs computed; stratification documented.
- [ ] Figures generated as PDF and PNG with consistent fonts/sizes.
- [ ] Qualitative examples include successes and failures (5 of each minimum).
- [ ] GeoShield comparison: both `geoshield_imperceptible` and `geoshield_visible_eps` runs attempted, results documented (or non-reproduction noted).
- [ ] Human eval (§13.E9) results in main table.
- [ ] Cost report: actual VLM spend per experiment vs. §4.1 estimates.
- [ ] Public demo does not reveal exact addresses; outputs only coarse region.
- [ ] All configs saved with outputs (config hash in every result row per §7.5).
- [ ] Limitations section drafted, listing: conditional-on-success framing, dataset coverage, model coverage, GeoShield cross-paradigm caveats, single-attribute private-attr pilot.
