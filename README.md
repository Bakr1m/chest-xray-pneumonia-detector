# Project 3: Chest X-Ray Pneumonia Detector

**Days 31–40 | Healthcare ML Portfolio**

## Business Context

Radiology triage support: flag likely-pneumonia chest X-rays for prioritized
review so critical cases reach a radiologist first. Decision support only —
never an autonomous diagnostic device.

## Dataset

- **Source**: `hf-vision/chest-xray-pneumonia` (Hugging Face mirror of the
  Kaggle "Chest X-Ray Images (Pneumonia)" set, CC BY 4.0) — verified
  Kaggle-identical by split counts
- **Scale**: train 5,216 (PNEUMONIA 3,875 / 74.3%), test 624 (390 / 62.5%),
  official val 16 (8/8 — unusable, stratified 521-image val carved from train)
- **Images**: JPEG, RGB, 441–2772 px wide → resized to 224, ImageNet normalized
- **Layout**: parquet shards under `data/hf_chestxray/data/` (gitignored)

## Approach

1. **EDA** (Day 31): balance, geometry, sample grid.
2. **Pipeline** (Day 32): train-only augmentation (flip/rotation/brightness);
   deterministic eval; stratified val split.
3. **Baseline** (Day 33): 3-conv SmallCNN from scratch (~150k params).
4. **Transfer** (Day 34): frozen MobileNetV2 + new head, cached 1280-dim
   features (CPU-feasible: one slow pass, seconds per epoch after).
5. **Fine-tune** (Day 35): unfreeze `features[-3:]` + head at 1e-4.
6. **Evaluation + Grad-CAM** (Day 36): consolidated metrics; TP/TN/FP overlays
   with central-mass focus check.
7. **Tracking + tests** (Day 37): MLflow `pneumonia_xray`; exact tensor-contract
   preprocessing tests.
8. **Serving** (Day 38): FastAPI `/predict` (class + confidence + overlay).
9. **Container** (Day 39): 1.63 GB CPU-only image, parity-verified.

## Results

| Model | Accuracy | ROC-AUC | PR-AUC | F1 (P / R) |
|-------|----------|---------|--------|------------|
| SmallCNN scratch | 0.7837 | 0.8550 | 0.9055 | 0.82 (0.86 / 0.79) |
| MobileNetV2 frozen | 0.8638 | 0.9584 | 0.9735 | 0.90 (0.84 / 0.97) |
| **MobileNetV2 finetuned** | **0.8686** | **0.9622** | **0.9714** | **0.90 (0.83 / 0.99)** |

- Only **4 missed pneumonias** out of 390 at 0.5 (recall 0.99).
- Grad-CAM: TP focus central (0.63 mass); FP at 0.99 confidence focuses
  peripherally (0.26) — artifact reliance caught, documented as the lead limit.

## Limitations

1. **Single-source pediatrics** (Guangzhou, ages 1–5): no adult, no multi-site
   validation — generalizability unproven.
2. **Shortcut evidence**: the most confident FP attends borders/corners; text
   markers and positioning artifacts are a known risk in this dataset.
3. **Retrospective, small test** (624 images): not a clinical validation study.
4. **Sparse-early-window analogue**: cropped/rotated/low-quality scans degrade
   silently — input QC needed before scoring.
5. **No calibration**: outputs are ranking scores; quote bands/thresholds, not
   precise probabilities.

## Ethical Considerations

- **Triage aid, never diagnosis**: every flag needs radiologist review; 1-in-6
  flags are false alarms at current precision.
- **Regulatory reality**: a diagnostic claim needs clinical validation studies,
  prospective testing, and FDA clearance — this repo is a proof of concept.
- **Fairness**: pediatric-only training must not be applied to adults without
  revalidation; audit across age/sites before any pilot.
- **Transparency**: every prediction ships its Grad-CAM overlay for inspection.

## Project Structure

```
project3_pneumonia/
├ data/            # hf_chestxray parquet + feature cache (gitignored)
├ notebooks/       # 01_explore ... 10_wrapup (all execute clean)
├ src/             # dataset, data, models, train_*, finetune, gradcam,
│                  #   explain, backfill_mlflow, serve
├ api/main.py      # thin entrypoint
├ tests/           # 18 hermetic tests (synthetic images only)
├ models/          # *.pt/*.json (gitignored)
├ mlruns/          # MLflow tracking (gitignored)
├ docs/            # samples_grid.png, gradcam_*.png, model_card.md
├ example via live curl (see Run with Docker)
├ Dockerfile (1.63 GB) + requirements-serve.txt (CPU torch, serving only)
├ requirements.txt (full local) / requirements-train.txt (torch+mlflow)
└ README.md
```

## Quick Start

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
make test     # 18 hermetic tests, no data/ needed
# training needs data/ (see Dataset): python src/train_baseline.py, etc.
python api/main.py   # :8000
```

## Try It in 60 Seconds (no local setup needed)

```bash
docker pull bakr1m/pneumonia-api:latest
docker run -d --name pneumonia -p 8003:8000 bakr1m/pneumonia-api:latest
curl http://localhost:8003/health
# {"status":"healthy"}
curl -X POST http://localhost:8003/predict -F "file=@example_smoke.png"
# -> {"predicted_class":"NORMAL","confidence":0.6433,"gradcam_png_base64":"..."}
docker stop pneumonia && docker rm pneumonia
```

`example_smoke.png` (in this repo) is a 1×1 pixel — it proves the full
wiring (upload → model → class + confidence + Grad-CAM overlay), not the
medicine. For a meaningful result, substitute any chest X-ray PNG for the
file; the endpoint resizes to 224×224 itself. Verified live against
`:latest`.

## Run with Docker

```bash
docker pull bakr1m/pneumonia-api:latest
docker run -p 8000:8000 bakr1m/pneumonia-api:latest
curl -X POST http://localhost:8000/predict -F "file=@xray.png"
# -> {"predicted_class":"PNEUMONIA","confidence":0.9775,"gradcam_png_base64":"..."}
```

## Problems Encountered (Build & Deploy)

1. **CI smoke assumed Pillow on the runner.** The smoke step generated its
   test PNG with `from PIL import Image` — which runs on the GitHub runner,
   not in the container, where Pillow isn't installed. Replaced with a
   stdlib base64 PNG (no dependency can be missing from the standard
   library). Lesson now applied fleet-wide: smoke scripts may only assume
   the base runner image.
2. **Release draft with no asset.** An interrupted upload left a draft
   `v1.0.0` with zero assets and a tag the delete command couldn't find
   (HTTP 422). Fixed by completing the upload in the background and
   verifying asset state before the Dockerfile consumed it.
3. **Torch CPU image weight.** The serving image needs torch CPU (~1.6 GB
   total) — accepted as the price of Grad-CAM at serve time, and the reason
   the Dockerfile installs from `requirements-serve.txt` only.
4. **The model cheats a little.** Grad-CAM showed the most confident false
   positive attending image borders/corners (0.26 central mass vs 0.63 for
   true positives) — text markers and positioning artifacts are a known
   shortcut in this dataset. Documented as the lead limitation, not tuned
   away silently.

## Key Learnings

1. **Imbalance disqualifies accuracy on day one** (74% for free).
2. **Transfer > scratch by +0.10 ROC** on 5k images; fine-tuning adds +0.004 —
   know when to stop digging.
3. **Freeze-then-thaw with 10x lower LR** protects pretrained filters.
4. **Abstractions leak**: pooling lives in `forward()`, flags in constructors —
   verify shapes/flags at every joint.
5. **Grad-CAM is a test the model can fail** — ours half-did, honestly reported.
6. **Serving deps derive from the serving import graph** (pandas, no matplotlib).
7. **CPU-only torch is a deployment decision** (~2 GB CUDA dead weight avoided).
