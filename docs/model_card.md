# Model Card — Chest X-Ray Pneumonia Detector v1.0.0

## Model details
- **Architecture:** MobileNetV2 (ImageNet), `features[-3:]` + head fine-tuned
  at 1e-4 after frozen-head phase. 224px RGB in, ImageNet-normalized.
- **Artifact:** `models/mobilenet_finetuned.pt` (9.1 MB); served via FastAPI
  with Grad-CAM overlay; Docker `bakr1m/pneumonia-api:v1` (1.63 GB, CPU-only).

## Intended use
- **Triage support only:** prioritize likely-pneumonia X-rays for radiologist
  review. **Out of scope:** diagnosis, autonomous decisions, adult or
  multi-site use without revalidation.

## Training data
- `hf-vision/chest-xray-pneumonia` (Kaggle mirror, CC BY 4.0): train 5,216
  (74% pneumonia), test 624, official val 16 (replaced by stratified 521 split).

## Evaluation
- Test: accuracy 0.8686, ROC-AUC 0.9622, PR-AUC 0.9714, F1 0.904
  (precision 0.832, recall 0.990 — 4 missed of 390).
- Grad-CAM: TP mid-lung (0.63 central mass); confident FP (0.99) peripheral
  (0.26) — artifact-reliance flag, disclosed as the lead limitation.

## Limitations & ethics
- Pediatric single-source data; retrospective 624-image test; no calibration;
  ~1-in-6 flags false alarms. Regulatory clearance + prospective studies
  required before any clinical use. See README for the full list.
