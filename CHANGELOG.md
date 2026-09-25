# Changelog

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [1.0.0] - 2026-09-25

### Added
- MobileNetV2 pneumonia classifier (test ROC-AUC 0.9622, recall 0.9897)
  with frozen-then-finetuned training path and cached backbone features.
- Small-CNN baseline (ROC-AUC 0.8550) establishing the floor.
- Grad-CAM explainability with central-mass focus check (TP central, FP
  peripheral caught and documented).
- MLflow `pneumonia_xray` experiment (3 runs, backfilled + tagged).
- FastAPI `/predict` (class + confidence + overlay PNG), Pydantic-validated.
- CPU-only Docker image (`bakr1m/pneumonia-api:v1`, 1.63 GB), parity-verified.
- 18 hermetic pytest tests; ruff-clean; 10 executing notebooks.

### Fixed
- MobileNetV2 pooling lives in `forward()`, not `.features` (cache shape bug).
- `requires_grad` flags come from the constructor, not the checkpoint.
- Forward hooks firing under `no_grad` (guarded registration).
- Serving: missing pandas dep, matplotlib-for-colormap (dependency-free jet LUT).
- Train/serve dtype trap (all-None columns -> object dtype -> coerce).
