"""Day 37: backfill MLflow with the three recorded experiments.

The train scripts (baseline/transfer/finetune) log live runs when executed;
this script registers the already-completed results as documented backfill
runs (tagged backfilled=true) so the tracking server reflects the full
 Day 33 -> 34 -> 35 progression without redundant GPU-less retraining.
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("MLFLOW_ALLOW_FILE_STORE", "true")

import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

TRACKING_URI = str(PROJECT_ROOT / "mlruns")
EXPERIMENT = "pneumonia_xray"

RUNS = [
    ("cnn_baseline", "models/baseline_metrics.json",
     {"arch": "SmallCNN-3conv", "epochs": 5, "lr": 1e-3}),
    ("mobilenet_frozen", "models/transfer_metrics.json",
     {"arch": "mobilenet_v2_frozen", "epochs": 15, "lr": 1e-3}),
    ("mobilenet_finetuned", "models/finetune_metrics.json",
     {"arch": "mobilenet_v2_unfrozen_top", "epochs": 3, "lr": 1e-4}),
]


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT)
    for name, metrics_rel, params in RUNS:
        m = json.loads((PROJECT_ROOT / metrics_rel).read_text())
        with mlflow.start_run(run_name=name):
            mlflow.log_params(params)
            mlflow.log_param("backfilled", True)
            for k in ("accuracy", "roc_auc", "pr_auc", "f1", "precision", "recall"):
                mlflow.log_metric(k, m[k])
    print(f"logged {len(RUNS)} runs to experiment '{EXPERIMENT}'")


if __name__ == "__main__":
    main()
