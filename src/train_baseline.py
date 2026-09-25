"""Day 33: small-CNN baseline trained from scratch (the floor).

Output: models/cnn_baseline.pt + models/baseline_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import make_loaders
from models import SmallCNN

MODEL_PATH = PROJECT_ROOT / "models" / "cnn_baseline.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "baseline_metrics.json"

EPOCHS = 5
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)


def train_epoch(model, loader, opt, loss_fn):
    model.train()
    tot, n = 0.0, 0
    for xb, yb in loader:
        opt.zero_grad()
        loss = loss_fn(model(xb), yb)
        loss.backward()
        opt.step()
        tot += loss.item() * len(xb)
        n += len(xb)
    return tot / n


@torch.no_grad()
def evaluate(model, loader):
    model.eval()
    probs, labels = [], []
    for xb, yb in loader:
        probs.extend(model(xb).softmax(1)[:, 1].tolist())
        labels.extend(yb.tolist())
    return labels, probs


def summarize(y_true, proba, threshold=0.5):
    import numpy as np

    y_true = np.asarray(y_true)
    proba = np.asarray(proba)
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()
    return {
        "accuracy": round(float(accuracy_score(y_true, pred)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, proba)), 4),
        "pr_auc": round(float(average_precision_score(y_true, proba)), 4),
        "f1": round(float(f1_score(y_true, pred)), 4),
        "precision": round(float(precision_score(y_true, pred)), 4),
        "recall": round(float(recall_score(y_true, pred)), 4),
        "confusion": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
    }


def main():
    t0 = time.time()
    train_dl, _, test_dl, _ = make_loaders(batch_size=32)
    # Class weights from the training split (inverse frequency).
    tr_labels = [int(train_dl.dataset[i][1]) for i in range(len(train_dl.dataset))]
    import numpy as np

    counts = np.bincount(tr_labels, minlength=2).astype(float)
    weight = torch.tensor(counts.sum() / (2 * counts), dtype=torch.float32)
    print(f"train rows: {len(tr_labels)}, class counts: {counts.tolist()}", flush=True)

    model = SmallCNN()
    opt = torch.optim.Adam(
        (p for p in model.parameters() if p.requires_grad), lr=LR
    )
    loss_fn = nn.CrossEntropyLoss(weight=weight)
    for ep in range(1, EPOCHS + 1):
        loss = train_epoch(model, train_dl, opt, loss_fn)
        print(f"  epoch {ep}/{EPOCHS} loss={loss:.4f}", flush=True)

    y_true, proba = evaluate(model, test_dl)
    out = {"epochs": EPOCHS, "lr": LR, **summarize(y_true, proba),
           "elapsed_s": round(time.time() - t0, 1)}
    torch.save({"state_dict": model.state_dict()}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
