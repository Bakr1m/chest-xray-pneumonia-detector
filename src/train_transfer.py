"""Day 34: transfer learning — frozen MobileNetV2 backbone + new head.

Speed trick for CPU: backbone features are cached to disk once
(data/cache_mobilenet/*.pt); the classifier head then trains on 1280-dim
vectors in seconds. Re-running with a warm cache skips all CNN forward passes.

Output: models/mobilenet_frozen.pt + models/transfer_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import make_loaders
from models import mobilenet_frozen
from train_baseline import summarize

CACHE_DIR = PROJECT_ROOT / "data" / "cache_mobilenet"
MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_frozen.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "transfer_metrics.json"

EPOCHS = 15
LR = 1e-3
SEED = 42

torch.manual_seed(SEED)


@torch.no_grad()
def cache_split(loader, backbone, name):
    """Run the frozen backbone once; store (features, labels) on disk."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    feats, labels = [], []
    for xb, yb in loader:
        x = backbone(xb)
        # Mirror MobileNetV2.forward: pool + flatten live outside .features.
        x = torch.flatten(F.adaptive_avg_pool2d(x, (1, 1)), 1)
        feats.append(x)
        labels.append(yb)
    X = torch.cat(feats)
    y = torch.cat(labels)
    torch.save({"X": X, "y": y}, CACHE_DIR / f"{name}.pt")
    print(f"  cached {name}: {len(y)} imgs", flush=True)


def cached_tensors(name):
    d = torch.load(CACHE_DIR / f"{name}.pt", weights_only=True)
    return d["X"], d["y"]


def train_head(head, Xtr, ytr, epochs=EPOCHS, lr=LR, batch=256):
    head.train()
    opt = torch.optim.Adam(head.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    ds = torch.utils.data.TensorDataset(Xtr, ytr)
    dl = torch.utils.data.DataLoader(ds, batch_size=batch, shuffle=True)
    for ep in range(1, epochs + 1):
        tot, n = 0.0, 0
        for xb, yb in dl:
            opt.zero_grad()
            loss = loss_fn(head(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        print(f"  epoch {ep}/{epochs} loss={tot / n:.4f}", flush=True)
    return head


def main():
    t0 = time.time()
    train_dl, val_dl, test_dl, _ = make_loaders(batch_size=32)
    net = mobilenet_frozen()
    backbone = net.features.eval()
    for p in backbone.parameters():
        p.requires_grad = False

    if not (CACHE_DIR / "train.pt").exists():
        print("caching backbone features (one slow pass)...", flush=True)
        cache_split(train_dl, backbone, "train")
        cache_split(val_dl, backbone, "val")
        cache_split(test_dl, backbone, "test")
    else:
        print("warm cache found — skipping CNN passes", flush=True)

    Xtr, ytr = cached_tensors("train")
    Xte, yte = cached_tensors("test")
    print(f"cached shapes: train {tuple(Xtr.shape)}, test {tuple(Xte.shape)}", flush=True)
    head = train_head(net.classifier, Xtr, ytr)

    full = mobilenet_frozen()
    full.classifier = head
    full.eval()
    with torch.no_grad():
        proba = full.classifier(Xte).softmax(1)[:, 1].tolist()
    out = {"epochs": EPOCHS, "lr": LR, "backbone": "mobilenet_v2_frozen",
           **summarize(yte.tolist(), proba),
           "elapsed_s": round(time.time() - t0, 1)}
    torch.save({"state_dict": full.state_dict()}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
