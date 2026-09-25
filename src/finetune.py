"""Day 35: fine-tune — unfreeze the top of the backbone, low LR, compare.

Unfreezes MobileNetV2 features[-3:] (last inverted-residual blocks) plus the
head, trains end-to-end at 10x lower LR than the head-only phase. Compares
against the frozen Day-34 metrics on the same test split.

Output: models/mobilenet_finetuned.pt + models/finetune_metrics.json
"""
import json
import sys
import time
from pathlib import Path

import torch
from torch import nn

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import make_loaders
from models import mobilenet_frozen
from train_baseline import evaluate, summarize

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_finetuned.pt"
METRICS_PATH = PROJECT_ROOT / "models" / "finetune_metrics.json"
FROZEN_PATH = PROJECT_ROOT / "models" / "mobilenet_frozen.pt"

EPOCHS = 3
LR = 1e-4  # 10x lower: pretrained weights are already near good values
SEED = 42

torch.manual_seed(SEED)


def main():
    t0 = time.time()
    train_dl, _, test_dl, _ = make_loaders(batch_size=32)
    net = mobilenet_frozen()
    net.load_state_dict(torch.load(FROZEN_PATH, weights_only=True)["state_dict"])

    # Unfreeze only the last blocks + head; keep early edge/texture filters fixed.
    for p in net.features.parameters():
        p.requires_grad = False
    for block in net.features[-3:]:
        for p in block.parameters():
            p.requires_grad = True
    n_trainable = sum(p.numel() for p in net.parameters() if p.requires_grad)
    print(f"trainable params: {n_trainable:,} (frozen backbone was 2,562)", flush=True)

    opt = torch.optim.Adam(
        (p for p in net.parameters() if p.requires_grad), lr=LR
    )
    loss_fn = nn.CrossEntropyLoss()
    net.train()
    for ep in range(1, EPOCHS + 1):
        tot, n = 0.0, 0
        for xb, yb in train_dl:
            opt.zero_grad()
            loss = loss_fn(net(xb), yb)
            loss.backward()
            opt.step()
            tot += loss.item() * len(xb)
            n += len(xb)
        print(f"  epoch {ep}/{EPOCHS} loss={tot / n:.4f}", flush=True)

    y_true, proba = evaluate(net, test_dl)
    out = {"epochs": EPOCHS, "lr": LR, "unfrozen": "features[-3:] + head",
           **summarize(y_true, proba), "elapsed_s": round(time.time() - t0, 1)}
    torch.save({"state_dict": net.state_dict()}, MODEL_PATH)
    METRICS_PATH.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))
    print(f"saved -> {MODEL_PATH}")


if __name__ == "__main__":
    main()
