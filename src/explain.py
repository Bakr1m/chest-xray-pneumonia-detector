"""Day 36: consolidated evaluation + Grad-CAM overlays on real test images.

Picks one true-positive pneumonia, one true-negative normal, and one false
positive from the test split (via the finetuned model), saves Grad-CAM
overlays to docs/, and records a quantitative focus check: fraction of cam
mass inside the central lung region vs borders/corners (shortcut sniff test).

Output: docs/gradcam_tp.png, docs/gradcam_tn.png, docs/gradcam_fp.png
        + models/gradcam_check.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from data import eval_transforms, make_loaders
from dataset import decode, load_split_frame
from gradcam import GradCAM, save_overlay
from models import mobilenet_frozen

MODEL_PATH = PROJECT_ROOT / "models" / "mobilenet_finetuned.pt"
OUT_PATH = PROJECT_ROOT / "models" / "gradcam_check.json"
TARGET_LAYER = "features.18"  # last conv block of MobileNetV2 features


def central_mass(cam, border=0.2):
    """Fraction of heat inside the central region (borders = artifact zone)."""
    h, w = cam.shape
    bh, bw = int(h * border), int(w * border)
    inner = cam[bh : h - bh, bw : w - bw]
    return float(inner.sum() / (cam.sum() + 1e-8))


@torch.no_grad()
def predict_all(model, loader):
    probs, labels = [], []
    for xb, yb in loader:
        probs.extend(model(xb).softmax(1)[:, 1].tolist())
        labels.extend(yb.tolist())
    return np.asarray(labels), np.asarray(probs)


def main():
    _, _, test_dl, _ = make_loaders(batch_size=64)
    net = mobilenet_frozen()
    net.load_state_dict(torch.load(MODEL_PATH, weights_only=True)["state_dict"])
    net.eval()
    # Flags come from the frozen constructor; weights are finetuned. Re-enable
    # grad flow for attribution (inference values unchanged).
    net.requires_grad_(True)
    y_true, proba = predict_all(net, test_dl)
    pred = (proba >= 0.5).astype(int)

    test_frame = load_split_frame("test").reset_index(drop=True)
    want = {}
    for kind, mask in (
        ("tp", (y_true == 1) & (pred == 1)),
        ("tn", (y_true == 0) & (pred == 0)),
        ("fp", (y_true == 0) & (pred == 1)),
    ):
        idx = np.nonzero(mask)[0]
        want[kind] = int(idx[0]) if len(idx) else None
    print(f"picks: {want}", flush=True)

    tf = eval_transforms()
    explainer = GradCAM(net, TARGET_LAYER)
    check = {}
    for kind, idx in want.items():
        if idx is None:
            continue
        row = test_frame.iloc[idx]
        pil_img = decode(row["image"])
        x = tf(pil_img).unsqueeze(0)
        with torch.enable_grad():
            cam = explainer.heatmap(x, target_class=int(y_true[idx]))
        path = PROJECT_ROOT / "docs" / f"gradcam_{kind}.png"
        save_overlay(pil_img, cam, path)
        mass = round(central_mass(cam), 3)
        check[kind] = {
            "test_index": idx,
            "true_label": int(y_true[idx]),
            "predicted_proba": round(float(proba[idx]), 4),
            "cam_central_mass": mass,
            "overlay": str(path.relative_to(PROJECT_ROOT)),
        }
        print(f"{kind}: proba={proba[idx]:.3f} central_mass={mass}", flush=True)
    OUT_PATH.write_text(json.dumps(check, indent=2))
    print(f"saved -> {OUT_PATH}")


if __name__ == "__main__":
    main()
