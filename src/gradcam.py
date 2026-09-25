"""Day 36: Grad-CAM — visualize where the model looks.

Hooks the last convolutional block, weights its activation maps by
global-average-pooled class gradients, and overlays the heatmap on the
original X-ray. Clinical purpose: verify focus on lung fields, not text
labels, borders, or other shortcut artifacts.
"""
import sys
from pathlib import Path

import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))


class GradCAM:
    """Gradient-weighted Class Activation Mapping for any conv layer."""

    def __init__(self, model, target_layer):
        self.model = model.eval()
        modules = dict(model.named_modules())
        assert target_layer in modules, f"{target_layer} not in model"
        self.activations = {}
        self.gradients = {}

        def fwd_hook(_mod, _inp, out):
            self.activations["v"] = out.detach()
            if out.requires_grad:
                out.register_hook(lambda g: self.gradients.__setitem__("v", g))

        modules[target_layer].register_forward_hook(fwd_hook)

    @torch.no_grad()
    def _check(self, x):
        assert x.dim() == 4, "expected batched (B,C,H,W) input"

    def heatmap(self, x, target_class):
        """H x W importance map in [0, 1] for target_class (batched x, uses row 0)."""
        self._check(x)
        self.model.zero_grad()
        logits = self.model(x)
        score = logits[0, target_class]
        score.backward()
        acts = self.activations["v"][0]  # C x h x w
        grads = self.gradients["v"][0]
        weights = grads.mean(dim=(1, 2), keepdim=True)
        cam = torch.relu((weights * acts).sum(dim=0))
        cam = cam / (cam.max() + 1e-8)
        return cam.detach().cpu().numpy()

    def overlay(self, pil_img, cam, alpha=0.45):
        """Blend heatmap onto the original image; returns PIL image."""
        return blend_heatmap(pil_img, cam, alpha)


def _jet_lut(n=256):
    """Textbook blue-cyan-yellow-red ramp as a uint8 LUT (no mpl dependency).

    jet(x): r=clip(1.5-|4x-3|), g=clip(1.5-|4x-2|), b=clip(1.5-|4x-1|).
    Hues may differ slightly from any one library's table; spatial pattern
    (the clinically relevant part) is unaffected. Serving must not haul
    matplotlib for a colormap.
    """
    x = np.linspace(0, 1, n)
    lut = np.clip(
        np.stack(
            [1.5 - np.abs(4 * x - 3), 1.5 - np.abs(4 * x - 2), 1.5 - np.abs(4 * x - 1)],
            axis=1,
        ),
        0,
        1,
    )
    return (lut * 255).astype(np.uint8)


def blend_heatmap(pil_img, cam, alpha=0.45):
    """Jet-colormap blend of a [0,1] cam onto an image (module-level helper)."""
    from PIL import Image as PImage

    lut = _jet_lut()
    idx = (np.clip(np.asarray(cam, dtype=float), 0, 1) * 255).astype(np.uint8)
    h, w = pil_img.size[1], pil_img.size[0]
    heat = np.asarray(PImage.fromarray(lut[idx]).resize((w, h)))
    base = np.asarray(pil_img.convert("RGB"))
    return PImage.fromarray(((1 - alpha) * base + alpha * heat).astype(np.uint8))


def save_overlay(pil_img, cam, path, alpha=0.45):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    blend_heatmap(pil_img, cam, alpha).save(path)
    return path
