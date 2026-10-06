"""Hermetic Grad-CAM tests (random tensors + SmallCNN, no data/ needed)."""
import numpy as np
import torch
from PIL import Image

from src.gradcam import GradCAM, blend_heatmap
from src.models import SmallCNN


def test_heatmap_shape_range():
    net = SmallCNN().eval()
    cam = GradCAM(net, "features.6")
    h = cam.heatmap(torch.zeros(1, 3, 224, 224), target_class=1)
    assert h.shape[0] == h.shape[1] and h.shape[0] <= 224  # pooled map
    assert float(h.min()) >= 0.0 and float(h.max()) <= 1.0


def test_heatmap_responds_to_input():
    torch.manual_seed(42)  # deterministic init: unseeded weights flaked (dead ReLUs -> identical maps)
    net = SmallCNN().eval()
    cam = GradCAM(net, "features.6")
    rng = torch.random.manual_seed(0)
    h1 = cam.heatmap(torch.rand(1, 3, 224, 224, generator=rng), target_class=1)
    h2 = cam.heatmap(torch.ones(1, 3, 224, 224), target_class=1)
    assert not np.allclose(h1, h2)  # content-dependent, not constant


def test_blend_output_size():
    img = Image.new("RGB", (100, 80), color=(10, 10, 10))
    out = blend_heatmap(img, np.linspace(0, 1, 80 * 100).reshape(80, 100))
    assert out.size == (100, 80) and out.mode == "RGB"
