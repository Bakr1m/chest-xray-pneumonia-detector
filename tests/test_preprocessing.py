"""Preprocessing contract tests: exact shape/dtype/value-range behavior.

Silent shape mismatches and wrong normalization are the classic hard-to-debug
image-pipeline bugs — these pin known-input -> known-output behavior.
"""
import torch
from PIL import Image

from src.data import eval_transforms


def test_eval_output_contract():
    t = eval_transforms()(Image.new("RGB", (500, 300), color=(128, 128, 128)))
    assert t.shape == (3, 224, 224) and t.dtype == torch.float32
    assert torch.isfinite(t).all()


def test_imagenet_normalization_exact():
    """Solid 128-gray must map to the exact per-channel normalized values."""
    t = eval_transforms()(Image.new("RGB", (64, 64), color=(128, 128, 128)))
    v = 128 / 255
    expected = [(v - 0.485) / 0.229, (v - 0.456) / 0.224, (v - 0.406) / 0.225]
    for c in range(3):
        assert torch.allclose(
            t[c], torch.full((224, 224), expected[c], dtype=torch.float32), atol=1e-5
        )


def test_eval_deterministic_across_sizes():
    t = eval_transforms()
    a = Image.new("RGB", (441, 194), color=(200, 100, 50))
    b = Image.new("RGB", (2772, 2628), color=(200, 100, 50))
    assert torch.equal(t(a), t(a))  # same input -> same tensor
    assert t(b).shape == (3, 224, 224)  # extremes resize cleanly
