"""Hermetic model tests (random tensors, no data/ needed)."""
import torch

from src.models import SmallCNN, mobilenet_frozen


def test_small_cnn_forward_shape():
    net = SmallCNN().eval()
    with torch.no_grad():
        out = net(torch.zeros(4, 3, 224, 224))
    assert out.shape == (4, 2)
    assert torch.isfinite(out).all()


def test_small_cnn_param_count_small():
    n = sum(p.numel() for p in SmallCNN().parameters())
    assert n < 300_000  # must stay a *small* baseline


def test_mobilenet_frozen_backbone_head():
    net = mobilenet_frozen().eval()
    frozen = [p for p in net.features.parameters() if not p.requires_grad]
    assert len(frozen) > 0  # backbone locked
    trainable = [p for p in net.parameters() if p.requires_grad]
    assert sum(p.numel() for p in trainable) < 10_000  # head only
    with torch.no_grad():
        out = net(torch.zeros(2, 3, 224, 224))
    assert out.shape == (2, 2)
