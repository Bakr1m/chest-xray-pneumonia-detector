"""Hermetic dataset tests (synthetic PNGs in tmp dir, no data/ needed)."""
import io

import pandas as pd
import pytest
import torch
from PIL import Image

from src.dataset import ChestXrayDataset, decode


def pack(color, size=(64, 48)):
    buf = io.BytesIO()
    Image.new("L", size, color=color).save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


@pytest.fixture
def tiny_parquet(tmp_path):
    rows = [
        {"image": pack(0), "label": 0},
        {"image": pack(255), "label": 1},
    ]
    pd.DataFrame(rows).to_parquet(tmp_path / "train-00000-of-00001.parquet")
    return tmp_path


def test_decode_returns_rgb():
    im = decode(pack(128))
    assert im.mode == "RGB"
    assert im.size == (64, 48)


def test_dataset_len_labels_transform(tiny_parquet):
    ds = ChestXrayDataset("train", data_dir=tiny_parquet)
    assert len(ds) == 2
    img, label = ds[0]
    assert label == 0 and isinstance(img, Image.Image)

    ds_t = ChestXrayDataset(
        "train", transform=lambda im: torch.zeros(3, 8, 8), data_dir=tiny_parquet
    )
    t, label = ds_t[1]
    assert label == 1 and t.shape == (3, 8, 8)


def test_dataset_missing_split_raises(tiny_parquet):
    with pytest.raises(AssertionError):
        ChestXrayDataset("test", data_dir=tiny_parquet)
