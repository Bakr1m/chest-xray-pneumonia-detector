"""Hermetic pipeline tests (synthetic parquet in tmp dir, no data/ needed)."""
import io

import pandas as pd
import torch
from PIL import Image

from src.data import (
    eval_transforms,
    make_loaders,
    stratified_val_indices,
    train_transforms,
)


def pack(seed, size=(100, 80)):
    rng = __import__("numpy").random.RandomState(seed)
    arr = (rng.rand(size[1], size[0], 3) * 255).astype("uint8")
    buf = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buf, format="PNG")
    return {"bytes": buf.getvalue()}


def write_shard(path, n_per_class, start_seed=0):
    rows = []
    for i in range(n_per_class):
        rows.append({"image": pack(start_seed + i), "label": 0})
        rows.append({"image": pack(start_seed + 100 + i), "label": 1})
    pd.DataFrame(rows).to_parquet(path)


def test_batch_shape_dtype(tmp_path):
    write_shard(tmp_path / "train-00000-of-00001.parquet", 20)
    write_shard(tmp_path / "test-00000-of-00001.parquet", 4)
    tr, va, te, _ = make_loaders(batch_size=4, data_dir=tmp_path)
    xb, yb = next(iter(tr))
    assert xb.shape == (4, 3, 224, 224) and xb.dtype == torch.float32
    assert yb.tolist() and all(v in (0, 1) for v in yb.tolist())
    assert len(va.dataset) > 0 and len(te.dataset) == 8


def test_train_augments_eval_does_not(tmp_path):
    write_shard(tmp_path / "train-00000-of-00001.parquet", 4)
    write_shard(tmp_path / "test-00000-of-00001.parquet", 2)
    t = train_transforms()
    e = eval_transforms()
    img = Image.new("RGB", (100, 80), color=(128, 128, 128))
    assert not torch.equal(t(img), t(img))  # random aug varies
    assert torch.equal(e(img), e(img))  # eval deterministic


def test_stratified_split_preserves_ratio():
    labels = [0] * 20 + [1] * 60
    tr, va = stratified_val_indices(labels, val_frac=0.1, seed=0)
    assert len(tr) + len(va) == 80 and not set(tr) & set(va)
    va_labels = [labels[i] for i in va]
    assert va_labels.count(0) == 2 and va_labels.count(1) == 6
