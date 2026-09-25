"""Day 32: image data pipeline — resize/normalize always, augment train only.

- Train: Resize(224) + rotation/flip/brightness + ImageNet normalize.
- Val/test: Resize(224) + ImageNet normalize, fully deterministic.
- The official 16-image val split is unusable, so a stratified val fraction
  is carved out of train (seeded, recorded in the returned index lists).
"""
import sys
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import DATA_DIR, ChestXrayDataset

IMG_SIZE = 224
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def train_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomRotation(degrees=10),
            transforms.ColorJitter(brightness=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def eval_transforms():
    return transforms.Compose(
        [
            transforms.Resize((IMG_SIZE, IMG_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )


def stratified_val_indices(labels, val_frac=0.1, seed=42):
    """Per-class index split -> (train_idx, val_idx), deterministic."""
    rng = np.random.RandomState(seed)
    tr, va = [], []
    for cls in (0, 1):
        idx = np.nonzero(np.asarray(labels) == cls)[0]
        rng.shuffle(idx)
        cut = int(len(idx) * val_frac)
        va.extend(idx[:cut].tolist())
        tr.extend(idx[cut:].tolist())
    return sorted(tr), sorted(va)


def make_loaders(batch_size=32, val_frac=0.1, seed=42, data_dir=DATA_DIR,
                 num_workers=0, train_limit=None):
    """Train (augmented) + val (deterministic) + test (deterministic) loaders."""
    full_train = ChestXrayDataset("train", transform=None, limit=train_limit,
                                  data_dir=data_dir)
    labels = [int(full_train.frame.iloc[i]["label"]) for i in range(len(full_train))]
    tr_idx, va_idx = stratified_val_indices(labels, val_frac, seed)
    train_ds = Subset(
        ChestXrayDataset("train", transform=train_transforms(), limit=train_limit,
                         data_dir=data_dir),
        tr_idx,
    )
    val_ds = Subset(
        ChestXrayDataset("train", transform=eval_transforms(), limit=train_limit,
                         data_dir=data_dir),
        va_idx,
    )
    test_ds = ChestXrayDataset("test", transform=eval_transforms(), data_dir=data_dir)
    kw = {"batch_size": batch_size, "num_workers": num_workers}
    return (
        DataLoader(train_ds, shuffle=True, **kw),
        DataLoader(val_ds, shuffle=False, **kw),
        DataLoader(test_ds, shuffle=False, **kw),
        {"train_idx": tr_idx, "val_idx": va_idx},
    )
