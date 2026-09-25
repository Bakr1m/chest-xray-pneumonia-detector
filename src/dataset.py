"""Day 31: dataset access for the HF mirror of Kaggle chest X-rays.

Provenance: `hf-vision/chest-xray-pneumonia` (CC BY 4.0), verified identical
to the Kaggle layout by split counts — train 5,216 (1,341 NORMAL + 3,875
PNEUMONIA), test 624 (234 + 390), val 16 (8 + 8). Parquet shards under
data/hf_chestxray/data/ with [image ({bytes}), label (0/1)] columns.
"""
import glob
import io
from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "hf_chestxray" / "data"

LABEL_NAMES = {0: "NORMAL", 1: "PNEUMONIA"}


def split_files(split, data_dir=DATA_DIR):
    return sorted(glob.glob(str(Path(data_dir) / f"{split}-*.parquet")))


def load_split_frame(split, data_dir=DATA_DIR):
    """Split's [image, label] table (images stay packed until decoded)."""
    files = split_files(split, data_dir)
    assert files, f"no parquet shards for split={split} in {DATA_DIR}"
    return pd.concat(
        [pd.read_parquet(f, columns=["image", "label"]) for f in files],
        ignore_index=True,
    )


def decode(img_dict):
    """Packed image dict -> RGB PIL image."""
    return Image.open(io.BytesIO(img_dict["bytes"])).convert("RGB")


class ChestXrayDataset(Dataset):
    """Torch dataset over one split; transform applied after RGB decode."""

    def __init__(self, split, transform=None, limit=None, data_dir=DATA_DIR):
        self.frame = load_split_frame(split, data_dir)
        if limit is not None:
            self.frame = self.frame.iloc[:limit].reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.frame)

    def __getitem__(self, idx):
        row = self.frame.iloc[idx]
        img = decode(row["image"])
        label = int(row["label"])
        if self.transform is not None:
            img = self.transform(img)
        return img, label
