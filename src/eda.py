"""Day 31 EDA: class balance, image geometry, sample grid.

Output: models/eda_summary.json + docs/samples_grid.png
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from dataset import LABEL_NAMES, decode, load_split_frame

SUMMARY_PATH = PROJECT_ROOT / "models" / "eda_summary.json"
GRID_PATH = PROJECT_ROOT / "docs" / "samples_grid.png"


def main(n_size_sample=300, seed=42):
    stats = {}
    for split in ("train", "test", "validation"):
        df = load_split_frame(split)
        counts = df["label"].value_counts().to_dict()
        stats[split] = {
            "n": len(df),
            "counts": {LABEL_NAMES[k]: int(v) for k, v in counts.items()},
        }
    train = load_split_frame("train").sample(
        n=min(n_size_sample, len(load_split_frame("train"))), random_state=seed
    )
    sizes, modes = [], []
    for _, row in train.iterrows():
        im = decode(row["image"])
        sizes.append(im.size)
        modes.append(im.mode)
    ws, hs = zip(*sizes)
    stats["geometry_sample_n"] = len(sizes)
    stats["width"] = {"min": min(ws), "max": max(ws)}
    stats["height"] = {"min": min(hs), "max": max(hs)}
    stats["modes"] = {m: modes.count(m) for m in sorted(set(modes))}

    # Sample grid: 2 rows (NORMAL/PNEUMONIA) x 4 cols, for the README.
    full_train = load_split_frame("train")
    fig, axes = plt.subplots(2, 4, figsize=(12, 6))
    for r, lab in enumerate((0, 1)):
        picks = full_train[full_train["label"] == lab].sample(4, random_state=seed)
        for c, (_, row) in enumerate(picks.iterrows()):
            axes[r, c].imshow(decode(row["image"]), cmap="gray")
            axes[r, c].axis("off")
            axes[r, c].set_title(LABEL_NAMES[lab] if c == 0 else "")
    fig.suptitle("Chest X-Ray samples: NORMAL (top) vs PNEUMONIA (bottom)")
    fig.tight_layout()
    GRID_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(GRID_PATH, dpi=100)
    plt.close(fig)

    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))
    print(f"saved -> {SUMMARY_PATH}, {GRID_PATH}")


if __name__ == "__main__":
    main()
