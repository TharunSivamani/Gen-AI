from datasets import load_dataset
from collections import defaultdict
from tqdm import tqdm
from PIL import Image
import torch
import numpy as np

# ============================================================
# 🧩 Config
# ============================================================
DATASET_PATH = "/home/cloudlyte/tharun/tiny-imagenet"  # Local HF dataset path
SPLIT = "train"  # or "valid"

# ============================================================
# 📦 Load Dataset
# ============================================================
print(f"📦 Loading Tiny ImageNet from {DATASET_PATH} (split={SPLIT})...")
dataset = load_dataset(DATASET_PATH, split=SPLIT)
print(f"✅ Loaded {len(dataset)} samples.")


# ============================================================
# 🧠 Analyze Channels
# ============================================================
def get_channels(img):
    """
    Determine number of channels for either PIL.Image or np.ndarray or torch.Tensor.
    """
    if isinstance(img, Image.Image):
        return len(img.getbands())
    elif isinstance(img, torch.Tensor):
        return img.size(0) if img.ndim == 3 else 1
    elif isinstance(img, np.ndarray):
        if img.ndim == 2:
            return 1
        elif img.ndim == 3:
            return img.shape[2]
    return -1  # unknown


def analyze_dataset(dataset):
    per_class_stats = defaultdict(lambda: {"total": 0, "grayscale": 0})
    total = 0
    gray = 0

    for example in tqdm(dataset, desc=f"Analyzing {SPLIT}"):
        img = example["image"]
        label = example["label"]

        n_channels = get_channels(img)

        per_class_stats[label]["total"] += 1
        total += 1

        if n_channels == 1:
            per_class_stats[label]["grayscale"] += 1
            gray += 1

    # ============================================================
    # 🧾 Summary
    # ============================================================
    print(f"\n📊 Tiny ImageNet {SPLIT} Split Statistics:")
    print(f"Total images: {total}")
    print(f"Grayscale images: {gray}")
    print(f"Percentage grayscale: {100.0 * gray / total:.3f}%")

    print("\n🧩 Per-Class Breakdown (top 10 by grayscale count):")
    sorted_classes = sorted(
        per_class_stats.items(), key=lambda x: x[1]["grayscale"], reverse=True
    )
    for cls, stats in sorted_classes[:10]:
        pct = 100.0 * stats["grayscale"] / stats["total"]
        print(
            f"Class {cls:3d}: {stats['grayscale']}/{stats['total']} grayscale ({pct:.2f}%)"
        )

    return per_class_stats


# ============================================================
# 🏁 Run Analysis
# ============================================================
if __name__ == "__main__":
    stats = analyze_dataset(dataset)

"""
📦 Loading Tiny ImageNet from /home/cloudlyte/tharun/tiny-imagenet (split=train)...
✅ Loaded 100000 samples.
Analyzing train: 100%|██████████████████████████████████████████████████| 100000/100000 [00:10<00:00, 9174.99it/s]

📊 Tiny ImageNet train Split Statistics:
Total images: 100000
Grayscale images: 1821
Percentage grayscale: 1.821%

🧩 Per-Class Breakdown (top 10 by grayscale count):
Class 101: 48/500 grayscale (9.60%)
Class  59: 47/500 grayscale (9.40%)
Class 111: 46/500 grayscale (9.20%)
Class  71: 40/500 grayscale (8.00%)
Class 116: 37/500 grayscale (7.40%)
Class  58: 35/500 grayscale (7.00%)
Class 145: 35/500 grayscale (7.00%)
Class  84: 34/500 grayscale (6.80%)
Class  94: 34/500 grayscale (6.80%)
Class 120: 33/500 grayscale (6.60%)
"""
