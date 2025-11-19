import torch
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image

# -------------------------------------------------------
# Convert images to tensor (no normalization, no aug)
# -------------------------------------------------------
to_tensor = transforms.Compose(
    [
        transforms.Lambda(lambda img: img.convert("RGB")),
        transforms.ToTensor(),  # values in [0,1]
    ]
)


# -------------------------------------------------------
# Function to compute mean and std
# -------------------------------------------------------
def compute_mean_std(dataset):
    """
    dataset: HF dataset with key "image"
    returns: mean (3,), std (3,)
    """
    n_images = len(dataset)
    mean = torch.zeros(3)
    std = torch.zeros(3)

    print(f"Computing statistics on {n_images} images...")

    for item in tqdm(dataset):
        img = item["image"]
        img = to_tensor(img)  # (3, H, W)

        mean += img.mean(dim=[1, 2])
        std += img.std(dim=[1, 2])

    mean /= n_images
    std /= n_images

    return mean, std


# -------------------------------------------------------
# Load Tiny ImageNet dataset
# -------------------------------------------------------
DATA_PATH = "/home/cloudlyte/tharun/tiny-imagenet"
dataset = load_dataset(DATA_PATH)

train_set = dataset["train"]
val_set = dataset["valid"]

# -------------------------------------------------------
# Compute stats
# -------------------------------------------------------
train_mean, train_std = compute_mean_std(train_set)
val_mean, val_std = compute_mean_std(val_set)

print("\n===== RESULTS =====")
print(f"Train Mean: {train_mean}")
print(f"Train Std : {train_std}")
print()
print(f"Val Mean  : {val_mean}")
print(f"Val Std   : {val_std}")


"""
===== RESULTS =====
Train Mean: tensor([0.4802, 0.4481, 0.3975])
Train Std : tensor([0.2296, 0.2263, 0.2255])

Val Mean  : tensor([0.4824, 0.4495, 0.3981])
Val Std   : tensor([0.2295, 0.2261, 0.2255])
"""
