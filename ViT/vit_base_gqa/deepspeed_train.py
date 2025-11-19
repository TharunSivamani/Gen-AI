import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
import deepspeed
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from datetime import datetime

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ============================================================
# 🧩 Import ViT Model
# ============================================================
from vit_base_gqa import ViTGQA


# ============================================================
# ⚙️ Config
# ============================================================
DATA_PATH = "/home/cloudlyte/tharun/tiny-imagenet"
BATCH_SIZE = 1024  # per-step micro batch
EPOCHS = 10
LR = 3e-4
NUM_CLASSES = 200
NUM_WORKERS = 8
SAVE_DIR = "./checkpoints_vit"
RESUME_DIR = None  # e.g. "./checkpoints_vit/epoch_5_end"
DS_CONFIG = "ds_config.json"

os.makedirs(SAVE_DIR, exist_ok=True)

device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")
print(f"🔧 Running on: {device}")


# ============================================================
# 📦 Transforms
# ============================================================
def ensure_rgb(img):
    return img.convert("RGB") if isinstance(img, Image.Image) else img


train_transform = transforms.Compose(
    [
        transforms.Lambda(ensure_rgb),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5] * 3, std=[0.5] * 3),
    ]
)


# ============================================================
# 📦 Transform function for dataset
# ============================================================
def transform_batch(example):
    if isinstance(example["image"], list):
        example["image"] = [train_transform(img) for img in example["image"]]
    else:
        example["image"] = train_transform(example["image"])
    return example


# ============================================================
# 📦 Load Tiny ImageNet
# ============================================================
def load_tiny_imagenet(dataset_path=DATA_PATH):
    print("📦 Loading Tiny ImageNet dataset from:", dataset_path)
    dataset = load_dataset(dataset_path)

    dataset["train"] = dataset["train"].with_transform(transform_batch)
    dataset["valid"] = dataset["valid"].with_transform(transform_batch)

    print("✅ Tiny ImageNet ready.")
    return dataset["train"], dataset["valid"]


# ============================================================
# 💾 Save / Load Checkpoints
# ============================================================
def save_checkpoint(folder, model_engine, meta):
    os.makedirs(folder, exist_ok=True)

    # unwrap model
    model = model_engine.module

    torch.save(model.state_dict(), os.path.join(folder, "model.pth"))

    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"💾 Saved checkpoint → {folder}")


def load_checkpoint(folder, model_engine):
    model = model_engine.module
    print(f"🔄 Loading checkpoint from: {folder}")
    model.load_state_dict(torch.load(os.path.join(folder, "model.pth")))

    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)

    print(f"🔁 Resuming from epoch {meta['epoch']} step {meta['step']}")
    return meta


# ============================================================
# 🧪 Evaluation
# ============================================================
@torch.no_grad()
def evaluate(model_engine, dataloader, criterion):
    model_engine.eval()
    model = model_engine.module

    total_loss, correct, total = 0.0, 0, 0
    pbar = tqdm(dataloader, desc="Eval", leave=False)

    for batch in pbar:
        images = batch["image"].to(model_engine.local_rank)
        labels = batch["label"].to(model_engine.local_rank)

        outputs = model(images)
        loss = criterion(outputs, labels)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()

        total_loss += loss.item() * images.size(0)
        total += labels.size(0)

    return total_loss / total, correct / total


# ============================================================
# 🚀 Main
# ============================================================
def main():

    train_dataset, val_dataset = load_tiny_imagenet()

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = ViTGQA(num_classes=NUM_CLASSES)

    parameters = filter(lambda p: p.requires_grad, model.parameters())
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model, model_parameters=parameters, config=DS_CONFIG
    )

    criterion = nn.CrossEntropyLoss()
    steps_per_epoch = len(train_loader)

    # ---------------- Evaluation before training ----------------
    print("\n🔎 Running evaluation BEFORE training (epoch 0)...")
    val_loss, val_acc = evaluate(model_engine, val_loader, criterion)
    print(f"📊 Epoch 0 Eval — Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%\n")

    # ---------------- Resume logic ----------------
    start_epoch = 1
    global_step = 0

    if RESUME_DIR:
        meta = load_checkpoint(RESUME_DIR, model_engine)
        start_epoch = meta["epoch"]
        global_step = meta["step"]

    # ---------------- Training loop ----------------
    for epoch in range(start_epoch, EPOCHS + 1):
        model_engine.train()
        running_loss = 0
        total = 0
        correct = 0

        print(f"\n🚀 Starting Epoch {epoch}/{EPOCHS}")

        for step, batch in enumerate(train_loader):
            global_step += 1

            images = batch["image"].to(model_engine.local_rank)
            labels = batch["label"].to(model_engine.local_rank)

            outputs = model_engine(images)
            loss = criterion(outputs, labels)

            model_engine.backward(loss)
            model_engine.step()

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            running_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            if step % 25 == 0:
                avg_loss = running_loss / total
                acc = 100 * correct / total
                print(
                    f"Step {step}/{steps_per_epoch} | "
                    f"Loss={avg_loss:.4f} | Acc={acc:.2f}%"
                )

        # ---------------- End of epoch evaluation ----------------
        print(f"\n🧪 End-of-Epoch Evaluation (epoch {epoch})...")
        val_loss, val_acc = evaluate(model_engine, val_loader, criterion)
        print(f"✅ Epoch {epoch} Eval — Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

        # ---------------- Save checkpoint ----------------
        ckpt_folder = os.path.join(SAVE_DIR, f"epoch_{epoch}_end")
        meta = {
            "epoch": epoch,
            "step": global_step,
            "val_loss": val_loss,
            "val_acc": val_acc * 100,
            "timestamp": datetime.now().isoformat(),
        }
        save_checkpoint(ckpt_folder, model_engine, meta)


if __name__ == "__main__":
    main()
