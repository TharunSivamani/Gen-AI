import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
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

os.environ["CUDA_VISIBLE_DEVICES"] = "5"

# ============================================================
# 🧩 Import ViT Model
# ============================================================
from vit_base_gqa import ViTGQA  # your ViT file

# ============================================================
# ⚙️ Config
# ============================================================
DATA_PATH = "../../tiny-imagenet"
BATCH_SIZE = 512
GRAD_ACC = 4
EPOCHS = 100
LR = 3e-4
NUM_CLASSES = 200
NUM_WORKERS = 8
SAVE_DIR = "./checkpoints_vit"
RESUME_DIR = None  # set this to resume: e.g. "./checkpoints_vit/epoch_1_step_300"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# ⚙️ Device
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# ============================================================
# 📦 Transforms
# ============================================================
def ensure_rgb(img):
    return img.convert("RGB") if isinstance(img, Image.Image) else img


mean = [0.4802, 0.4481, 0.3975]
std = [0.2296, 0.2263, 0.2255]

train_transform = transforms.Compose(
    [
        transforms.Lambda(ensure_rgb),
        transforms.RandomResizedCrop(224, scale=(0.6, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
        transforms.RandomGrayscale(p=0.1),
        transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Lambda(ensure_rgb),
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)


# ============================================================
# 📦 Load Tiny ImageNet
# ============================================================
def load_tiny_imagenet(dataset_path=DATA_PATH):
    print("📦 Loading Tiny ImageNet dataset from:", dataset_path)
    dataset = load_dataset(dataset_path)

    def transform_batch(example):
        if isinstance(example["image"], list):
            example["image"] = [train_transform(img) for img in example["image"]]
        else:
            example["image"] = train_transform(example["image"])
        return example

    def transform_val_batch(example):
        if isinstance(example["image"], list):
            example["image"] = [val_transform(img) for img in example["image"]]
        else:
            example["image"] = val_transform(example["image"])
        return example

    dataset["train"] = dataset["train"].with_transform(transform_batch)
    dataset["valid"] = dataset["valid"].with_transform(transform_val_batch)

    print("✅ Tiny ImageNet ready.")
    return dataset["train"], dataset["valid"]


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

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

# ============================================================
# 🧠 Model / Loss / Optimizer / Scheduler / AMP
# ============================================================
model = ViTGQA(
    num_classes=NUM_CLASSES,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)
steps_per_epoch = len(train_loader)

scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer,
    max_lr=LR,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    pct_start=0.3,
    anneal_strategy="cos",
)

scaler = torch.cuda.amp.GradScaler()


# ============================================================
# 💾 Checkpoint save/load utilities
# ============================================================
def save_checkpoint(folder, model, optimizer, scheduler, scaler, meta):
    os.makedirs(folder, exist_ok=True)

    torch.save(model.state_dict(), os.path.join(folder, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(folder, "optimizer.pth"))
    torch.save(scheduler.state_dict(), os.path.join(folder, "scheduler.pth"))
    torch.save(scaler.state_dict(), os.path.join(folder, "scaler.pth"))

    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"💾 Saved checkpoint → {folder}")


def load_checkpoint(folder, model, optimizer, scheduler, scaler):
    print(f"🔄 Loading checkpoint from: {folder}")

    model.load_state_dict(
        torch.load(os.path.join(folder, "model.pth"), map_location=device)
    )
    optimizer.load_state_dict(
        torch.load(os.path.join(folder, "optimizer.pth"), map_location=device)
    )
    scheduler.load_state_dict(
        torch.load(os.path.join(folder, "scheduler.pth"), map_location=device)
    )
    scaler.load_state_dict(
        torch.load(os.path.join(folder, "scaler.pth"), map_location=device)
    )

    with open(os.path.join(folder, "meta.json")) as f:
        meta = json.load(f)

    print(f"🔁 Resuming from epoch {meta['epoch']}, step {meta['step']}")
    return meta


# ============================================================
# 🧪 Evaluation
# ============================================================
def evaluate(model, dataloader):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Eval", leave=False):
            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()

            total_loss += loss.item() * images.size(0)
            total += labels.size(0)

    return total_loss / total, correct / total


print_every = 25  # <-- print interval


# ============================================================
# 🏁 EPOCH 0 EVALUATION BEFORE TRAINING
# ============================================================
print("\n🔎 Running evaluation BEFORE training (epoch 0)...")
val_loss, val_acc = evaluate(model, val_loader)
print(f"📊 Epoch 0 Eval — Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

# No checkpoint saved at epoch 0 (pre-training)

# ============================================================
# 🏋️ TRAINING LOOP
# ============================================================
start_epoch = 1
start_step = 0

if RESUME_DIR:
    meta = load_checkpoint(RESUME_DIR, model, optimizer, scheduler, scaler)
    start_epoch = meta["epoch"]
    start_step = meta["step"]

prev_epoch_time = time.time()  # For epoch timing

for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    running_loss = 0
    correct = 0
    total = 0

    print(f"\n🚀 Starting Epoch {epoch}/{EPOCHS}")

    for step, batch in enumerate(train_loader):
        if epoch == start_epoch and step < start_step:
            continue

        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)
        start_time = time.time()

        with torch.cuda.amp.autocast(dtype=torch.bfloat16):
            outputs = model(images)
            loss = criterion(outputs, labels) / GRAD_ACC

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACC == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)
            scheduler.step()

        # measure time
        torch.cuda.synchronize()
        elapsed = time.time() - start_time  # <-- full step time

        # metrics
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0) * GRAD_ACC
        total += labels.size(0)

        avg_loss = running_loss / total
        acc = 100 * correct / total
        lr = scheduler.get_last_lr()[0]

        # --------------------------------------------------------
        # 🔵 PRINT EVERY N STEPS (WITH TIME + THROUGHPUT)
        # --------------------------------------------------------
        if step % print_every == 0:
            step_time_ms = elapsed * 1000
            throughput = images.size(0) / elapsed

            print(
                f"[Epoch {epoch}] Step {step}/{steps_per_epoch} | "
                f"Loss={avg_loss:.4f} | Acc={acc:.2f}% | LR={lr:.6f} | "
                f"{step_time_ms:.1f} ms/step | {throughput:.1f} img/s"
            )

    # ============================================================
    # 🧪 END-OF-EPOCH EVALUATION
    # ============================================================
    epoch_time = time.time()
    print(f"\n🧪 End-of-Epoch Evaluation (epoch {epoch})...")
    val_loss, val_acc = evaluate(model, val_loader)
    print(f"✅ Epoch {epoch} Eval — Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    # Print time between epochs
    elapsed_epoch = epoch_time - prev_epoch_time
    print(f"⏱️ Time taken for epoch {epoch}: {elapsed_epoch:.2f} seconds")
    prev_epoch_time = time.time()

    # Save only at halfway and last epoch
    save_points = []
    halfway = EPOCHS // 2
    if halfway == 0:
        halfway = 1  # avoid zero division if EPOCHS==1
    save_points.append(halfway)
    save_points.append(EPOCHS)
    if epoch in save_points:
        ckpt_folder = os.path.join(SAVE_DIR, f"epoch_{epoch}_end")
        meta = {
            "epoch": epoch,
            "step": steps_per_epoch - 1,
            "train_loss": avg_loss,
            "train_acc": acc,
            "val_loss": val_loss,
            "val_acc": val_acc * 100,
            "lr": lr,
            "timestamp": datetime.now().isoformat(),
        }
        save_checkpoint(ckpt_folder, model, optimizer, scheduler, scaler, meta)

print("\n🎉 Training Complete!")
