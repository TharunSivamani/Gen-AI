import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm

# ============================================================
# 🧩 Import your ViT model
# ============================================================
from vit_base.vit_base_patch16_224 import ViTBase  # adjust as per your repo

# ============================================================
# ⚙️ Config
# ============================================================
DATA_PATH = "/home/cloudlyte/tharun/tiny-imagenet"
BATCH_SIZE = 1024
GRAD_ACC = 2
EPOCHS = 1
LR = 3e-4
NUM_CLASSES = 200
NUM_WORKERS = 16
SAVE_DIR = "./checkpoints_vit"
os.makedirs(SAVE_DIR, exist_ok=True)

# ============================================================
# ⚙️ Device Config
# ============================================================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# ============================================================
# 📦 Load Tiny ImageNet using Hugging Face Datasets
# ============================================================
def load_tiny_imagenet(dataset_path=DATA_PATH):
    print("📦 Loading Tiny ImageNet dataset from:", dataset_path)
    dataset = load_dataset(dataset_path)

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    )

    def transform_batch(example):
        if isinstance(example["image"], list):
            example["image"] = [transform(img) for img in example["image"]]
        else:
            example["image"] = transform(example["image"])
        return example

    dataset["train"] = dataset["train"].with_transform(transform_batch)
    dataset["valid"] = dataset["valid"].with_transform(transform_batch)
    print("✅ Tiny ImageNet ready.")
    return dataset["train"], dataset["valid"]


# ============================================================
# 🧠 Model, Optimizer, Loss
# ============================================================
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

model = ViTBase(
    image_size=224,
    patch_size=16,
    embedding_dim=768,
    dropout=0.1,
    in_channels=3,
    num_heads=12,
    mlp_size=3072,
    attn_dropout=0.1,
    num_transformer_layers=8,
    num_classes=NUM_CLASSES,
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=LR, weight_decay=0.05)

# Compile for speed
if hasattr(torch, "compile"):
    model = torch.compile(model, mode="reduce-overhead")
    print("🚀 Using torch.compile() for optimized execution!")

scaler = torch.cuda.amp.GradScaler()


# ============================================================
# 🧩 Train and Evaluate
# ============================================================
def evaluate(model, dataloader, step_label="Eval"):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"{step_label}", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            with torch.cuda.amp.autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total_loss += loss.item() * images.size(0)
            total += labels.size(0)
            acc = 100.0 * correct / total
            loop.set_postfix(loss=loss.item(), acc=f"{acc:.2f}%")

    return total_loss / total, correct / total


def train_one_epoch(model, loader, optimizer, epoch, total_epochs):
    model.train()
    total_steps = len(loader)
    print(f"Total Number of Steps: {total_steps}")
    running_loss, correct, total = 0.0, 0, 0

    best_checkpoints = []  # store (acc, path)
    step_times = []

    for step, batch in enumerate(loader):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"].to(device, non_blocking=True)

        start_time = time.time()
        with torch.cuda.amp.autocast():
            outputs = model(images)
            loss = criterion(outputs, labels) / GRAD_ACC

        scaler.scale(loss).backward()

        if (step + 1) % GRAD_ACC == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad(set_to_none=True)

        torch.cuda.synchronize()
        elapsed = time.time() - start_time
        step_times.append(elapsed)

        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        running_loss += loss.item() * images.size(0) * GRAD_ACC  # scale back
        total += labels.size(0)

        avg_loss = running_loss / total
        acc = 100.0 * correct / total
        throughput = (images.size(0) * GRAD_ACC) / elapsed  # images per sec

        tqdm.write(
            f"[Epoch {epoch}/{total_epochs}] Step {step+1}/{total_steps} | "
            f"Loss={avg_loss:.4f} | Acc={acc:.2f}% | Time={elapsed*1000:.2f}ms | "
            f"Throughput={throughput:.2f} img/s"
        )

        # Evaluate mid-way and at end
        if step in [total_steps // 2, total_steps - 1]:
            print(f"🔍 Evaluation at step {step}/{total_steps}")
            val_loss, val_acc = evaluate(
                model, val_loader, step_label=f"Eval @ step {step}"
            )

            ckpt_path = os.path.join(
                SAVE_DIR, f"checkpoint_step{step}_acc{val_acc*100:.2f}.pth"
            )
            torch.save(model.state_dict(), ckpt_path)
            best_checkpoints.append((val_acc, ckpt_path))
            print(f"💾 Saved checkpoint: {ckpt_path}")

            # Keep only best 2 checkpoints
            best_checkpoints = sorted(
                best_checkpoints, key=lambda x: x[0], reverse=True
            )[:2]
            for _, path in best_checkpoints[2:]:
                if os.path.exists(path):
                    os.remove(path)
                    print(f"🧹 Removed older checkpoint: {path}")

    avg_time = sum(step_times) / len(step_times)
    print(
        f"🕒 Avg Step Time: {avg_time*1000:.2f} ms | Avg Throughput: {(BATCH_SIZE/avg_time):.2f} img/s"
    )


# ============================================================
# 🏁 Training Loop
# ============================================================
for epoch in range(1, EPOCHS + 1):
    train_one_epoch(model, train_loader, optimizer, epoch, EPOCHS)

print("✅ Training complete!")
