import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm

# from vit_base import ViTBase  # ← your ViTBase definition file


# -----------------------------
# 1. Device Config
# -----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")


# -----------------------------
# 2. Dataset Setup (Tiny ImageNet)
# -----------------------------
# Expected folder structure:
# tiny-imagenet-200/
# ├── train/
# │   ├── n01443537/
# │   │   ├── images/
# │   │   └── ...
# ├── val/
# │   ├── images/
# │   ├── val_annotations.txt
# └── wnids.txt

# ⚠️ IMPORTANT:
# Download: https://tiny-imagenet.herokuapp.com/
# or https://www.kaggle.com/c/tiny-imagenet

DATA_DIR = "./tiny-imagenet-200"

# Image preprocessing
train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

# Create datasets and loaders
train_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "train"), transform=train_transform
)
val_dataset = datasets.ImageFolder(
    os.path.join(DATA_DIR, "val"), transform=val_transform
)

train_loader = DataLoader(
    train_dataset, batch_size=64, shuffle=True, num_workers=4, pin_memory=True
)
val_loader = DataLoader(
    val_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True
)

print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")


# -----------------------------
# 3. Model Setup
# -----------------------------
model = ViTBase(
    image_size=224,
    patch_size=16,
    embedding_dim=768,
    dropout=0.1,
    in_channels=3,
    num_heads=12,
    mlp_size=3072,
    attn_dropout=0.1,
    num_transformer_layers=8,  # 8 layers for TinyImageNet to reduce compute
    num_classes=200,  # Tiny ImageNet → 200 classes
).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.05)

# Optionally compile model (PyTorch 2.0+)
if hasattr(torch, "compile"):
    model = torch.compile(model)
    print("🚀 Using torch.compile() for faster training!")


# -----------------------------
# 4. Training Function
# -----------------------------
def train_one_epoch(model, dataloader, criterion, optimizer, device, epoch):
    model.train()
    running_loss, correct, total = 0.0, 0, 0

    loop = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)
        preds = outputs.argmax(dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    return running_loss / total, correct / total


# -----------------------------
# 5. Validation Function
# -----------------------------
def evaluate(model, dataloader, criterion, device, epoch):
    model.eval()
    running_loss, correct, total = 0.0, 0, 0

    with torch.no_grad():
        loop = tqdm(dataloader, desc=f"Epoch {epoch} [Val]", leave=False)
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100.0 * correct / total)

    return running_loss / total, correct / total


# -----------------------------
# 6. Main Training Loop
# -----------------------------
epochs = 20
best_val_acc = 0.0

for epoch in range(1, epochs + 1):
    train_loss, train_acc = train_one_epoch(
        model, train_loader, criterion, optimizer, device, epoch
    )
    val_loss, val_acc = evaluate(model, val_loader, criterion, device, epoch)

    print(f"\nEpoch [{epoch}/{epochs}]")
    print(f"Train: Loss={train_loss:.4f}, Acc={train_acc*100:.2f}%")
    print(f"Val:   Loss={val_loss:.4f}, Acc={val_acc*100:.2f}%")

    # Save checkpoint
    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), "vit_tinyimagenet_best.pth")
        print(f"💾 Saved best model (Val Acc: {val_acc*100:.2f}%)")

torch.save(model.state_dict(), "vit_tinyimagenet_final.pth")
print("✅ Training complete. Model saved!")
