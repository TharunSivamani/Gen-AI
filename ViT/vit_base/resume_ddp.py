import os
import json
import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
from torchvision import transforms
from datasets import load_dataset
from tqdm import tqdm
from PIL import Image
from datetime import datetime
import torch.multiprocessing as mp

import warnings

warnings.filterwarnings("ignore")

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# ============================================================
# 🧩 Import ViT Model
# ============================================================
from vit_base_patch16_224 import ViTBase  # your ViT file

# ============================================================
# ⚙️ Config
# ============================================================
DATA_PATH = "/home/cloudlyte/tharun/tiny-imagenet"
BATCH_SIZE = 1024  # Per GPU batch size
GRAD_ACC = 32
EPOCHS = 10
LR = 3e-4
NUM_CLASSES = 200
NUM_WORKERS = 8
SAVE_DIR = "./checkpoints_vit"
RESUME_DIR = "/home/cloudlyte/tharun/ViT/vit_base/checkpoints_vit/epoch_5_end"  # set this to resume: e.g. "./checkpoints_vit/epoch_1_step_300"
os.makedirs(SAVE_DIR, exist_ok=True)


# ============================================================
# 🔧 DDP Setup
# ============================================================
def ddp_setup(rank, world_size, master_port="12355"):
    """Initialize DDP process group."""
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = master_port
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


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
# 📦 Transform function (module-level for pickling)
# ============================================================
def transform_batch(example):
    """Transform function for dataset batches. Must be module-level for pickling."""
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
# 💾 Checkpoint save/load utilities
# ============================================================
def save_checkpoint(folder, model, optimizer, scheduler, scaler, meta, rank):
    """Save checkpoint (only on rank 0)."""
    if rank != 0:
        return

    os.makedirs(folder, exist_ok=True)

    # Save model state dict (unwrap DDP model)
    model_state = (
        model.module.state_dict() if isinstance(model, DDP) else model.state_dict()
    )
    torch.save(model_state, os.path.join(folder, "model.pth"))
    torch.save(optimizer.state_dict(), os.path.join(folder, "optimizer.pth"))
    torch.save(scheduler.state_dict(), os.path.join(folder, "scheduler.pth"))
    torch.save(scaler.state_dict(), os.path.join(folder, "scaler.pth"))

    with open(os.path.join(folder, "meta.json"), "w") as f:
        json.dump(meta, f, indent=4)

    print(f"💾 Saved checkpoint → {folder}")


def load_checkpoint(folder, model, optimizer, scheduler, scaler, rank):
    """Load checkpoint."""
    if rank != 0:
        # On non-rank-0, wait for rank 0 to load first, then load
        torch.distributed.barrier()

    device = torch.device(f"cuda:{rank}")

    # Load model state dict
    model_state = torch.load(os.path.join(folder, "model.pth"), map_location=device)
    if isinstance(model, DDP):
        model.module.load_state_dict(model_state)
    else:
        model.load_state_dict(model_state)

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

    if rank == 0:
        print(f"🔁 Resuming from epoch {meta['epoch']}, step {meta['step']}")

    if rank != 0:
        torch.distributed.barrier()

    return meta


# ============================================================
# 🧪 Evaluation
# ============================================================
def evaluate(model, dataloader, rank, world_size):
    """Evaluate model with DDP support."""
    model.eval()
    total_loss, correct, total = 0.0, 0, 0

    autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for batch in tqdm(
            dataloader, desc=f"Eval [Rank {rank}]", leave=False, disable=(rank != 0)
        ):
            images = batch["image"].to(rank, non_blocking=True)
            labels = batch["label"].to(rank, non_blocking=True)

            with torch.cuda.amp.autocast(dtype=autocast_dtype):
                outputs = model(images)
                loss = criterion(outputs, labels)

            preds = outputs.argmax(dim=1)
            batch_correct = (preds == labels).sum().item()
            batch_total = labels.size(0)
            batch_loss = loss.item() * batch_total

            # Accumulate local stats
            correct += batch_correct
            total += batch_total
            total_loss += batch_loss

    # Gather stats from all ranks
    stats = torch.tensor([total_loss, correct, total], dtype=torch.float64, device=rank)
    torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)

    total_loss_all = stats[0].item()
    correct_all = int(stats[1].item())
    total_all = int(stats[2].item())

    avg_loss = total_loss_all / total_all
    accuracy = correct_all / total_all

    return avg_loss, accuracy


print_every = 25  # <-- print interval
autocast_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


# ============================================================
# 🏋️ Main Training Function
# ============================================================
def main(rank, world_size):
    """Main training function for each process."""
    # Setup DDP
    ddp_setup(rank, world_size)

    device = torch.device(f"cuda:{rank}")

    if rank == 0:
        print(f"✅ Using device: {device} (Rank {rank}/{world_size})")
        print(f"🌍 World size: {world_size}")

    # Load datasets
    train_dataset, val_dataset = load_tiny_imagenet()

    if rank == 0:
        print(f"Train samples: {len(train_dataset)} | Val samples: {len(val_dataset)}")

    # Create distributed samplers
    train_sampler = DistributedSampler(
        train_dataset, num_replicas=world_size, rank=rank, shuffle=True, drop_last=False
    )
    val_sampler = DistributedSampler(
        val_dataset, num_replicas=world_size, rank=rank, shuffle=False, drop_last=False
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        sampler=train_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        sampler=val_sampler,
        num_workers=NUM_WORKERS,
        pin_memory=True,
        drop_last=False,
    )

    # ============================================================
    # 🧠 Model / Loss / Optimizer / Scheduler / AMP
    # ============================================================
    model = ViTBase(num_classes=NUM_CLASSES).to(device)
    model = DDP(model, device_ids=[rank], find_unused_parameters=False)

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
    # 🏁 EPOCH 0 EVALUATION BEFORE TRAINING
    # ============================================================
    if rank == 0:
        print("\n🔎 Running evaluation BEFORE training (epoch 0)...")
    val_loss, val_acc = evaluate(model, val_loader, rank, world_size)
    if rank == 0:
        print(f"📊 Epoch 0 Eval — Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%")

    # ============================================================
    # 🏋️ TRAINING LOOP
    # ============================================================
    start_epoch = 1
    start_step = 0

    if RESUME_DIR:
        meta = load_checkpoint(RESUME_DIR, model, optimizer, scheduler, scaler, rank)
        start_epoch = meta["epoch"]
        start_step = meta["step"]
        # Calculate global_step: (epochs_before * steps_per_epoch) + steps_in_current_epoch
        global_step = (start_epoch - 1) * steps_per_epoch + start_step
    else:
        global_step = 0

    prev_epoch_time = time.time()

    for epoch in range(start_epoch, EPOCHS + 1):
        # Set epoch for distributed sampler (important for shuffling)
        train_sampler.set_epoch(epoch)

        model.train()
        running_loss = 0
        correct = 0
        total = 0

        if rank == 0:
            print(f"\n🚀 Starting Epoch {epoch}/{EPOCHS}")

        for step, batch in enumerate(train_loader):
            if epoch == start_epoch and step < start_step:
                continue

            global_step += 1
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            start_time = time.time()

            with torch.cuda.amp.autocast(dtype=autocast_dtype):
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
            elapsed = time.time() - start_time

            # metrics
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            # Accumulate true loss (multiply by GRAD_ACC to undo the division for gradient accumulation)
            running_loss += loss.item() * images.size(0) * GRAD_ACC
            total += labels.size(0)

            avg_loss = running_loss / total
            acc = 100 * correct / total
            lr = scheduler.get_last_lr()[0]

            # --------------------------------------------------------
            # 🔵 PRINT EVERY N STEPS (WITH TIME + THROUGHPUT)
            # --------------------------------------------------------
            if rank == 0 and step % print_every == 0:
                step_time_ms = elapsed * 1000
                throughput = (
                    images.size(0) * world_size / elapsed
                )  # Total throughput across all GPUs

                print(
                    f"[Epoch {epoch}] Step {step}/{steps_per_epoch} | "
                    f"Loss={avg_loss:.4f} | Acc={acc:.2f}% | LR={lr:.6f} | "
                    f"{step_time_ms:.1f} ms/step | {throughput:.1f} img/s (total)"
                )

        # ============================================================
        # 🧪 END-OF-EPOCH EVALUATION
        # ============================================================
        epoch_time = time.time()
        if rank == 0:
            print(f"\n🧪 End-of-Epoch Evaluation (epoch {epoch})...")
        val_loss, val_acc = evaluate(model, val_loader, rank, world_size)
        if rank == 0:
            print(
                f"✅ Epoch {epoch} Eval — Loss: {val_loss:.4f}, Acc: {val_acc*100:.2f}%"
            )

            # Print time between epochs
            elapsed_epoch = epoch_time - prev_epoch_time
            print(f"⏱️ Time taken for epoch {epoch}: {elapsed_epoch:.2f} seconds")
        prev_epoch_time = time.time()

        # Save only at halfway and last epoch
        save_points = []
        halfway = EPOCHS // 2
        if halfway == 0:
            halfway = 1
        save_points.append(halfway)
        save_points.append(EPOCHS)
        if epoch in save_points:
            ckpt_folder = os.path.join(SAVE_DIR, f"epoch_{epoch}_end")
            meta = {
                "epoch": epoch,
                "step": global_step - 1,
                "train_loss": avg_loss,
                "train_acc": acc,
                "val_loss": val_loss,
                "val_acc": val_acc * 100,
                "lr": lr,
                "timestamp": datetime.now().isoformat(),
            }
            save_checkpoint(
                ckpt_folder, model, optimizer, scheduler, scaler, meta, rank
            )

    if rank == 0:
        print("\n🎉 Training Complete!")

    destroy_process_group()


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    if world_size == 0:
        raise RuntimeError("No CUDA devices available!")

    print(f"🚀 Starting DDP training on {world_size} GPU(s)")
    mp.spawn(main, args=(world_size,), nprocs=world_size, join=True)

# CUDA_VISIBLE_DEVICES=2,5 python3 resume_ddp.py
