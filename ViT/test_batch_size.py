import torch
from datasets import load_dataset
from torchvision import transforms

# Import your models
from vit_base.vit_base_patch16_224 import ViTBase
from vit_base_rope.vit_base_rope import ViTBaseRoPE
from vit_base_gqa.vit_base_gqa import ViTGQA
from vit_base_mhla.vit_base_mhla import ViTMHLA

import torch
import time
import traceback
from torch.utils.data import DataLoader


train_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]
)

tiny_imagenet = load_dataset("Maysee/tiny-imagenet")


def transform_train(example):
    example["image"] = train_transform(example["image"])
    return example


tiny_imagenet["train"] = tiny_imagenet["train"].with_transform(transform_train)


def find_max_batch_size_realdata(
    model,
    dataset,
    num_classes,
    start_bs=1,
    max_bs=4096,
    num_workers=4,
    device=None,
    inference_only=False,
    warmup_steps=2,
):
    """
    Gradually increases batch size on a real dataset until an Out-of-Memory (OOM) occurs.

    Returns:
        (best_batch_size, avg_time_per_step)
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device)
    model.train(not inference_only)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    best_batch_size = start_bs
    best_time = None

    print(
        f"🚀 Starting batch size search on {device} (inference_only={inference_only})...\n"
    )

    batch_size = start_bs
    oom_hit = False

    while batch_size <= max_bs:
        try:
            # Build a small dataloader (1 batch per size test)
            loader = DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            batch = next(iter(loader))
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # Warm-up to stabilize timing
            for _ in range(warmup_steps):
                with torch.no_grad():
                    _ = model(images)

            torch.cuda.synchronize()
            start_time = time.time()

            # Forward + backward (if training)
            if inference_only:
                with torch.no_grad():
                    outputs = model(images)
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()

            torch.cuda.synchronize()
            elapsed = time.time() - start_time
            mem_used = torch.cuda.max_memory_allocated(device) / (1024**2)

            print(
                f"✅ Batch Size: {batch_size:<4d} | Time/step: {elapsed*1000:.2f} ms | GPU Mem: {mem_used:.1f} MB"
            )

            best_batch_size = batch_size
            best_time = elapsed
            batch_size *= 2  # try next power of 2

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"💥 OOM at batch size {batch_size}! Stopping test.")
                torch.cuda.empty_cache()
                oom_hit = True
                break
            else:
                print(f"⚠️ RuntimeError at batch size {batch_size}: {e}")
                traceback.print_exc()
                break
        except StopIteration:
            print("⚠️ Dataset exhausted unexpectedly. Exiting.")
            break

    if not oom_hit:
        print("⚠️ Reached max batch size without OOM.")

    print("\n🏁 Finished search!")
    print(f"✅ Best Batch Size: {best_batch_size}")
    if best_time:
        print(f"⏱ Estimated Time per Step: {best_time*1000:.2f} ms")

    return best_batch_size, best_time


def load_tiny_imagenet():
    """Load and transform Tiny ImageNet dataset once."""
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    print("📦 Loading Tiny ImageNet...")
    tiny_imagenet = load_dataset("Maysee/tiny-imagenet")

    def transform_train(example):
        example["image"] = train_transform(example["image"])
        return example

    tiny_imagenet["train"] = tiny_imagenet["train"].with_transform(transform_train)
    print("✅ Tiny ImageNet ready.")
    return tiny_imagenet["train"]


def get_model_variants(num_classes=200):
    """Return dict of model name -> model instance."""
    models = {
        "ViTBase": ViTBase(
            num_classes=num_classes,
        ),
        "ViTBaseRoPE": ViTBaseRoPE(
            num_classes=num_classes,
        ),
        "ViTGQA": ViTGQA(
            num_classes=num_classes,
        ),
        "ViTMHLA": ViTMHLA(
            num_classes=num_classes,
        ),
    }
    return models


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    dataset = load_tiny_imagenet()
    models = get_model_variants(num_classes=200)

    results = []  # store (model_name, best_batch_size, time_per_step)

    for name, model in models.items():
        print(f"\n{'=' * 80}")
        print(f"🚀 Testing model: {name}")
        print(f"{'=' * 80}")

        try:
            best_bs, best_time = find_max_batch_size_realdata(
                model=model,
                dataset=dataset,
                num_classes=200,
                start_bs=1,
                max_bs=1024,
            )
            results.append((name, best_bs, best_time))
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            results.append((name, "FAILED", None))

    # -----------------------------
    # Print Summary
    # -----------------------------
    print("\n\n🏁 SUMMARY REPORT")
    print("-" * 80)
    print(f"{'Model':<20}{'Best Batch Size':<20}{'Time/Step (ms)':<20}")
    print("-" * 80)
    for name, bs, t in results:
        t_str = f"{t*1000:.2f}" if t else "-"
        print(f"{name:<20}{str(bs):<20}{t_str:<20}")
    print("-" * 80)


if __name__ == "__main__":
    main()
