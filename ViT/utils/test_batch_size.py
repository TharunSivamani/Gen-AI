import torch
import time
import traceback
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# Import your models
from vit_base.vit_base_patch16_224 import ViTBase
from vit_base_rope.vit_base_rope import ViTBaseRoPE
from vit_base_gqa.vit_base_gqa import ViTGQA
from vit_base_mhla.vit_base_mhla import ViTMHLA


# ====================================================================================
# 🧩 Data Loading and Transformation
# ====================================================================================


def load_tiny_imagenet(dataset_path="/home/cloudlyte/tharun/tiny-imagenet"):
    """
    Load and preprocess Tiny ImageNet dataset for training.
    """
    print("📦 Loading Tiny ImageNet dataset...")

    try:
        tiny_imagenet = load_dataset(dataset_path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load dataset from {dataset_path}: {e}")

    # Define train transformations
    train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def transform_train(example):
        # Handle both single and batched samples
        if isinstance(example["image"], list):
            example["image"] = [train_transform(img) for img in example["image"]]
        else:
            example["image"] = train_transform(example["image"])
        return example

    # Apply transformation to train split
    tiny_imagenet["train"] = tiny_imagenet["train"].with_transform(transform_train)

    print("✅ Tiny ImageNet ready for training.")
    return tiny_imagenet["train"]


# ====================================================================================
# ⚙️ Model Management
# ====================================================================================


def get_model_variants(num_classes=200):
    """
    Return a dictionary of all model variants to test.
    """
    return {
        "ViTBase": ViTBase(num_classes=num_classes),
        "ViTBaseRoPE": ViTBaseRoPE(num_classes=num_classes),
        "ViTGQA": ViTGQA(num_classes=num_classes),
        "ViTMHLA": ViTMHLA(num_classes=num_classes),
    }


# ====================================================================================
# 🚀 Batch Size Finder
# ====================================================================================


def find_max_batch_size_realdata(
    model,
    dataset,
    num_classes=200,
    start_bs=1,
    max_bs=4096,
    num_workers=4,
    device=None,
    inference_only=False,
    warmup_steps=2,
):
    """
    Gradually increases batch size using a real dataset until OOM occurs.

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
    oom_hit = False

    print(
        f"🚀 Starting batch size search on {device} (inference_only={inference_only})...\n"
    )

    batch_size = start_bs
    while batch_size <= max_bs:
        try:
            # Create DataLoader (1 batch per test)
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

            # Reset memory stats before timing
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            # Warm-up runs (for timing stability)
            for _ in range(warmup_steps):
                with torch.no_grad():
                    _ = model(images)

            torch.cuda.synchronize()
            start_time = time.time()

            # Forward (and backward if training)
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

            # Measure GPU memory
            mem_used = (
                torch.cuda.max_memory_allocated(device) / (1024**2)
                if device.type == "cuda"
                else 0.0
            )

            print(
                f"✅ Batch Size: {batch_size:<4d} | Time/step: {elapsed*1000:.2f} ms | GPU Mem: {mem_used:.1f} MB"
            )

            best_batch_size = batch_size
            best_time = elapsed
            batch_size *= 2  # Test next power of two

        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                print(f"💥 OOM at batch size {batch_size}! Stopping test.")
                if device.type == "cuda":
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


# ====================================================================================
# 🧠 Main Experiment Runner
# ====================================================================================


def main():
    """
    Main entry point — loads data, tests each ViT variant, and reports results.
    """
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
                max_bs=4096,
                device=device,
                inference_only=False,
            )
            results.append((name, best_bs, best_time))
        except Exception as e:
            print(f"❌ Error testing {name}: {e}")
            traceback.print_exc()
            results.append((name, "FAILED", None))

    # -----------------------------
    # 📊 Print Summary
    # -----------------------------
    print("\n\n🏁 SUMMARY REPORT")
    print("-" * 80)
    print(f"{'Model':<20}{'Best Batch Size':<20}{'Time/Step (ms)':<20}")
    print("-" * 80)
    for name, bs, t in results:
        t_str = f"{t*1000:.2f}" if t else "-"
        print(f"{name:<20}{str(bs):<20}{t_str:<20}")
    print("-" * 80)


# ====================================================================================
# 🏁 Entry Point
# ====================================================================================

if __name__ == "__main__":
    main()
