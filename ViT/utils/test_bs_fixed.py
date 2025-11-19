import torch
import time
import traceback
import csv
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset

# ============================================================
# 🧩 Import your models
# ============================================================
from vit_base.vit_base_patch16_224 import ViTBase
from vit_base_rope.vit_base_rope import ViTBaseRoPE
from vit_base_gqa.vit_base_gqa import ViTGQA
from vit_base_mhla.vit_base_mhla import ViTMHLA


# ============================================================
# 🧩 Data Loading and Transformation
# ============================================================


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


# ============================================================
# ⚙️ Model Management
# ============================================================


def get_model_variants():
    """
    Return model constructors (not instances).
    This allows re-initialization per batch-size test.
    """
    return {
        "ViTBase": ViTBase,
        "ViTBaseRoPE": ViTBaseRoPE,
        "ViTGQA": ViTGQA,
        "ViTMHLA": ViTMHLA,
    }


# ============================================================
# 🚀 Batch Size Finder
# ============================================================


def find_max_batch_size_realdata(
    model_fn,
    dataset,
    num_classes=200,
    start_bs=1,
    max_bs=4096,
    num_workers=4,
    device=None,
    inference_only=False,
    warmup_steps=2,
    csv_writer=None,
    model_name=None,
):
    """
    Gradually increases batch size using a real dataset until OOM occurs.
    Reinitializes model & optimizer at each step to ensure fresh GPU memory.
    Saves metrics to CSV if csv_writer is provided.
    """

    if device is None:
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    print(
        f"🚀 Starting batch size search on {device} (inference_only={inference_only})...\n"
    )

    criterion = torch.nn.CrossEntropyLoss()
    best_batch_size = start_bs
    best_time = None
    oom_hit = False
    batch_size = start_bs

    while batch_size <= max_bs:
        try:
            print(f"\n🧩 Testing Batch Size = {batch_size}")
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            # --- Fresh model & optimizer each time ---
            model = model_fn(num_classes=num_classes).to(device)
            model.train(not inference_only)
            optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

            # --- DataLoader ---
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

            # --- Warm-up ---
            for _ in range(warmup_steps):
                with torch.no_grad():
                    _ = model(images)

            torch.cuda.synchronize()
            start_time = time.time()

            # --- Forward/backward pass ---
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

            # --- GPU memory stats ---
            mem_alloc = torch.cuda.memory_allocated(device) / (1024**2)
            mem_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
            param_size = sum(p.numel() for p in model.parameters()) * 4 / (1024**2)
            opt_state = (
                sum(p.numel() for g in optimizer.param_groups for p in g["params"])
                * 4
                / (1024**2)
            )

            print(f"✅ Batch Size: {batch_size:<4d} | Time/step: {elapsed*1000:.2f} ms")
            print(f"   GPU Mem (alloc): {mem_alloc:.1f} MB | Peak: {mem_peak:.1f} MB")
            print(
                f"   Model Params: {param_size:.1f} MB | Optimizer State: {opt_state:.1f} MB"
            )

            # --- Write CSV row ---
            if csv_writer:
                csv_writer.writerow(
                    [
                        model_name,
                        batch_size,
                        f"{elapsed*1000:.2f}",
                        f"{mem_alloc:.1f}",
                        f"{mem_peak:.1f}",
                        f"{param_size:.1f}",
                        f"{opt_state:.1f}",
                    ]
                )

            best_batch_size = batch_size
            best_time = elapsed
            batch_size *= 2

            # Cleanup
            del model, optimizer, images, labels, outputs, loss
            torch.cuda.empty_cache()

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


# ============================================================
# 🧠 Main Experiment Runner
# ============================================================


def main():
    """
    Main entry point — loads data, tests each ViT variant, and reports results.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    dataset = load_tiny_imagenet()
    model_variants = get_model_variants()
    results = []

    csv_path = "/home/cloudlyte/tharun/batchsize_results.csv"
    with open(csv_path, "w", newline="") as csvfile:
        csv_writer = csv.writer(csvfile)
        csv_writer.writerow(
            [
                "Model",
                "Batch Size",
                "Time/Step (ms)",
                "GPU Mem Alloc (MB)",
                "GPU Mem Peak (MB)",
                "Model Params (MB)",
                "Optimizer (MB)",
            ]
        )

        for name, model_class in model_variants.items():
            print(f"\n{'=' * 80}")
            print(f"🚀 Testing model: {name}")
            print(f"{'=' * 80}")

            try:
                best_bs, best_time = find_max_batch_size_realdata(
                    model_fn=model_class,
                    dataset=dataset,
                    num_classes=200,
                    start_bs=1,
                    max_bs=4096,
                    device=device,
                    inference_only=False,
                    csv_writer=csv_writer,
                    model_name=name,
                )
                results.append((name, best_bs, best_time))
            except Exception as e:
                print(f"❌ Error testing {name}: {e}")
                traceback.print_exc()
                results.append((name, "FAILED", None))

    # --- Summary ---
    print("\n\n🏁 SUMMARY REPORT")
    print("-" * 80)
    print(f"{'Model':<20}{'Best Batch Size':<20}{'Time/Step (ms)':<20}")
    print("-" * 80)
    for name, bs, t in results:
        t_str = f"{t*1000:.2f}" if t else "-"
        print(f"{name:<20}{str(bs):<20}{t_str:<20}")
    print("-" * 80)
    print(f"📁 Results saved to: {csv_path}")


# ============================================================
# 🏁 Entry Point
# ============================================================

if __name__ == "__main__":
    main()
