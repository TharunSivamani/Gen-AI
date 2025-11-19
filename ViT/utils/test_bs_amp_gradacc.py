import torch
import time
import traceback
import os
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
    """Load and preprocess Tiny ImageNet dataset."""
    print("📦 Loading Tiny ImageNet dataset...")

    try:
        tiny_imagenet = load_dataset(dataset_path)
    except Exception as e:
        raise RuntimeError(f"❌ Failed to load dataset: {e}")

    transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    def transform_train(example):
        if isinstance(example["image"], list):
            example["image"] = [transform(img) for img in example["image"]]
        else:
            example["image"] = transform(example["image"])
        return example

    tiny_imagenet["train"] = tiny_imagenet["train"].with_transform(transform_train)
    print("✅ Tiny ImageNet ready for training.")
    return tiny_imagenet["train"]


# ============================================================
# ⚙️ Model Management
# ============================================================
def get_model_variants():
    """Return model constructors (not instances)."""
    return {
        "ViTBase": ViTBase,
        "ViTBaseRoPE": ViTBaseRoPE,
        "ViTGQA": ViTGQA,
        "ViTMHLA": ViTMHLA,
    }


# ============================================================
# 🚀 AMP + GradAcc Batch Size Benchmark
# ============================================================
def run_amp_gradacc_benchmark(
    model_fn,
    dataset,
    model_name,
    batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
    grad_acc_steps_list=[1, 2, 4, 8, 16, 32, 64],
    num_classes=200,
    num_workers=16,
    device=None,
    results_path="/home/cloudlyte/tharun/ViT/README_results.md",
):
    """
    Sweeps batch sizes and gradient accumulation settings with AMP enabled.
    Logs successful runs with timing and memory metrics.
    """

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    criterion = torch.nn.CrossEntropyLoss()

    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write(
                "# 🧠 Batch Size Benchmark Results (AMP + Gradient Accumulation)\n\n"
            )
            f.write(
                "| Model | Batch Size | Grad Acc | Eff Batch | Avg Time/Step (ms) | GPU Peak (MB) |\n"
            )
            f.write(
                "|--------|-------------|-----------|-------------|---------------------|----------------|\n"
            )

    print(f"\n🚀 Starting AMP + GradAcc benchmark for {model_name}")

    for bs in batch_sizes:
        for grad_acc in grad_acc_steps_list:
            eff_bs = bs * grad_acc
            print(f"\n🧩 Testing: Batch Size={bs}, GradAcc={grad_acc} (Eff={eff_bs})")

            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

            try:
                model = model_fn(num_classes=num_classes).to(device)
                optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
                scaler = torch.cuda.amp.GradScaler()

                loader = DataLoader(
                    dataset,
                    batch_size=bs,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

                model.train(True)
                total_time = 0.0
                steps = 0
                max_steps = grad_acc  # test small loop for fairness

                for step, batch in enumerate(loader):
                    if step >= max_steps:
                        break

                    images = batch["image"].to(device, non_blocking=True)
                    labels = batch["label"].to(device, non_blocking=True)

                    torch.cuda.synchronize()
                    start_time = time.time()

                    with torch.cuda.amp.autocast():
                        outputs = model(images)
                        loss = criterion(outputs, labels) / grad_acc

                    scaler.scale(loss).backward()

                    if (step + 1) % grad_acc == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad(set_to_none=True)

                    torch.cuda.synchronize()
                    elapsed = time.time() - start_time
                    total_time += elapsed
                    steps += 1

                    print(
                        f"✅ Step {step+1}/{grad_acc} | Time: {elapsed*1000:.2f} ms | Loss: {loss.item():.4f}"
                    )

                # GPU metrics
                mem_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
                avg_time = (total_time / steps) * 1000

                print(
                    f"🏁 DONE: BS={bs} | GradAcc={grad_acc} | AvgTime={avg_time:.2f} ms | PeakMem={mem_peak:.1f} MB"
                )

                # log to README
                with open(results_path, "a") as f:
                    f.write(
                        f"| {model_name} | {bs} | {grad_acc} | {eff_bs} | {avg_time:.2f} | {mem_peak:.1f} |\n"
                    )

                del model, optimizer, scaler
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"💥 OOM at Batch Size={bs}, GradAcc={grad_acc}")
                    torch.cuda.empty_cache()
                else:
                    print(f"❌ Error: {e}")
                    traceback.print_exc()
                    torch.cuda.empty_cache()
                continue

    print(f"\n✅ Finished model: {model_name}")
    print(f"🧾 Results appended to: {results_path}")


# ============================================================
# 🧠 Main Runner
# ============================================================
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"🔥 Using device: {device}")

    dataset = load_tiny_imagenet()
    model_variants = get_model_variants()

    for name, model_class in model_variants.items():
        print(f"\n{'='*80}")
        print(f"🚀 Testing model: {name}")
        print(f"{'='*80}")

        run_amp_gradacc_benchmark(
            model_fn=model_class,
            dataset=dataset,
            model_name=name,
            batch_sizes=[1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024],
            grad_acc_steps_list=[1, 2, 4, 8, 16, 32, 64],
            device=device,
        )


# ============================================================
# 🏁 Entry Point
# ============================================================
if __name__ == "__main__":
    main()
