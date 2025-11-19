import torch
import time
import traceback
import os
from torch.utils.data import DataLoader
from torchvision import transforms
from datasets import load_dataset
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import fcntl

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
    """Standalone picklable function."""
    if isinstance(example["image"], list):
        example["image"] = [train_transform(img) for img in example["image"]]
    else:
        example["image"] = train_transform(example["image"])
    return example


def load_tiny_imagenet(dataset_path="/home/cloudlyte/tharun/tiny-imagenet"):
    """Load and preprocess Tiny ImageNet dataset."""
    print("📦 Loading Tiny ImageNet dataset...")

    tiny_imagenet = load_dataset(dataset_path)
    tiny_imagenet["train"] = tiny_imagenet["train"].with_transform(transform_train)

    print("✅ Tiny ImageNet ready for training.")
    return tiny_imagenet["train"]


# ============================================================
# ⚙️ Model Management
# ============================================================
def get_model_variants():
    """Return model constructors."""
    return {
        "ViTBase": ViTBase,
        "ViTBaseRoPE": ViTBaseRoPE,
        "ViTGQA": ViTGQA,
        "ViTMHLA": ViTMHLA,
    }


# ============================================================
# 🧮 File-safe Logger
# ============================================================
def safe_log(results_path, text):
    """Thread-safe/Process-safe write to README using file lock."""
    with open(results_path, "a") as f:
        fcntl.flock(f, fcntl.LOCK_EX)
        f.write(text)
        fcntl.flock(f, fcntl.LOCK_UN)


# ============================================================
# 🚀 AMP + GradAcc Benchmark (Single GPU)
# ============================================================
def run_amp_gradacc_benchmark(
    model_name,
    model_fn,
    dataset_path,
    device_id,
    results_path,
    batch_sizes,
    grad_acc_steps_list,
    num_classes=200,
    num_workers=0,
):
    """Runs all batch-size × grad-acc tests for one model on a given GPU."""
    torch.cuda.set_device(device_id)
    device = torch.device(f"cuda:{device_id}")

    dataset = load_tiny_imagenet(dataset_path)
    criterion = torch.nn.CrossEntropyLoss()

    safe_log(results_path, f"\n\n## 🧠 {model_name} on GPU:{device_id}\n")
    safe_log(
        results_path,
        "| Batch Size | Grad Acc | Eff Batch | Avg Time (ms) | Peak Mem (MB) |\n",
    )
    safe_log(
        results_path,
        "|-------------|-----------|-------------|----------------|----------------|\n",
    )

    for bs in batch_sizes:
        for grad_acc in grad_acc_steps_list:
            eff_bs = bs * grad_acc
            print(
                f"[GPU {device_id}] {model_name} → BS={bs}, GradAcc={grad_acc} (Eff={eff_bs})"
            )

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
                max_steps = grad_acc

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
                        f"[GPU {device_id}] Step {step+1}/{grad_acc} | Time: {elapsed*1000:.2f} ms | Loss: {loss.item():.4f}"
                    )

                mem_peak = torch.cuda.max_memory_allocated(device) / (1024**2)
                avg_time = (total_time / steps) * 1000

                safe_log(
                    results_path,
                    f"| {bs} | {grad_acc} | {eff_bs} | {avg_time:.2f} | {mem_peak:.1f} |\n",
                )

                del model, optimizer, scaler
                torch.cuda.empty_cache()

            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"[GPU {device_id}] 💥 OOM at BS={bs}, GradAcc={grad_acc}")
                    torch.cuda.empty_cache()
                else:
                    print(f"[GPU {device_id}] ❌ RuntimeError: {e}")
                    traceback.print_exc()
                continue

    print(f"✅ {model_name} completed on GPU:{device_id}")


# ============================================================
# 🧠 Multi-GPU Coordinator
# ============================================================
def main():
    dataset_path = "/home/cloudlyte/tharun/tiny-imagenet"
    results_path = "/home/cloudlyte/tharun/ViT/README_results.md"
    device_ids = [0, 2, 4, 5]  # GPUs available

    if not os.path.exists(results_path):
        with open(results_path, "w") as f:
            f.write("# 🧠 Multi-GPU AMP + GradAcc Benchmark Results\n\n")

    model_variants = get_model_variants()
    batch_sizes = [128, 256, 512, 1024]
    grad_acc_steps_list = [1, 2, 4, 8, 16, 32, 64]

    print(f"🔥 Launching benchmark across GPUs: {device_ids}\n")

    # Launch each model on its own GPU
    with ProcessPoolExecutor(max_workers=len(device_ids)) as executor:
        futures = []
        for i, (name, model_class) in enumerate(model_variants.items()):
            gpu_id = device_ids[i % len(device_ids)]
            futures.append(
                executor.submit(
                    run_amp_gradacc_benchmark,
                    name,
                    model_class,
                    dataset_path,
                    gpu_id,
                    results_path,
                    batch_sizes,
                    grad_acc_steps_list,
                )
            )

        for f in futures:
            f.result()  # wait for all to finish

    print("\n🏁 All models completed! Results saved to:")
    print(results_path)


# ============================================================
# 🏁 Entry Point
# ============================================================
if __name__ == "__main__":
    multiprocessing.set_start_method("spawn", force=True)
    main()
