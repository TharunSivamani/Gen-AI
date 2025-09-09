# train_with_deepspeed_h100.py
# Run with: deepspeed --num_gpus=8 train_with_deepspeed_h100.py

# nohup deepspeed --num_gpus=8 train_cosmopedia_v2.py > cosmopedia_v2_training.log 2>&1 &
# [1] 664896

import warnings

# Ignore all warnings
warnings.filterwarnings("ignore")
import os
import json
import torch
import torch.nn as nn
import deepspeed
from transformers import AutoTokenizer
import time

from gemma import Gemma3Model, GEMMA3_CONFIG_270M
from dataset import dataset_loader

# ---------- Paths and basic config ----------
deepspeed_config_path = "cosmopedia_v2.json"
global_step = 0
max_steps = 19999
log_n_steps = 50
save_every = 5000
ckpt_dir = "cosmopedia_v2_checkpoints/"
os.makedirs(ckpt_dir, exist_ok=True)

# ---------- Model + Tokenizer ----------
model = Gemma3Model(GEMMA3_CONFIG_270M)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")


# ---------- Dataloader ----------
train_dataloader = dataset_loader(
    "HuggingFaceTB/smollm-corpus",
    "cosmopedia-v2",
    "train",
    tokenizer=tokenizer,
    batch_size=1,            # micro-batch per GPU
    seq_length=16384    # doubled sequence length
)

# ---------- Criterion ----------
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ---------- Optimizer & Scheduler ----------
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

# ---------- Load DeepSpeed config ----------
if os.path.exists(deepspeed_config_path):
    with open(deepspeed_config_path) as f:
        ds_config = json.load(f)
else:
    raise FileNotFoundError(f"DeepSpeed config not found at {deepspeed_config_path}")

# ---------- DeepSpeed initialize ----------
engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config
)

device = engine.device
rank = engine.global_rank  # rank 0 prints logs

if rank == 0:
    print(f"[DeepSpeed] engine initialized. device={device}, local_rank={engine.local_rank}")

# ---------- Training loop ----------
training_start_time = time.time()  # Start global timer
for epoch in range(9999):
    for batch_idx, batch in enumerate(train_dataloader):
        # Move batch to device
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            input_ids = batch.get("input_ids")
            labels = batch.get("labels", batch.get("input_ids"))
        else:
            input_ids = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else batch[0].to(device)

        # Forward
        outputs = engine(input_ids=input_ids)
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        # Shift logits/labels for LM loss
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )

        # Backward + step
        engine.backward(loss)
        engine.step()

        global_step += 1

        # Logging (step, loss, lr)
        if rank == 0 and (global_step % log_n_steps == 0 or global_step == 1):
            lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else None
            print(f"[step {global_step}] loss={loss.item():.4f} lr={lr:.2e}")

        # Save checkpoint at intervals
        if global_step % save_every == 0:
            tag = f"step-{global_step}"
            if rank == 0:
                print(f"[DeepSpeed] saving checkpoint {tag} ...")
            engine.save_checkpoint(ckpt_dir, tag=tag)

        # Exit when max_steps reached — save final ckpt before breaking
        if global_step >= max_steps:
            if rank == 0:
                training_end_time = time.time()
                total_seconds = training_end_time - training_start_time
                hours = int(total_seconds // 3600)
                minutes = int((total_seconds % 3600) // 60)
                seconds = int(total_seconds % 60)

                print("[Training] Reached max_steps, saving final checkpoint and exiting.")
                tag = f"step-{global_step}"
                engine.save_checkpoint(ckpt_dir, tag=tag)
                print(f"[Training] Final checkpoint saved at step {global_step}")
                print(f"[Training] Total time elapsed: {hours}h {minutes}m {seconds}s")
            break

    if global_step >= max_steps:
        break
