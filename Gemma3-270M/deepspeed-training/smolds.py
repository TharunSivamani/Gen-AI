# train_with_deepspeed_h100.py
# Run with: deepspeed --num_gpus=8 train_with_deepspeed_h100.py

import os
import json
import torch
import torch.nn as nn
import deepspeed
from transformers import AutoTokenizer

from gemma import Gemma3Model, GEMMA3_CONFIG_270M
from dataset import dataset_loader

# ---------- Paths and basic config ----------
deepspeed_config_path = "deepspeed_config.json"
max_steps = 200
log_n_steps = 10
save_every = 75
ckpt_dir = "./ds_checkpoints"
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

lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

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
global_step = 0
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

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))

        # Backward + step
        engine.backward(loss)
        engine.step()

        global_step += 1

        # Logging
        if rank == 0 and (global_step % log_n_steps == 0 or global_step == 1):
            lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else None
            print(f"[step {global_step}] loss={loss.item():.4f} lr={lr}")

        # Save checkpoint
        if global_step % save_every == 0:
            tag = f"step-{global_step}"
            if rank == 0:
                print(f"[DeepSpeed] saving checkpoint {tag} ...")
            engine.save_checkpoint(ckpt_dir, tag=tag)

        if global_step >= max_steps:
            if rank == 0:
                print("[Training] Reached max_steps, exiting.")
            break
    if global_step >= max_steps:
        break

# ---------- Save final model weights only ----------
if rank == 0:
    final_model_path = "./final_model.pt"
    print(f"[Training] Saving final model weights only to {final_model_path}")
    torch.save(engine.module.state_dict(), final_model_path)

if rank == 0:
    print("Training finished.")
