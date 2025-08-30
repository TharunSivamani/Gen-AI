# train_with_deepspeed.py
# deepspeed --num_gpus=4 smolds.py 
import os
import json
import torch
import torch.nn as nn
import deepspeed
from transformers import AutoTokenizer

# your model imports
from gemma import Gemma3Model, GEMMA3_CONFIG_270M
# NOTE: we are not using your train.train_loop; we implement a small loop here
from dataset import dataset_loader

# ---------- Basic config ----------
deepspeed_config_path = "deepspeed_config.json"
max_steps = 100
log_n_steps = 5
save_every = 50

# ---------- Model + tokenizer ----------
model = Gemma3Model(GEMMA3_CONFIG_270M)
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

# ---------- Dataloader ----------
train_dataloader = dataset_loader(
    "HuggingFaceTB/smollm-corpus",
    "cosmopedia-v2",
    "train",
    tokenizer=tokenizer,
    batch_size=1,
    seq_length=16384
)

# ---------- Criterion ----------
criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)

# ---------- Prepare optimizer & scheduler (we'll hand them to DeepSpeed) ----------
# create optimizer as you had it to preserve hyperparams; DeepSpeed can also create optimizer from config.
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=3e-4,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=0.1
)

# You created a CosineAnnealingLR. DeepSpeed supports passing a scheduler object at initialization.
lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=5000)

# ---------- DeepSpeed initialization ----------
# Load json config or dict
if os.path.exists(deepspeed_config_path):
    with open(deepspeed_config_path) as f:
        ds_config = json.load(f)
else:
    raise FileNotFoundError(f"DeepSpeed config not found at {deepspeed_config_path}")

# deepspeed.initialize returns (engine, optimizer, training_dataloader, lr_scheduler) sometimes a tuple of length 4.
# We pass model, optimizer, model parameters, and scheduler so DeepSpeed can wire everything (and use ZeRO offload).
# Initialize DeepSpeed and have it construct optimizer from config:
engine, optimizer, _, lr_scheduler = deepspeed.initialize(
    model=model,
    model_parameters=model.parameters(),
    config_params=ds_config  # or config file path
)

device = engine.device  # target device for inputs (usually cuda:local_rank or cpu if no CUDA)
print(f"[DeepSpeed] engine initialized. device={device}, local_rank={engine.local_rank}")

# If you want to save every N steps using DeepSpeed save, use engine.save_checkpoint(path, tag=...)
ckpt_dir = "./ds_checkpoints"
os.makedirs(ckpt_dir, exist_ok=True)

# ---------- Simple training loop using DeepSpeed engine ----------
global_step = 0
for epoch in range(0, 9999):  # epoch loop but we break by max_steps
    for batch_idx, batch in enumerate(train_dataloader):
        # Expect batch to be dict-like with input_ids and labels; adapt to your dataset loader format.
        # Move tensors to engine.device:
        if isinstance(batch, dict):
            batch = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}
            input_ids = batch.get("input_ids")
            labels = batch.get("labels", batch.get("input_ids"))  # adapt as needed
        else:
            # If dataset returns a tuple/list
            input_ids = batch[0].to(device)
            labels = batch[1].to(device) if len(batch) > 1 else batch[0].to(device)

        # Forward - call model (ensure your model returns logits)
        # If your model returns a dict or tuple, adapt accordingly.
        outputs = engine(input_ids=input_ids)  # engine wraps model
        # assume outputs is logits: [batch, seq_len, vocab_size]
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits

        # shift logits/labels per language model CrossEntropy expectation:
        shift_logits = logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()

        loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
        # DeepSpeed backward
        engine.backward(loss)
        engine.step()  # handles optimizer step + lr scheduler step under the hood

        global_step += 1

        if global_step % log_n_steps == 0 or global_step == 1:
            # get lr from optimizer param group (DeepSpeed wraps optimizer)
            lr = optimizer.param_groups[0]["lr"] if hasattr(optimizer, "param_groups") else None
            print(f"[step {global_step}] loss={loss.item():.4f} lr={lr}")

        if global_step % save_every == 0:
            # tag-based checkpoint; DeepSpeed will save ZeRO partitioned state
            tag = f"step-{global_step}"
            print(f"[DeepSpeed] saving checkpoint {tag} ...")
            engine.save_checkpoint(ckpt_dir, tag=tag)

        if global_step >= max_steps:
            print("[Training] Reached max_steps, exiting.")
            break
    if global_step >= max_steps:
        break

print("Training finished.")
