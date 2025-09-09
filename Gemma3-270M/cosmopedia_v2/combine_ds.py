# python -m deepspeed.checkpoint.consolidate \
#     --checkpoint_dir /home/jovyan/Gemma/ds_checkpoints/step-100 \
#     --tag consolidated

# from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict

# checkpoint_dir = "/home/jovyan/Gemma/ds_checkpoints"
# output_file = "/home/jovyan/Gemma/merged_model"

# convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_file)

from deepspeed.utils.zero_to_fp32 import convert_zero_checkpoint_to_fp32_state_dict
import torch

checkpoint_dir = "/home/jovyan/Gemma/cosmopedia_v2_checkpoints"
output_file = "/home/jovyan/Gemma/merged_model-21k"

# 1. Merge to FP32 first
convert_zero_checkpoint_to_fp32_state_dict(checkpoint_dir, output_file)

# 2. Load and cast to BF16
state_dict = torch.load("/home/jovyan/Gemma/merged_model-21k/pytorch_model.bin", map_location="cpu")
state_dict_bf16 = {k: v.to(torch.bfloat16) for k, v in state_dict.items()}

torch.save(state_dict_bf16, "/home/jovyan/Gemma/merged_model-21k/bf16.pt")
