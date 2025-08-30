""" 
If saved in fp32 and need bf16 for inference
import torch
from gemma import Gemma3Model, GEMMA3_CONFIG_270M

# load merged checkpoint (FP32)
state_dict = torch.load("/home/jovyan/Gemma/merged_model.pt", map_location="cpu")

model = Gemma3Model(GEMMA3_CONFIG_270M)
model.load_state_dict(state_dict, strict=False)

# move to GPU and optionally cast
device = torch.device("cuda")
model = model.to(device).bfloat16()  # cast to BF16
model.eval()
"""

import torch
from gemma import Gemma3Model, GEMMA3_CONFIG_270M
from transformers import AutoTokenizer
import torch.nn.functional as F

# 1. Initialize model and load checkpoint
model = Gemma3Model(GEMMA3_CONFIG_270M)

state_dict = torch.load(
    "/home/jovyan/Gemma/merged_model_375_bf16.pt",
    map_location="cpu"
)
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda")
model = model.to(device).eval()

# 2. Tokenizer
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

# Prompt
prompt = "Tell me about AI"
input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)

# 3. Autoregressive generation
max_new_tokens = 10
temperature = 0.8
top_k = 50

generated = input_ids
print(f"Starting generation with prompt: '{prompt}'")
print(f"Input shape: {generated.shape}")

with torch.no_grad():
    for step in range(max_new_tokens):
        # Forward pass - get logits for the last token
        logits = model(generated)
        
        # Handle different output formats
        if isinstance(logits, tuple):
            logits = logits[0]  # Take first element if tuple
        elif hasattr(logits, 'logits'):
            logits = logits.logits  # Extract logits attribute
        
        # Get logits for the last position only
        next_token_logits = logits[:, -1, :] / temperature
        
        # Apply top-k filtering if specified
        if top_k > 0:
            # Get top-k tokens and their logits
            top_k_logits, top_k_indices = torch.topk(next_token_logits, top_k, dim=-1)
            
            # Create a mask for top-k tokens
            mask = torch.full_like(next_token_logits, float('-inf'))
            mask.scatter_(-1, top_k_indices, top_k_logits)
            next_token_logits = mask
        
        # Convert to probabilities and sample
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1)
        
        # Append the new token
        generated = torch.cat([generated, next_token], dim=1)
        
        # Optional: print progress
        current_text = tokenizer.decode(generated[0], skip_special_tokens=True)
        print(f"Step {step + 1}: {current_text}")
        
        # Optional: stop on EOS token
        if next_token.item() == tokenizer.eos_token_id:
            print("Generated EOS token, stopping...")
            break

# 4. Decode final output
final_text = tokenizer.decode(generated[0], skip_special_tokens=True)
print(f"\nFinal generated text: '{final_text}'")

# Optional: show just the new tokens
new_tokens_only = generated[0][input_ids.shape[1]:]
new_text = tokenizer.decode(new_tokens_only, skip_special_tokens=True)
print(f"New tokens only: '{new_text}'")