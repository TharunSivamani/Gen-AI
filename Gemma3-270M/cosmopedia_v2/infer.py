import torch
from gemma import Gemma3Model, GEMMA3_CONFIG_270M
from transformers import AutoTokenizer

# ---------- Initialize model and load checkpoint ----------
model = Gemma3Model(GEMMA3_CONFIG_270M)

state_dict = torch.load(
    "/home/jovyan/Gemma/merged_model-21k/bf16.pt",
    map_location="cpu"
)
model.load_state_dict(state_dict, strict=False)

device = torch.device("cuda")
model = model.to(device).eval()  # switch to eval mode for inference

# ---------- Tokenizer ----------
tokenizer = AutoTokenizer.from_pretrained("google/gemma-3-270m")

# ---------- Sample prompts ----------
sample_texts = [
    {"prompt": "The future of robotics is"},
    {"prompt": "In a distant galaxy,"},
    {"prompt": "Explain quantum computing in simple terms:"},
    {"prompt": "Once upon a time in a small village,"},
    {"prompt": "Top 5 tips for healthy living are:"},
]


# ---------- Inference ----------
with torch.no_grad():
    for i, sample in enumerate(sample_texts):
        print(f"\nGeneration {i+1} with prompt: {sample['prompt']}")
        # Encode input prompt
        inputs = tokenizer(sample['prompt'], return_tensors="pt", padding=True, truncation=True).to(device)
        # Forward pass
        outputs = model(inputs["input_ids"])
        logits = outputs if isinstance(outputs, torch.Tensor) else outputs.logits
        # Take argmax for each token
        prediction = torch.argmax(logits, dim=-1)
        # Decode generated tokens
        decoded = tokenizer.decode(prediction[0], skip_special_tokens=True)
        print(f"Generated: {decoded}\n")
