import torch
from rope import apply_rope, compute_rope_params

# Example dimensions
batch_size = 1
num_heads = 1
seq_len = 4
head_dim = 8  # Must be even

# Create random tensor representing attention input
x = torch.randn(batch_size, num_heads, seq_len, head_dim)

# Compute cos/sin matrices used in RoPE
cos, sin = compute_rope_params(head_dim=head_dim, context_length=seq_len)

# Apply RoPE
x_rope = apply_rope(x, cos, sin)

# Output
print("Original x:\n", x)
print("RoPE applied x:\n", x_rope)
