import torch

def compute_rope_params(head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32):

    assert head_dim % 2 == 0, "Head Dimension must be divisible by 2"

    # compute inverse frequency

    # generate position indices

    # compute the angles

    # expand the angles to match the head dim

    # pre-compute sine and cosine angles

    # return cos, sin
    pass

def apply_rope(x, cos, sin):

    # x: (batch_size, n_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head Dimension must be divisible by 2"

    # split x into first half, second half

    # adjust sine and cosine shapes

    # aply the rotary transformations

    # return the rotated matrix
    pass