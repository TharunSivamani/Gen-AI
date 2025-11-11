import torch


def compute_rope_params(
    head_dim, theta_base=10_000, context_length=4096, dtype=torch.float32
):

    assert head_dim % 2 == 0, "Head Dimension must be divisible by 2"

    # compute inverse frequency
    inv_freq = 1.0 / (
        theta_base
        ** (
            torch.arange(0, head_dim, 2, dtype=dtype)[: head_dim // 2].float()
            / head_dim
        )
    )

    # generate position indices
    positions = torch.arange(context_length, dtype=dtype)

    # compute the angles
    angles = (
        positions[:, None] * inv_freq[None, :]
    )  # Shape: (context_length, head_dim // 2)

    # expand the angles to match the head dim
    angles = torch.cat([angles, angles], dim=1)  # Shape: (context_length, head_dim)

    # pre-compute sine and cosine angles
    cos = torch.cos(angles)
    sin = torch.sin(angles)

    return cos, sin


def apply_rope(x, cos, sin):

    # x: (batch_size, n_heads, seq_len, head_dim)
    batch_size, num_heads, seq_len, head_dim = x.shape
    assert head_dim % 2 == 0, "Head Dimension must be divisible by 2"

    # split x into first half, second half
    x1 = x[..., : head_dim // 2]
    x2 = x[..., head_dim // 2 :]

    # adjust sine and cosine shapes
    # Shape: (1, 1, seq_len, head_dim)
    cos = cos[:seq_len, :].unsqueeze(0).unsqueeze(0)
    sin = sin[:seq_len, :].unsqueeze(0).unsqueeze(0)

    # aply the rotary transformations
    rotated = torch.cat((-x2, x1), dim=-1)
    x_rotated = (x * cos) + (rotated * sin)

    # return the rotated matrix
    return x_rotated.to(dtype=x.dtype)
