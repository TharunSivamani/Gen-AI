import torch
import torch.nn as nn

class PixelShuffle(nn.Module):
    """
    Pixel Shuffle implementation from scratch.

    Rearranges elements in a tensor of shape (N, C*r^2, H, W)
    to a tensor of shape (N, C, H*r, W*r).

    Args:
        upscale_factor (int): Factor to increase spatial resolution.

    Reference:
        Shi et al., "Real-Time Single Image and Video Super-Resolution
        Using an Efficient Sub-Pixel Convolutional Neural Network"
        (https://arxiv.org/abs/1609.05158)
    """

    def __init__(self, upscale_factor: int):
        super().__init__()
        self.r = upscale_factor

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c_in, h, w = x.size()
        r = self.r
        if c_in % (r * r) != 0:
            raise ValueError(f"Input Channel ({c_in}) not divisible by upscale_factor ({self.r})")
        c_out = c_in // (r * r)

        # (N, c*r^2, H, W) -> (N, C, r, r, H, W)
        x = x.view(n, c_out, r, r, h, w)

        # (N, C, H, r, W, r)
        x = x.permute(0, 1, 4, 2, 5, 3)

        # (N, C, H*r, W*r)
        return x.contiguous().view(n, c_out, h * r, w * r)
    

class PixelUnshuffle(nn.Module):
    """
    Inverse of Pixel Shuffle.

    Rearranges elements in a tensor of shape (N, C, H*r, W*r)
    to a tensor of shape (N, C*r^2, H, W).

    Args:
        downscale_factor (int): Factor to reduce spatial resolution.
    """

    def __init__(self, downscale_factor: int) -> None:
        super().__init__()
        self.r = downscale_factor

    def forward(self, x: torch.Tensor):
        n, c_in, h_in, w_in = x.size()
        r = self.r
        if h_in % r != 0 or w_in % r != 0:
            raise ValueError(f"Height in ({h_in}) and Width in ({w_in}) should be divisible by downscale factor ({r})")
        
        # (N, c_in, h * r, w * r) - divide by r to check for divisibility.
        h, w = h_in // r, w_in // r

        # (N, C, H*r, W*r) → (N, C, H, r, W, r)
        x = x.view(n, c_in, h, r, w, r)

        # (N, C, r, r, H, W)
        x = x.permute(0, 1, 3, 5, 2, 4)

        # (N, C*r^2, H, W)
        return x.contiguous().view(n, c_in * r * r, h, w)
    

if __name__ == "__main__":

    N, C, H, W, r = 1, 2, 3, 3, 2

    x = torch.arange(N * C * r * r * H * W).float().reshape(N, C * r * r, H, W)

    print("Input shape:", x.shape)

    ps = PixelShuffle(r)
    pu = PixelUnshuffle(r)

    y = ps(x)
    print("After PixelShuffle:", y.shape)

    z = pu(y)
    print("After PixelUnshuffle:", z.shape)

    # Verify correctness
    print("Restored identical to input:", torch.allclose(x, z))

"""

Input shape: torch.Size([1, 8, 3, 3])
After PixelShuffle: torch.Size([1, 2, 6, 6])
After PixelUnshuffle: torch.Size([1, 8, 3, 3])
Restored identical to input: True

"""