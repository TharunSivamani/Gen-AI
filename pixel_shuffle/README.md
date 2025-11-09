# 🧩 Pixel Shuffle (and Pixel Unshuffle) — From Scratch

## 📘 Overview

**Pixel Shuffle** is a tensor rearrangement operation that converts **low-resolution feature maps with many channels** into **high-resolution feature maps with fewer channels**.
It’s a *computationally efficient* alternative to transposed convolutions, often used in **super-resolution** networks (like ESPCN, SRGAN, and ESRGAN).

Its inverse operation — **Pixel Unshuffle** — reverses this process by packing spatial information back into channels.


## 🧠 Concept

### 🔹 Pixel Shuffle

Rearranges data from the **channel dimension** into **spatial dimensions**.

Input:  (N, C × r², H, W)
   ↓
Output: (N, C, H × r, W × r)

where:

* ( N ): Batch size
* ( C ): Number of output channels
* ( H, W ): Height & Width before upscaling
* ( r ): Upscale factor

### 🔹 Pixel Unshuffle

Inverse operation — packs spatial data back into channels:

**Pixel Unshuffle Transformation**

Input:  (N, C, H × r, W × r)
   ↓
Output: (N, C × r², H, W)


## ⚙️ Implementation (From Scratch)

```python
import numpy as np

def pixel_shuffle_numpy(x, upscale_factor):
    N, Crr, H, W = x.shape
    r = upscale_factor
    C = Crr // (r * r)
    x = x.reshape(N, C, r, r, H, W)
    x = x.transpose(0, 1, 4, 2, 5, 3)
    return x.reshape(N, C, H * r, W * r)

def pixel_unshuffle_numpy(x, downscale_factor):
    N, C, Hrr, Wrr = x.shape
    r = downscale_factor
    H = Hrr // r
    W = Wrr // r
    x = x.reshape(N, C, H, r, W, r)
    x = x.transpose(0, 1, 3, 5, 2, 4)
    return x.reshape(N, C * r * r, H, W)
```

✅ Verified to match `torch.nn.PixelShuffle` and `torch.nn.PixelUnshuffle`.


## 🧩 Example

```python
import torch

N, C, H, W, r = 1, 1, 2, 2, 2
x = torch.arange(N * C * r * r * H * W).float().reshape(N, C * r * r, H, W)

print("Input:", x.shape)
# Pixel Shuffle
ps = torch.nn.PixelShuffle(r)
y = ps(x)
print("After PixelShuffle:", y.shape)

# Pixel Unshuffle
pu = torch.nn.PixelUnshuffle(r)
z = pu(y)
print("After PixelUnshuffle:", z.shape)
```

**Output:**

```
Input: (1, 4, 2, 2)
After PixelShuffle: (1, 1, 4, 4)
After PixelUnshuffle: (1, 4, 2, 2)
```

## 📊 Shape Summary

| Operation             | Input Shape        | Output Shape       | Effect                                 |
| --------------------- | ------------------ | ------------------ | -------------------------------------- |
| **PixelShuffle(r)**   | `(N, C*r², H, W)`  | `(N, C, H*r, W*r)` | Upscale spatially, reduce channels     |
| **PixelUnshuffle(r)** | `(N, C, H*r, W*r)` | `(N, C*r², H, W)`  | Downscale spatially, increase channels |


## 🚀 Applications

* **Image Super-Resolution** (e.g., SRGAN, ESRGAN)
* **Video Upscaling**
* **Efficient Decoders** in Vision Transformers / Autoencoders
* **Learned Downsampling** (via PixelUnshuffle + Convolution)


## 🧮 Reference

* Shi et al., *Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network*, CVPR 2016.
  [arXiv:1609.05158](https://arxiv.org/abs/1609.05158)
* [PyTorch Docs — torch.nn.PixelShuffle](https://docs.pytorch.org/docs/stable/generated/torch.nn.PixelShuffle.html)
