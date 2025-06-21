# 🌀 Rotary Position Embedding (RoPE)

A minimal and clear PyTorch implementation of [**Rotary Position Embedding (RoPE)**](https://arxiv.org/abs/2104.09864v5) — a technique used to inject positional information into transformer attention mechanisms using rotation matrices.

---

## 📄 Paper

**[RoFormer: Enhanced Transformer with Rotary Position Embedding](https://arxiv.org/abs/2104.09864v5)**  
*Authors: Jianlin Su, Yu Lu, Shengfeng Pan, Bo Wen, Yunfeng Liu*

---

## 📌 Overview

Unlike traditional absolute or relative positional embeddings, **RoPE** introduces a rotation-based approach to encode positions directly into attention queries and keys.

### ✨ Key Benefits:
- Captures relative position in a continuous and elegant way.
- Scales better to long sequences.
- Simple to implement and integrates cleanly with existing transformer architectures.

---

## 🧠 How It Works

RoPE rotates the query and key vectors in a sinusoidal pattern based on their position. This is equivalent to a complex number multiplication:

```text
RoPE(qᵢ, kⱼ) = qᵢ • R(θᵢ)ᵗ • R(θⱼ) • kⱼᵗ


Code: https://nn.labml.ai/transformers/rope/index.html