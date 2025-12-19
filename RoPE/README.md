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
```

---
*Reference*: `rope.ipynb`

## 1. What this function is trying to do (high level)

`simple_rotary_embedding` applies a **rotary positional embedding (RoPE)** idea:

* The embedding vector is split into **pairs of dimensions**.
* Each pair is treated as a 2D point.
* That point is **rotated** by an angle that depends on the token position.
* Different positions → different rotations → position information is encoded.

This is inspired by how modern transformers encode position.

---

## 2. Inputs to the function

```python
x = [1, 2, 4, 1]
position = 0, 1, or 2
```

* `x` is a vector of length 4
* It will be processed as **two pairs**:

  * Pair 1: `(1, 2)`
  * Pair 2: `(4, 1)`

---

## 3. Compute the rotation angle

```python
theta = 0.1 * position
```

So:

| Position | θ (radians) |
| -------- | ----------- |
| 0        | 0.0         |
| 1        | 0.1         |
| 2        | 0.2         |

---

## 4. Initialize the output vector

```python
rotated_x = np.zeros_like(x)
```

⚠️ **Important detail**
`x` is an **integer array**, so `rotated_x` is also **integer**.

This means:

* Any floating-point results will be **truncated** (not rounded!)

This explains your output later.

---

## 5. Loop through the vector in steps of 2

```python
for i in range(0, dim, 2):
```

This runs for:

* `i = 0` → dimensions `(0, 1)`
* `i = 2` → dimensions `(2, 3)`

---

## 6. Apply 2D rotation formula

For each pair `(x[i], x[i+1])`, you apply:

```
x′ = x * cosθ − y * sinθ
y′ ​= x * sinθ + y * cosθ​
```

This is the standard 2D rotation matrix.

---

## 7. Position 0 (θ = 0)

### Math

* cos(0) = 1
* sin(0) = 0

So rotation does nothing.

### Pair 1: (1, 2)

```
x' = 1*1 - 2*0 = 1
y' = 1*0 + 2*1 = 2
```

### Pair 2: (4, 1)

```
x' = 4*1 - 1*0 = 4
y' = 4*0 + 1*1 = 1
```

### Result

```python
[1, 2, 4, 1]
```

✔ Matches output.

---

## 8. Position 1 (θ = 0.1)

### Trigonometric values

* cos(0.1) ≈ 0.995
* sin(0.1) ≈ 0.0998

---

### Pair 1: (1, 2)

**Float math**

```
x' ≈ 1*0.995 - 2*0.0998 = 0.795
y' ≈ 1*0.0998 + 2*0.995 = 2.0898
```

**Stored as integers**

```
x' → 0
y' → 2
```

---

### Pair 2: (4, 1)

**Float math**

```
x' ≈ 4*0.995 - 1*0.0998 = 3.88
y' ≈ 4*0.0998 + 1*0.995 = 1.39
```

**Stored as integers**

```
x' → 3
y' → 1
```

---

### Result

```python
[0, 2, 3, 1]
```

✔ Matches output.

---

## 9. Position 2 (θ = 0.2)

### Trigonometric values

* cos(0.2) ≈ 0.980
* sin(0.2) ≈ 0.199

The floating-point results are **slightly different** from position 1, but after truncation to integers, they become the same.

### Result (after truncation)

```python
[0, 2, 3, 1]
```

✔ Same output as position 1.

---

## 10. Why position 1 and 2 give the same output

**Key reason: integer truncation**

Because:

```python
rotated_x = np.zeros_like(x)
```

* The array is `int`
* Decimal values are discarded
* Small changes in rotation are lost

---

## 11. How to fix it (important!)

If you want **correct rotary embeddings**, use floats:

```python
rotated_x = np.zeros_like(x, dtype=float)
```

Then you would see different outputs for different positions.

---

## 12. Conceptual takeaway

* Rotary embeddings encode position by **rotating embedding vectors**
* Each pair of dimensions acts like a **complex number**
* Rotation angle increases with position
* **Data type matters**: integers destroy precision
