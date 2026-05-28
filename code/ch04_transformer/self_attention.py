"""
实验4.1：自注意力机制
对应章节：第4章 - Transformer详解
目标：用 NumPy 实现简单的多头自注意力，并可视化注意力权重
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
np.random.seed(42)
SEQ_LEN = 8
D_MODEL = 32
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS
TOKEN_POSITIONS = np.arange(SEQ_LEN)

# ============ 核心逻辑 ============
def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    scores = scores - np.max(scores, axis=-1, keepdims=True)
    attn_weights = np.exp(scores)
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)
    output = attn_weights @ V
    return output, attn_weights


def build_synthetic_sequence(seq_len, d_model):
    positions = np.arange(seq_len)
    x = np.zeros((1, seq_len, d_model))

    x[0, :, 0] = positions / seq_len
    x[0, :, 1] = np.sin(positions)
    x[0, :, 2] = np.cos(positions)
    x[0, :, 3] = (positions % 2) * 2 - 1
    x[0, :, 4:] = 0.05 * np.random.randn(seq_len, d_model - 4)
    return x


def multi_head_attention(X, W_q, W_k, W_v, W_o):
    batch_size, seq_len, _ = X.shape

    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v

    Q = Q.reshape(batch_size, seq_len, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)

    attn_outputs = []
    attn_weights_list = []
    for h in range(NUM_HEADS):
        output, attn_weights = scaled_dot_product_attention(Q[0, h], K[0, h], V[0, h])
        attn_outputs.append(output)
        attn_weights_list.append(attn_weights)

    attn_output = np.concatenate(attn_outputs, axis=-1)
    output = attn_output @ W_o
    return output, attn_weights_list


X = build_synthetic_sequence(SEQ_LEN, D_MODEL)

W_q = np.zeros((D_MODEL, D_MODEL))
W_k = np.zeros((D_MODEL, D_MODEL))
W_v = np.eye(D_MODEL)
W_o = np.eye(D_MODEL)

for h in range(NUM_HEADS):
    start = h * D_HEAD
    W_q[start, start] = 2.0
    W_k[start, start] = 2.0
    W_q[start + 1, start + 1] = 1.5
    W_k[start + 1, start + 1] = 1.5

output, attn_weights_list = multi_head_attention(X, W_q, W_k, W_v, W_o)
avg_attn_weights = np.mean(np.array(attn_weights_list), axis=0)

print("=" * 70)
print("Self-Attention")
print("=" * 70)
print(f"Sequence length: {SEQ_LEN}")
print(f"Model dimension: {D_MODEL}")
print(f"Number of heads: {NUM_HEADS}")
print(f"Head dimension: {D_HEAD}")
print()

print("Input shape:")
print("-" * 70)
print(f"X: {X.shape}")
print()

print("Output shape:")
print("-" * 70)
print(f"output: {output.shape}")
print()

print("Attention weight statistics:")
print("-" * 70)
for h in range(NUM_HEADS):
    weights = attn_weights_list[h]
    print(f"Head {h}: min={weights.min():.4f}, max={weights.max():.4f}, mean={weights.mean():.4f}")
print()

print("Average attention weights (across heads):")
print("-" * 70)
print(avg_attn_weights.round(4))
print()

print("Attention pattern analysis:")
print("-" * 70)
for i in range(SEQ_LEN):
    top_k = 3
    top_indices = np.argsort(avg_attn_weights[i])[-top_k:][::-1]
    top_weights = avg_attn_weights[i, top_indices]
    print(f"Position {i} attends most to: {top_indices} (weights: {top_weights.round(4)})")
print()

print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
im = ax.imshow(avg_attn_weights, cmap="YlOrRd", aspect="auto")
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
ax.set_xticks(TOKEN_POSITIONS)
ax.set_yticks(TOKEN_POSITIONS)
ax.set_title("Average Attention Weights")
plt.colorbar(im, ax=ax)

for h in range(min(2, NUM_HEADS)):
    ax = axes[1, h]
    im = ax.imshow(attn_weights_list[h], cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    ax.set_xticks(TOKEN_POSITIONS)
    ax.set_yticks(TOKEN_POSITIONS)
    ax.set_title(f"Head {h} Attention Weights")
    plt.colorbar(im, ax=ax)

ax = axes[0, 1]
for h in range(NUM_HEADS):
    weights_flat = attn_weights_list[h].flatten()
    ax.hist(weights_flat, bins=25, alpha=0.45, label=f"Head {h}")
ax.set_xlabel("Attention Weight")
ax.set_ylabel("Frequency")
ax.set_title("Attention Weight Distribution")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

plt.tight_layout()
plt.savefig("assets/ch04_self_attention.png", dpi=100, bbox_inches="tight")
print("Figure saved to: assets/ch04_self_attention.png")
