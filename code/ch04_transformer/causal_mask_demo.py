"""
实验4.3：因果掩码演示
对应章节：第4章 - Transformer详解
目标：对比双向注意力和 causal mask 对信息流的影响
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
SEQ_LEN = 8
D_MODEL = 16

X = np.random.randn(SEQ_LEN, D_MODEL)
W_q = np.random.randn(D_MODEL, D_MODEL) * 0.1
W_k = np.random.randn(D_MODEL, D_MODEL) * 0.1
W_v = np.random.randn(D_MODEL, D_MODEL) * 0.1

Q = X @ W_q
K = X @ W_k
V = X @ W_v


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


scores = Q @ K.T / np.sqrt(D_MODEL)
bidirectional_weights = softmax(scores)

causal_mask = np.triu(np.full((SEQ_LEN, SEQ_LEN), -1e9), k=1)
causal_weights = softmax(scores + causal_mask)

bidirectional_output = bidirectional_weights @ V
causal_output = causal_weights @ V

print("=" * 70)
print("Causal Mask Demo")
print("=" * 70)
print(f"Sequence length: {SEQ_LEN}")
print(f"Model dimension: {D_MODEL}")
print()

print("Bidirectional attention: each position can attend to the full sequence.")
print("Causal attention: each position can only attend to itself and previous positions.")
print()

for i in range(SEQ_LEN):
    bi_top = np.argsort(bidirectional_weights[i])[-3:][::-1]
    ca_top = np.argsort(causal_weights[i])[-3:][::-1]
    print(f"Position {i}: bidirectional top attention {bi_top}, causal top attention {ca_top}")

print()
print("Output difference (L2 norm):")
for i in range(SEQ_LEN):
    diff = np.linalg.norm(bidirectional_output[i] - causal_output[i])
    print(f"Position {i}: {diff:.4f}")

fig, axes = plt.subplots(2, 2, figsize=(12, 10))

ax = axes[0, 0]
im = ax.imshow(scores, cmap="coolwarm", aspect="auto")
ax.set_title("Raw Attention Scores")
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
im = ax.imshow(bidirectional_weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_title("Bidirectional Attention")
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
plt.colorbar(im, ax=ax)

ax = axes[1, 0]
im = ax.imshow(causal_mask, cmap="gray", aspect="auto")
ax.set_title("Causal Mask")
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
plt.colorbar(im, ax=ax)

ax = axes[1, 1]
im = ax.imshow(causal_weights, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
ax.set_title("Causal Attention")
ax.set_xlabel("Key Position")
ax.set_ylabel("Query Position")
plt.colorbar(im, ax=ax)

plt.tight_layout()
plt.savefig("assets/ch04_causal_mask.png", dpi=120, bbox_inches="tight")
print("Figure saved to: assets/ch04_causal_mask.png")
