"""
实验4.4：位置编码
对应章节：第4章 - Transformer详解
目标：打印位置编码矩阵前几行，观察其周期性、频率结构与相邻位置关系
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

SEQ_LEN = 32
D_MODEL = 64


def positional_encoding(seq_len, d_model):
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))

    pe[:, 0::2] = np.sin(position * div_term)
    pe[:, 1::2] = np.cos(position * div_term)
    return pe


pe = positional_encoding(SEQ_LEN, D_MODEL)

print("=" * 70)
print("Positional Encoding")
print("=" * 70)
print(f"Sequence length: {SEQ_LEN}")
print(f"Model dimension: {D_MODEL}")
print(f"Encoding matrix shape: {pe.shape}")
print()

print("First 5 positions (first 8 dimensions):")
print("-" * 70)
for pos in range(5):
    print(f"Position {pos}: {pe[pos, :8].round(4)}")
print()

print("Periodicity check:")
print("-" * 70)
sim_01 = np.dot(pe[0], pe[1]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[1]))
sim_12 = np.dot(pe[1], pe[2]) / (np.linalg.norm(pe[1]) * np.linalg.norm(pe[2]))
print(f"Cosine similarity between positions 0 and 1: {sim_01:.4f}")
print(f"Cosine similarity between positions 1 and 2: {sim_12:.4f}")
print()

print("Orthogonality check (first 4 dimensions):")
print("-" * 70)
pe_subset = pe[:, :4]
gram_matrix = pe_subset.T @ pe_subset / SEQ_LEN
print("Gram matrix (should be near identity after normalization):")
print(gram_matrix.round(4))
print()

print("=" * 70)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

ax = axes[0, 0]
im = ax.imshow(pe.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
ax.set_xlabel("Position")
ax.set_ylabel("Dimension")
ax.set_title("Positional Encoding Heatmap")
plt.colorbar(im, ax=ax)

ax = axes[0, 1]
for dim in [0, 2, 4, 6]:
    ax.plot(pe[:, dim], label=f"Dim {dim}", linewidth=1.5)
ax.set_xlabel("Position")
ax.set_ylabel("Value")
ax.set_title("Periodicity of Different Dimensions")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 0]
norms = np.linalg.norm(pe, axis=1)
ax.plot(norms, "b-", linewidth=2)
ax.set_xlabel("Position")
ax.set_ylabel("Norm")
ax.set_title("Norm of Positional Encoding")
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
similarities = []
for i in range(SEQ_LEN - 1):
    sim = np.dot(pe[i], pe[i + 1]) / (np.linalg.norm(pe[i]) * np.linalg.norm(pe[i + 1]))
    similarities.append(sim)
ax.plot(similarities, "g-", linewidth=2)
ax.set_xlabel("Position")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Similarity Between Adjacent Positions")
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assets/ch04_positional_encoding.png", dpi=100, bbox_inches="tight")
print("Figure saved to: assets/ch04_positional_encoding.png")
