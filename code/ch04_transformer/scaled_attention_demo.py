"""
实验4.2：缩放点积注意力演示
对应章节：第4章 - Transformer详解
目标：展示为什么注意力分数需要除以 sqrt(d_k)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
NUM_SAMPLES = 4000
DIMS = [8, 32, 128, 512]

raw_score_stds = []
scaled_score_stds = []
raw_softmax_peaks = []
scaled_softmax_peaks = []

print("=" * 70)
print("Scaled Dot-Product Attention Demo")
print("=" * 70)

for d_k in DIMS:
    Q = np.random.randn(NUM_SAMPLES, d_k)
    K = np.random.randn(NUM_SAMPLES, d_k)

    raw_scores = np.sum(Q * K, axis=1)
    scaled_scores = raw_scores / np.sqrt(d_k)

    logits_raw = np.vstack([raw_scores[:50], np.zeros(50)]).T
    logits_scaled = np.vstack([scaled_scores[:50], np.zeros(50)]).T

    exp_raw = np.exp(logits_raw - np.max(logits_raw, axis=1, keepdims=True))
    exp_scaled = np.exp(logits_scaled - np.max(logits_scaled, axis=1, keepdims=True))

    probs_raw = exp_raw / np.sum(exp_raw, axis=1, keepdims=True)
    probs_scaled = exp_scaled / np.sum(exp_scaled, axis=1, keepdims=True)

    raw_score_stds.append(np.std(raw_scores))
    scaled_score_stds.append(np.std(scaled_scores))
    raw_softmax_peaks.append(np.mean(np.max(probs_raw, axis=1)))
    scaled_softmax_peaks.append(np.mean(np.max(probs_scaled, axis=1)))

    print(f"d_k={d_k:>3}: raw std={raw_score_stds[-1]:.4f}, scaled std={scaled_score_stds[-1]:.4f}, "
          f"raw peak={raw_softmax_peaks[-1]:.4f}, scaled peak={scaled_softmax_peaks[-1]:.4f}")

print("=" * 70)
print("Conclusion: as dimension grows, unscaled dot products make softmax too sharp; dividing by sqrt(d_k) keeps it stable.\n")

fig, axes = plt.subplots(2, 2, figsize=(13, 10))

ax = axes[0, 0]
for d_k in DIMS:
    Q = np.random.randn(NUM_SAMPLES, d_k)
    K = np.random.randn(NUM_SAMPLES, d_k)
    raw_scores = np.sum(Q * K, axis=1)
    ax.hist(raw_scores, bins=50, alpha=0.45, label=f"d_k={d_k}")
ax.set_title("Raw Dot-Product Score Distribution")
ax.set_xlabel("Score")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

ax = axes[0, 1]
for d_k in DIMS:
    Q = np.random.randn(NUM_SAMPLES, d_k)
    K = np.random.randn(NUM_SAMPLES, d_k)
    scaled_scores = np.sum(Q * K, axis=1) / np.sqrt(d_k)
    ax.hist(scaled_scores, bins=50, alpha=0.45, label=f"d_k={d_k}")
ax.set_title("Scaled Score Distribution")
ax.set_xlabel("Score / sqrt(d_k)")
ax.set_ylabel("Frequency")
ax.legend()
ax.grid(True, alpha=0.3, axis="y")

ax = axes[1, 0]
ax.plot(DIMS, raw_score_stds, marker="o", label="Raw score std")
ax.plot(DIMS, scaled_score_stds, marker="o", label="Scaled score std")
ax.set_title("Score Standard Deviation vs d_k")
ax.set_xlabel("d_k")
ax.set_ylabel("Standard deviation")
ax.legend()
ax.grid(True, alpha=0.3)

ax = axes[1, 1]
ax.plot(DIMS, raw_softmax_peaks, marker="o", label="Raw softmax peak")
ax.plot(DIMS, scaled_softmax_peaks, marker="o", label="Scaled softmax peak")
ax.set_title("Average Softmax Peak vs d_k")
ax.set_xlabel("d_k")
ax.set_ylabel("Average max probability")
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("assets/ch04_scaled_attention.png", dpi=120, bbox_inches="tight")
print("Figure saved to: assets/ch04_scaled_attention.png")
