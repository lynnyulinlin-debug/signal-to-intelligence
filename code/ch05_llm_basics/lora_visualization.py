"""
实验5.4a：LoRA 低秩分解可视化
对应章节：第5章 5.4 微调
目标：展示 LoRA 的矩阵分解原理，对比全量微调与 LoRA 的参数量差异
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "../../assets/ch05_lora_visualization.png"

np.random.seed(42)

# ============ LoRA 矩阵近似实验 ============

def low_rank_approx(W, rank):
    """用 SVD 做低秩近似，模拟 LoRA 的 A×B 分解"""
    U, S, Vt = np.linalg.svd(W, full_matrices=False)
    W_approx = U[:, :rank] @ np.diag(S[:rank]) @ Vt[:rank, :]
    error = np.linalg.norm(W - W_approx, "fro") / np.linalg.norm(W, "fro")
    return W_approx, error

# 模拟一个小的权重矩阵（代表 attention 层的一部分）
d = 64   # 简化维度（实际是 4096）
W_original = np.random.randn(d, d) * 0.1
# 加入低秩结构（模拟真实微调中的低秩变化）
rank_true = 4
A_true = np.random.randn(d, rank_true) * 0.3
B_true = np.random.randn(rank_true, d) * 0.3
W_delta = A_true @ B_true   # 真实的权重变化是低秩的

ranks = [1, 2, 4, 8, 16, 32]
errors = []
param_ratios = []
for r in ranks:
    _, err = low_rank_approx(W_delta, r)
    errors.append(err)
    # LoRA 参数量 = d*r + r*d = 2*d*r，全量 = d*d
    param_ratios.append(2 * d * r / (d * d) * 100)

# ============ 绘图 ============
fig = plt.figure(figsize=(16, 7))
fig.suptitle("LoRA: Low-Rank Adaptation for Efficient Finetuning", fontsize=14, fontweight="bold")

gs = plt.GridSpec(1, 3, figure=fig, wspace=0.35)

# --- 左图：矩阵分解示意 ---
ax1 = fig.add_subplot(gs[0])

# 绘制矩阵示意（用色块表示）
d_show = 8   # 展示用的小矩阵
r_show = 2

W_show = np.random.randn(d_show, d_show)
A_show = np.random.randn(d_show, r_show)
B_show = np.random.randn(r_show, d_show)

# 位置布局
positions = {
    "W": (0.0, 0.1, 0.35, 0.8),    # (left, bottom, width, height) in axes coords
    "A": (0.45, 0.1, 0.12, 0.8),
    "B": (0.62, 0.35, 0.35, 0.3),
}

norm = Normalize(vmin=-2, vmax=2)
cmap = plt.cm.RdBu_r

for label, (left, bottom, width, height) in positions.items():
    if label == "W":
        data = W_show
    elif label == "A":
        data = A_show
    else:
        data = B_show
    ax_inset = ax1.inset_axes([left, bottom, width, height])
    ax_inset.imshow(data, cmap=cmap, norm=norm, aspect="auto")
    ax_inset.set_xticks([])
    ax_inset.set_yticks([])
    shape_str = f"{data.shape[0]}×{data.shape[1]}"
    ax_inset.set_title(f"{label}\n({shape_str})", fontsize=9, fontweight="bold")

ax1.text(0.41, 0.5, "≈", transform=ax1.transAxes, fontsize=20, ha="center", va="center")
ax1.text(0.59, 0.5, "×", transform=ax1.transAxes, fontsize=16, ha="center", va="center")
ax1.axis("off")
ax1.set_title(f"Matrix Decomposition\nΔW({d_show}×{d_show}) ≈ A({d_show}×{r_show}) × B({r_show}×{d_show})", fontsize=11)

# 参数量标注
ax1.text(0.5, 0.02,
         f"Full: {d_show}×{d_show}={d_show**2} params\n"
         f"LoRA (r={r_show}): {d_show}×{r_show}+{r_show}×{d_show}={2*d_show*r_show} params\n"
         f"Reduction: {2*d_show*r_show/(d_show**2)*100:.0f}% of original",
         transform=ax1.transAxes, fontsize=8.5, ha="center", va="bottom",
         bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", alpha=0.8))

# --- 中图：rank vs 近似误差 ---
ax2 = fig.add_subplot(gs[1])
color1 = "#E53935"
ax2.plot(ranks, [e * 100 for e in errors], "o-", color=color1,
         linewidth=2, markersize=7, label="Approximation error (%)")
ax2.fill_between(ranks, [e * 100 for e in errors], alpha=0.15, color=color1)

ax2.axhline(5, color="#999", linestyle="--", linewidth=1, label="5% error threshold")
ax2.set_xlabel("LoRA Rank (r)", fontsize=10)
ax2.set_ylabel("Relative Frobenius Error (%)", fontsize=10)
ax2.set_title("Approximation Quality vs Rank\n(lower rank = fewer params)", fontsize=11)
ax2.legend(fontsize=9)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)
ax2.grid(True, alpha=0.3)

# 标注 r=4 的位置
r4_idx = ranks.index(4)
ax2.annotate(f"r=4: {errors[r4_idx]*100:.1f}% error\n{param_ratios[r4_idx]:.1f}% params",
             xy=(4, errors[r4_idx] * 100),
             xytext=(10, errors[r4_idx] * 100 + 5),
             fontsize=8, color="#2E7D32",
             arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1))

# --- 右图：参数量对比 ---
ax3 = fig.add_subplot(gs[2])

# 实际 LLaMA-7B 的参数量对比
model_configs = [
    ("Full\nFinetune", 7000, "#E53935"),
    ("LoRA\nr=64",     8.4,  "#FF9800"),
    ("LoRA\nr=16",     2.1,  "#FFC107"),
    ("LoRA\nr=8",      1.05, "#4CAF50"),
    ("LoRA\nr=4",      0.52, "#2E7D32"),
]

names = [c[0] for c in model_configs]
params = [c[1] for c in model_configs]
colors = [c[2] for c in model_configs]

bars = ax3.bar(names, params, color=colors, edgecolor="white", linewidth=1)
ax3.set_yscale("log")
ax3.set_ylabel("Trainable Parameters (M, log scale)", fontsize=10)
ax3.set_title("Trainable Parameters\nLLaMA-7B (7B total params)", fontsize=11)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

for bar, val in zip(bars, params):
    label = f"{val:.1f}M" if val < 100 else f"{val:.0f}M"
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
             label, ha="center", fontsize=8.5, fontweight="bold")

ax3.annotate("0.12% of\nfull model",
             xy=(4, 0.52), xytext=(3.2, 0.15),
             fontsize=8, color="#2E7D32",
             arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1))

plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
plt.close()
