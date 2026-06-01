"""
实验5.1c：自回归生成过程演示
对应章节：第5章 5.1 预训练
目标：展示自回归生成中每步的 token 概率分布，以及不同采样策略的效果
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
OUTPUT_PATH = "assets/ch05_autoregressive_generation.png"

# ============ 模拟 token 概率分布 ============

# 模拟生成 "The cat sat on the" 后的下一个 token 概率
tokens = ["mat", "floor", "chair", "table", "roof",
          "bed", "ground", "sofa", "wall", "door"]
# 模拟 softmax 输出（temperature=1.0）
logits = np.array([3.2, 2.8, 2.1, 1.9, 0.8, 0.7, 0.6, 0.4, 0.2, 0.1])
probs_t1 = np.exp(logits) / np.exp(logits).sum()

def apply_temperature(logits, temperature):
    scaled = logits / temperature
    return np.exp(scaled) / np.exp(scaled).sum()

def top_k_filter(probs, k):
    filtered = np.zeros_like(probs)
    top_k_idx = np.argsort(probs)[-k:]
    filtered[top_k_idx] = probs[top_k_idx]
    filtered /= filtered.sum()
    return filtered

def top_p_filter(probs, p):
    sorted_idx = np.argsort(probs)[::-1]
    sorted_probs = probs[sorted_idx]
    cumsum = np.cumsum(sorted_probs)
    cutoff = np.searchsorted(cumsum, p) + 1
    filtered = np.zeros_like(probs)
    filtered[sorted_idx[:cutoff]] = probs[sorted_idx[:cutoff]]
    filtered /= filtered.sum()
    return filtered

probs_t02 = apply_temperature(logits, 0.2)   # 低温：更确定
probs_t10 = apply_temperature(logits, 1.0)   # 标准
probs_t20 = apply_temperature(logits, 2.0)   # 高温：更随机
probs_topk = top_k_filter(probs_t10, k=3)
probs_topp = top_p_filter(probs_t10, p=0.9)

# ============ 绘图 ============
fig = plt.figure(figsize=(15, 8))
fig.suptitle('Autoregressive Generation: Token Probability Distributions\n'
             'Context: "The cat sat on the ___"', fontsize=13, fontweight="bold")

gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.45, wspace=0.35)

configs = [
    (probs_t02, "Temperature = 0.2\n(Greedy-like)", "#1565C0"),
    (probs_t10, "Temperature = 1.0\n(Standard)", "#2E7D32"),
    (probs_t20, "Temperature = 2.0\n(More Random)", "#B71C1C"),
    (probs_topk, "Top-k (k=3)\nOnly top 3 tokens", "#6A1B9A"),
    (probs_topp, "Top-p (p=0.9)\nNucleus sampling", "#E65100"),
]

for idx, (probs, title, color) in enumerate(configs):
    row, col = divmod(idx, 3)
    ax = fig.add_subplot(gs[row, col])

    colors = [color if p > 0 else "#EEEEEE" for p in probs]
    bars = ax.bar(range(len(tokens)), probs * 100, color=colors,
                  edgecolor="white", linewidth=0.5)

    # 标注最高概率
    max_idx = np.argmax(probs)
    ax.bar(max_idx, probs[max_idx] * 100, color=color,
           edgecolor="#333", linewidth=1.5)
    ax.text(max_idx, probs[max_idx] * 100 + 0.5,
            f"{probs[max_idx]*100:.1f}%", ha="center", fontsize=8, fontweight="bold")

    ax.set_xticks(range(len(tokens)))
    ax.set_xticklabels(tokens, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("Probability (%)", fontsize=8)
    ax.set_title(title, fontsize=9.5, fontweight="bold")
    ax.set_ylim(0, max(probs) * 100 * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

# 第6格：自回归步骤示意
ax6 = fig.add_subplot(gs[1, 2])
ax6.axis("off")
steps = [
    ('Step 1', '"The"', '#E3F2FD'),
    ('Step 2', '"The cat"', '#BBDEFB'),
    ('Step 3', '"The cat sat"', '#90CAF9'),
    ('Step 4', '"The cat sat on"', '#64B5F6'),
    ('Step 5', '"The cat sat on the"', '#42A5F5'),
    ('Step 6', '"The cat sat on the mat"', '#1E88E5'),
]
for i, (step, text, color) in enumerate(steps):
    y = 0.88 - i * 0.14
    ax6.add_patch(plt.Rectangle((0.02, y - 0.06), 0.96, 0.11,
                                  facecolor=color, edgecolor="white",
                                  linewidth=1, transform=ax6.transAxes))
    ax6.text(0.06, y, step, transform=ax6.transAxes,
             fontsize=8, color="white", fontweight="bold", va="center")
    ax6.text(0.28, y, text, transform=ax6.transAxes,
             fontsize=8, color="white", va="center")
    if i < len(steps) - 1:
        ax6.annotate("", xy=(0.5, y - 0.065), xytext=(0.5, y - 0.03),
                     xycoords="axes fraction", textcoords="axes fraction",
                     arrowprops=dict(arrowstyle="->", color="#555", lw=1.2))

ax6.set_title("Autoregressive Steps", fontsize=9.5, fontweight="bold")

plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
plt.close()
