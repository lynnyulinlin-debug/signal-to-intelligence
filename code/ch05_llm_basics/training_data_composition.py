"""
实验5.2a：训练数据构成分析
对应章节：第5章 5.2 训练数据
目标：展示主流 LLM 训练数据的来源构成和各阶段数据配比
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "assets/ch05_training_data_composition.png"

# ============ 数据 ============

# GPT-3 预训练数据构成（近似，来自 Brown et al. 2020）
pretrain_labels = ["Common Crawl\n(filtered)", "WebText2", "Books1", "Books2", "Wikipedia"]
pretrain_sizes  = [410, 19, 12, 55, 3]   # GB（近似）
pretrain_colors = ["#1565C0", "#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB"]

# 各训练阶段数据量对比（近似）
stages = ["Pretraining\n(GPT-3)", "SFT\n(InstructGPT)", "RLHF\n(InstructGPT)", "DPO\n(Zephyr)"]
data_sizes = [499, 0.077, 0.033, 0.2]   # GB
stage_colors = ["#1565C0", "#2E7D32", "#F57F17", "#6A1B9A"]

# 数据质量 vs 数量（Phi 系列 vs 传统大模型）
models = ["GPT-3\n175B", "LLaMA-1\n65B", "Phi-1\n1.3B", "Phi-1.5\n1.3B", "Phi-2\n2.7B"]
train_tokens = [300, 1400, 7, 30, 250]   # 十亿 tokens
mmlu_scores  = [43.9, 63.4, 50.9, 55.5, 70.8]   # MMLU 准确率 %

# ============ 绘图 ============
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("LLM Training Data: Composition, Scale, and Quality", fontsize=14, fontweight="bold")

# --- 左图：预训练数据构成饼图 ---
ax1 = axes[0]
total = sum(pretrain_sizes)
pcts = [s / total * 100 for s in pretrain_sizes]
wedges, texts, autotexts = ax1.pie(
    pretrain_sizes, labels=None, colors=pretrain_colors,
    autopct="%1.1f%%", startangle=140,
    pctdistance=0.75, wedgeprops=dict(edgecolor="white", linewidth=1.5)
)
for at in autotexts:
    at.set_fontsize(8.5)
ax1.legend(
    [mpatches.Patch(color=c) for c in pretrain_colors],
    [f"{l.replace(chr(10), ' ')} ({s}GB)" for l, s in zip(pretrain_labels, pretrain_sizes)],
    loc="lower center", bbox_to_anchor=(0.5, -0.22), fontsize=8, ncol=1
)
ax1.set_title("GPT-3 Pretraining Data\nComposition (~499 GB)", fontsize=11)

# --- 中图：各阶段数据量对比（对数坐标）---
ax2 = axes[1]
bars = ax2.bar(stages, data_sizes, color=stage_colors, edgecolor="white", linewidth=1)
ax2.set_yscale("log")
ax2.set_ylabel("Data Size (GB, log scale)", fontsize=10)
ax2.set_title("Data Volume by Training Stage\n(Log Scale)", fontsize=11)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for bar, val in zip(bars, data_sizes):
    label = f"{val}GB" if val >= 1 else f"{val*1000:.0f}MB"
    ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() * 1.3,
             label, ha="center", fontsize=9, fontweight="bold")

ax2.annotate("Pretraining needs\nhundreds of GB",
             xy=(0, 499), xytext=(1.2, 200),
             fontsize=8, color="#1565C0",
             arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1))
ax2.annotate("Alignment needs\nonly ~MB of\nhigh-quality data",
             xy=(2, 0.033), xytext=(2.5, 0.005),
             fontsize=8, color="#F57F17",
             arrowprops=dict(arrowstyle="->", color="#F57F17", lw=1))

# --- 右图：数据质量 vs 数量（Phi 效率）---
ax3 = axes[2]
scatter_colors = ["#1565C0", "#1976D2", "#E53935", "#E53935", "#E53935"]
sizes = [200, 180, 120, 120, 140]

for i, (model, tokens, score, color, sz) in enumerate(
        zip(models, train_tokens, mmlu_scores, scatter_colors, sizes)):
    ax3.scatter(tokens, score, s=sz, color=color, zorder=5,
                marker="o" if color == "#1565C0" or color == "#1976D2" else "^")
    ax3.annotate(model, (tokens, score),
                 xytext=(tokens + 15, score + 0.5),
                 fontsize=8, color=color)

ax3.set_xlabel("Training Tokens (Billions)", fontsize=10)
ax3.set_ylabel("MMLU Score (%)", fontsize=10)
ax3.set_title("Data Quality vs Quantity\n(Phi series: small model, high quality data)", fontsize=11)
ax3.set_xlim(-50, 1600)
ax3.set_ylim(35, 80)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)
ax3.grid(True, alpha=0.3)

legend_elements = [
    mpatches.Patch(color="#1565C0", label="Traditional LLMs (large data)"),
    mpatches.Patch(color="#E53935", label="Phi series (quality data)"),
]
ax3.legend(handles=legend_elements, fontsize=8, loc="lower right")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
plt.close()
