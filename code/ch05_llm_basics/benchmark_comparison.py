"""
实验5.6a：LLM 评估基准对比
对应章节：第5章 5.6 模型评估
目标：展示主流 LLM 在标准基准上的性能对比，以及 ELO 评分系统的工作原理
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "../../assets/ch05_benchmark_comparison.png"

np.random.seed(42)

# ============ 基准数据（近似值，来自公开排行榜）============

models = ["GPT-4o", "Claude-3.5\nSonnet", "Gemini\n1.5 Pro", "LLaMA-3\n70B", "Qwen2.5\n72B", "Mistral\nLarge"]

# 各基准分数（近似）
benchmarks = {
    "MMLU\n(Knowledge)":  [88.7, 88.3, 85.9, 82.0, 85.0, 81.2],
    "HumanEval\n(Code)":  [90.2, 92.0, 84.1, 81.1, 86.6, 73.2],
    "GSM8K\n(Math)":      [95.8, 96.4, 91.7, 93.0, 94.5, 88.7],
    "MT-Bench\n(Dialog)": [9.1,  9.0,  8.9,  8.2,  8.7,  8.1],
}

# Chatbot Arena ELO 分数（近似）
elo_scores = [1314, 1298, 1261, 1207, 1258, 1158]

# ============ 绘图 ============
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
fig.suptitle("LLM Evaluation: Benchmark Comparison and ELO Ratings", fontsize=14, fontweight="bold")

# --- 左图：雷达图（多基准对比）---
ax1 = axes[0]

# 归一化分数到 0-100（MT-Bench 原始是 0-10）
normalized = {}
for bench, scores in benchmarks.items():
    if "MT-Bench" in bench:
        normalized[bench] = [s * 10 for s in scores]
    else:
        normalized[bench] = scores

bench_names = list(normalized.keys())
n_bench = len(bench_names)
n_models = len(models)

x = np.arange(n_bench)
width = 0.12
colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00838F", "#795548"]

for i, (model, color) in enumerate(zip(models, colors)):
    scores = [normalized[b][i] for b in bench_names]
    offset = (i - n_models / 2 + 0.5) * width
    bars = ax1.bar(x + offset, scores, width, label=model.replace("\n", " "),
                   color=color, alpha=0.85, edgecolor="white", linewidth=0.5)

ax1.set_xticks(x)
ax1.set_xticklabels(bench_names, fontsize=9.5)
ax1.set_ylabel("Score (%)", fontsize=10)
ax1.set_ylim(60, 105)
ax1.set_title("Benchmark Scores Across Tasks\n(normalized to 0-100%)", fontsize=11)
ax1.legend(fontsize=8, loc="lower right", ncol=2)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)
ax1.grid(True, alpha=0.2, axis="y")

# --- 右图：ELO 评分 + ELO 原理示意 ---
ax2 = axes[1]

# 主图：ELO 排行榜
sorted_idx = np.argsort(elo_scores)[::-1]
sorted_models = [models[i].replace("\n", " ") for i in sorted_idx]
sorted_elo = [elo_scores[i] for i in sorted_idx]
sorted_colors = [colors[i] for i in sorted_idx]

bars = ax2.barh(range(len(sorted_models)), sorted_elo, color=sorted_colors,
                edgecolor="white", linewidth=1, alpha=0.85)
ax2.set_yticks(range(len(sorted_models)))
ax2.set_yticklabels(sorted_models, fontsize=10)
ax2.set_xlabel("Chatbot Arena ELO Score", fontsize=10)
ax2.set_title("Chatbot Arena ELO Leaderboard\n(Human preference voting)", fontsize=11)
ax2.set_xlim(1100, 1360)
ax2.spines["top"].set_visible(False)
ax2.spines["right"].set_visible(False)

for bar, elo in zip(bars, sorted_elo):
    ax2.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
             str(elo), va="center", fontsize=9.5, fontweight="bold")

# ELO 公式注释框
ax2.text(0.02, 0.08,
         "ELO Update Rule:\n"
         "E[A wins] = 1 / (1 + 10^((R_B - R_A)/400))\n"
         "R_A_new = R_A + K × (actual - expected)\n"
         "K=32 (update speed)",
         transform=ax2.transAxes, fontsize=8,
         bbox=dict(boxstyle="round,pad=0.4", facecolor="#FFF9C4", edgecolor="#F9A825", alpha=0.9),
         verticalalignment="bottom")

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
plt.close()
