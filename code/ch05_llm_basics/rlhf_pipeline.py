"""
实验5.5a：RLHF 训练流程与 PPO 优化
对应章节：第5章 5.5 强化学习对齐
目标：模拟 RLHF 的奖励模型训练和 PPO 优化过程，展示对齐前后的输出分布变化
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = "../../assets/ch05_rlhf_pipeline.png"

np.random.seed(42)

# ============ 模拟数据 ============

# 模拟奖励模型的偏好分数分布
# 假设有两类回答：helpful（高分）和 harmful/unhelpful（低分）
n_samples = 500

# 对齐前：SFT 模型的输出分布（混合了好坏回答）
pre_align_scores = np.concatenate([
    np.random.normal(3.5, 1.2, n_samples // 2),   # 较好的回答
    np.random.normal(1.5, 1.0, n_samples // 2),   # 较差的回答
])

# 对齐后：RLHF 模型的输出分布（向高分集中）
post_align_scores = np.concatenate([
    np.random.normal(5.5, 0.8, int(n_samples * 0.85)),   # 大多数是好回答
    np.random.normal(2.0, 0.7, int(n_samples * 0.15)),   # 少数较差
])

# PPO 训练过程：奖励随迭代步数的变化
steps = np.arange(0, 1000, 10)
# 模拟 PPO 训练曲线（带噪声的上升曲线）
reward_curve = 2.0 + 3.5 * (1 - np.exp(-steps / 300)) + np.random.normal(0, 0.15, len(steps))
kl_curve = 0.0 + 2.5 * (1 - np.exp(-steps / 200)) + np.random.normal(0, 0.08, len(steps))

# DPO vs RLHF 对比：在不同任务上的 win rate
tasks = ["Helpfulness", "Harmlessness", "Honesty", "Instruction\nFollowing", "Coding"]
rlhf_winrate = [72, 68, 65, 78, 61]
dpo_winrate  = [69, 71, 67, 74, 63]

# ============ 绘图 ============
fig, axes = plt.subplots(1, 3, figsize=(16, 6))
fig.suptitle("RLHF: Reinforcement Learning from Human Feedback", fontsize=14, fontweight="bold")

# --- 左图：对齐前后奖励分布对比 ---
ax1 = axes[0]
bins = np.linspace(-1, 9, 40)
ax1.hist(pre_align_scores, bins=bins, alpha=0.6, color="#E53935",
         label="Before RLHF (SFT)", density=True, edgecolor="white")
ax1.hist(post_align_scores, bins=bins, alpha=0.6, color="#2E7D32",
         label="After RLHF", density=True, edgecolor="white")

ax1.axvline(np.mean(pre_align_scores), color="#E53935", linestyle="--", linewidth=1.5,
            label=f"Pre mean: {np.mean(pre_align_scores):.1f}")
ax1.axvline(np.mean(post_align_scores), color="#2E7D32", linestyle="--", linewidth=1.5,
            label=f"Post mean: {np.mean(post_align_scores):.1f}")

ax1.set_xlabel("Reward Model Score", fontsize=10)
ax1.set_ylabel("Density", fontsize=10)
ax1.set_title("Reward Distribution\nBefore vs After RLHF", fontsize=11)
ax1.legend(fontsize=8.5)
ax1.spines["top"].set_visible(False)
ax1.spines["right"].set_visible(False)

# --- 中图：PPO 训练曲线 ---
ax2 = axes[1]
ax2_twin = ax2.twinx()

line1, = ax2.plot(steps, reward_curve, color="#2196F3", linewidth=2, label="Avg Reward")
line2, = ax2_twin.plot(steps, kl_curve, color="#FF9800", linewidth=2,
                        linestyle="--", label="KL Divergence")

ax2.set_xlabel("PPO Training Steps", fontsize=10)
ax2.set_ylabel("Average Reward", fontsize=10, color="#2196F3")
ax2_twin.set_ylabel("KL Divergence from SFT", fontsize=10, color="#FF9800")
ax2.set_title("PPO Training Dynamics\n(Reward ↑, KL constraint keeps model stable)", fontsize=11)
ax2.tick_params(axis="y", labelcolor="#2196F3")
ax2_twin.tick_params(axis="y", labelcolor="#FF9800")

# 标注 KL 约束区域
ax2_twin.axhline(2.0, color="#FF9800", linestyle=":", linewidth=1, alpha=0.5)
ax2_twin.text(800, 2.1, "KL limit", fontsize=8, color="#FF9800", alpha=0.7)

lines = [line1, line2]
labels = [l.get_label() for l in lines]
ax2.legend(lines, labels, fontsize=9, loc="lower right")
ax2.spines["top"].set_visible(False)
ax2_twin.spines["top"].set_visible(False)

# --- 右图：RLHF vs DPO win rate 对比 ---
ax3 = axes[2]
x = np.arange(len(tasks))
width = 0.35

bars1 = ax3.bar(x - width/2, rlhf_winrate, width, label="RLHF (PPO)",
                color="#1565C0", alpha=0.85, edgecolor="white")
bars2 = ax3.bar(x + width/2, dpo_winrate, width, label="DPO",
                color="#6A1B9A", alpha=0.85, edgecolor="white")

ax3.axhline(50, color="#999", linestyle="--", linewidth=1, label="50% baseline")
ax3.set_xticks(x)
ax3.set_xticklabels(tasks, fontsize=9)
ax3.set_ylabel("Win Rate vs SFT baseline (%)", fontsize=10)
ax3.set_ylim(40, 90)
ax3.set_title("RLHF vs DPO\nWin Rate on Key Dimensions", fontsize=11)
ax3.legend(fontsize=9)
ax3.spines["top"].set_visible(False)
ax3.spines["right"].set_visible(False)

for bar in bars1:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{int(bar.get_height())}%", ha="center", fontsize=8)
for bar in bars2:
    ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
             f"{int(bar.get_height())}%", ha="center", fontsize=8)

plt.tight_layout()
plt.savefig(OUTPUT_PATH, dpi=120, bbox_inches="tight")
print(f"Saved: {OUTPUT_PATH}")
plt.close()
