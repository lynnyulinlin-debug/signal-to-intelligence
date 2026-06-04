"""
实验5.5a：RLHF 训练流程与 PPO 优化
对应章节：第5章 5.5 强化学习对齐
目标：模拟 RLHF 的奖励模型训练和 PPO 优化过程，
展示对齐前后的输出分布变化
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch05_rlhf_pipeline.png")
N_SAMPLES = 500


def run_experiment(seed=42, n_samples=N_SAMPLES):
    rng = np.random.RandomState(seed)
    pre_align_scores = np.concatenate(
        [
            rng.normal(3.5, 1.2, n_samples // 2),
            rng.normal(1.5, 1.0, n_samples // 2),
        ]
    )
    post_align_scores = np.concatenate(
        [
            rng.normal(5.5, 0.8, int(n_samples * 0.85)),
            rng.normal(2.0, 0.7, int(n_samples * 0.15)),
        ]
    )
    steps = np.arange(0, 1000, 10)
    reward_curve = 2.0 + 3.5 * (1 - np.exp(-steps / 300)) + rng.normal(0, 0.15, len(steps))
    kl_curve = 2.5 * (1 - np.exp(-steps / 200)) + rng.normal(0, 0.08, len(steps))
    tasks = ["Helpfulness", "Harmlessness", "Honesty", "Instruction\nFollowing", "Coding"]
    rlhf_winrate = np.array([72, 68, 65, 78, 61], dtype=float)
    dpo_winrate = np.array([69, 71, 67, 74, 63], dtype=float)

    return {
        "pre_align_scores": pre_align_scores,
        "post_align_scores": post_align_scores,
        "steps": steps,
        "reward_curve": reward_curve,
        "kl_curve": kl_curve,
        "tasks": tasks,
        "rlhf_winrate": rlhf_winrate,
        "dpo_winrate": dpo_winrate,
    }


def print_summary(result):
    print("=" * 70)
    print("RLHF: Reinforcement Learning from Human Feedback")
    print("=" * 70)
    print(f"Pre-align mean: {result['pre_align_scores'].mean():.2f}")
    print(f"Post-align mean: {result['post_align_scores'].mean():.2f}")
    print(f"Final reward: {result['reward_curve'][-1]:.2f}")
    print(f"Final KL: {result['kl_curve'][-1]:.2f}")


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "RLHF: Reinforcement Learning from Human Feedback",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = axes[0]
    bins = np.linspace(-1, 9, 40)
    ax1.hist(
        result["pre_align_scores"],
        bins=bins,
        alpha=0.6,
        color="#E53935",
        label="Before RLHF (SFT)",
        density=True,
        edgecolor="white",
    )
    ax1.hist(
        result["post_align_scores"],
        bins=bins,
        alpha=0.6,
        color="#2E7D32",
        label="After RLHF",
        density=True,
        edgecolor="white",
    )
    ax1.axvline(
        np.mean(result["pre_align_scores"]),
        color="#E53935",
        linestyle="--",
        linewidth=1.5,
        label=f"Pre mean: {np.mean(result['pre_align_scores']):.1f}",
    )
    ax1.axvline(
        np.mean(result["post_align_scores"]),
        color="#2E7D32",
        linestyle="--",
        linewidth=1.5,
        label=f"Post mean: {np.mean(result['post_align_scores']):.1f}",
    )
    ax1.set_xlabel("Reward Model Score", fontsize=10)
    ax1.set_ylabel("Density", fontsize=10)
    ax1.set_title("Reward Distribution\nBefore vs After RLHF", fontsize=11)
    ax1.legend(fontsize=8.5)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    ax2_twin = ax2.twinx()
    line1, = ax2.plot(
        result["steps"],
        result["reward_curve"],
        color="#2196F3",
        linewidth=2,
        label="Avg Reward",
    )
    line2, = ax2_twin.plot(
        result["steps"],
        result["kl_curve"],
        color="#FF9800",
        linewidth=2,
        linestyle="--",
        label="KL Divergence",
    )
    ax2.set_xlabel("PPO Training Steps", fontsize=10)
    ax2.set_ylabel("Average Reward", fontsize=10, color="#2196F3")
    ax2_twin.set_ylabel("KL Divergence from SFT", fontsize=10, color="#FF9800")
    ax2.set_title(
        "PPO Training Dynamics\n(Reward ↑, KL constraint keeps model stable)",
        fontsize=11,
    )
    ax2.tick_params(axis="y", labelcolor="#2196F3")
    ax2_twin.tick_params(axis="y", labelcolor="#FF9800")
    ax2_twin.axhline(2.0, color="#FF9800", linestyle=":", linewidth=1, alpha=0.5)
    ax2_twin.text(800, 2.1, "KL limit", fontsize=8, color="#FF9800", alpha=0.7)
    ax2.legend(
        [line1, line2],
        [line1.get_label(), line2.get_label()],
        fontsize=9,
        loc="lower right",
    )
    ax2.spines["top"].set_visible(False)
    ax2_twin.spines["top"].set_visible(False)

    ax3 = axes[2]
    x = np.arange(len(result["tasks"]))
    width = 0.35
    bars1 = ax3.bar(
        x - width / 2,
        result["rlhf_winrate"],
        width,
        label="RLHF (PPO)",
        color="#1565C0",
        alpha=0.85,
        edgecolor="white",
    )
    bars2 = ax3.bar(
        x + width / 2,
        result["dpo_winrate"],
        width,
        label="DPO",
        color="#6A1B9A",
        alpha=0.85,
        edgecolor="white",
    )
    ax3.axhline(50, color="#999", linestyle="--", linewidth=1, label="50% baseline")
    ax3.set_xticks(x)
    ax3.set_xticklabels(result["tasks"], fontsize=9)
    ax3.set_ylabel("Win Rate vs SFT baseline (%)", fontsize=10)
    ax3.set_ylim(40, 90)
    ax3.set_title("RLHF vs DPO\nWin Rate on Key Dimensions", fontsize=11)
    ax3.legend(fontsize=9)
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for bar in bars1:
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(bar.get_height())}%",
            ha="center",
            fontsize=8,
        )
    for bar in bars2:
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.5,
            f"{int(bar.get_height())}%",
            ha="center",
            fontsize=8,
        )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
