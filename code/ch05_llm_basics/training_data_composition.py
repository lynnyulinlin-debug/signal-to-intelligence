"""
实验5.2a：训练数据构成分析
对应章节：第5章 5.2 训练数据
目标：展示主流 LLM 训练数据的来源构成和各阶段数据配比
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch05_training_data_composition.png")


def run_experiment():
    pretrain_labels = [
        "Common Crawl\n(filtered)",
        "WebText2",
        "Books1",
        "Books2",
        "Wikipedia",
    ]
    pretrain_sizes = np.array([410, 19, 12, 55, 3], dtype=float)
    pretrain_colors = ["#1565C0", "#1976D2", "#42A5F5", "#90CAF9", "#BBDEFB"]

    stages = ["Pretraining\n(GPT-3)", "SFT\n(InstructGPT)", "RLHF\n(InstructGPT)", "DPO\n(Zephyr)"]
    data_sizes = np.array([499, 0.077, 0.033, 0.2], dtype=float)
    stage_colors = ["#1565C0", "#2E7D32", "#F57F17", "#6A1B9A"]

    models = ["GPT-3\n175B", "LLaMA-1\n65B", "Phi-1\n1.3B", "Phi-1.5\n1.3B", "Phi-2\n2.7B"]
    train_tokens = np.array([300, 1400, 7, 30, 250], dtype=float)
    mmlu_scores = np.array([43.9, 63.4, 50.9, 55.5, 70.8], dtype=float)

    return {
        "pretrain_labels": pretrain_labels,
        "pretrain_sizes": pretrain_sizes,
        "pretrain_colors": pretrain_colors,
        "stages": stages,
        "data_sizes": data_sizes,
        "stage_colors": stage_colors,
        "models": models,
        "train_tokens": train_tokens,
        "mmlu_scores": mmlu_scores,
    }


def print_summary(result):
    print("=" * 70)
    print("LLM Training Data: Composition, Scale, and Quality")
    print("=" * 70)
    print(f"Pretraining total: {result['pretrain_sizes'].sum():.1f} GB")
    print(f"Stage data sizes: {result['data_sizes'].tolist()}")
    print(f"Best MMLU score: {result['mmlu_scores'].max():.1f}")


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, 3, figsize=(16, 6))
    fig.suptitle(
        "LLM Training Data: Composition, Scale, and Quality",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = axes[0]
    wedges, texts, autotexts = ax1.pie(
        result["pretrain_sizes"],
        labels=None,
        colors=result["pretrain_colors"],
        autopct="%1.1f%%",
        startangle=140,
        pctdistance=0.75,
        wedgeprops=dict(edgecolor="white", linewidth=1.5),
    )
    for at in autotexts:
        at.set_fontsize(8.5)
    ax1.legend(
        [mpatches.Patch(color=c) for c in result["pretrain_colors"]],
        [
            f"{label.replace(chr(10), ' ')} ({size}GB)"
            for label, size in zip(result["pretrain_labels"], result["pretrain_sizes"])
        ],
        loc="lower center",
        bbox_to_anchor=(0.5, -0.22),
        fontsize=8,
        ncol=1,
    )
    ax1.set_title("GPT-3 Pretraining Data\nComposition (~499 GB)", fontsize=11)

    ax2 = axes[1]
    bars = ax2.bar(
        result["stages"],
        result["data_sizes"],
        color=result["stage_colors"],
        edgecolor="white",
        linewidth=1,
    )
    ax2.set_yscale("log")
    ax2.set_ylabel("Data Size (GB, log scale)", fontsize=10)
    ax2.set_title("Data Volume by Training Stage\n(Log Scale)", fontsize=11)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, val in zip(bars, result["data_sizes"]):
        label = f"{val}GB" if val >= 1 else f"{val * 1000:.0f}MB"
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            label,
            ha="center",
            fontsize=9,
            fontweight="bold",
        )
    ax2.annotate(
        "Pretraining needs\nhundreds of GB",
        xy=(0, 499),
        xytext=(1.2, 200),
        fontsize=8,
        color="#1565C0",
        arrowprops=dict(arrowstyle="->", color="#1565C0", lw=1),
    )
    ax2.annotate(
        "Alignment needs\nonly ~MB of\nhigh-quality data",
        xy=(2, 0.033),
        xytext=(2.5, 0.005),
        fontsize=8,
        color="#F57F17",
        arrowprops=dict(arrowstyle="->", color="#F57F17", lw=1),
    )

    ax3 = axes[2]
    scatter_colors = ["#1565C0", "#1976D2", "#E53935", "#E53935", "#E53935"]
    sizes = [200, 180, 120, 120, 140]
    for model, tokens, score, color, size in zip(
        result["models"],
        result["train_tokens"],
        result["mmlu_scores"],
        scatter_colors,
        sizes,
    ):
        ax3.scatter(
            tokens,
            score,
            s=size,
            color=color,
            zorder=5,
            marker="o" if color in {"#1565C0", "#1976D2"} else "^",
        )
        ax3.annotate(
            model,
            (tokens, score),
            xytext=(tokens + 15, score + 0.5),
            fontsize=8,
            color=color,
        )
    ax3.set_xlabel("Training Tokens (Billions)", fontsize=10)
    ax3.set_ylabel("MMLU Score (%)", fontsize=10)
    ax3.set_title(
        "Data Quality vs Quantity\n(Phi series: small model, high quality data)",
        fontsize=11,
    )
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
