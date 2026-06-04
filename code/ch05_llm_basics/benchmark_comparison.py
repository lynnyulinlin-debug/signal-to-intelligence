"""
实验5.6a：LLM 评估基准对比
对应章节：第5章 5.6 模型评估
目标：展示主流 LLM 在标准基准上的性能对比，以及 ELO 评分系统的工作原理
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch05_benchmark_comparison.png")


def run_experiment():
    models = [
        "GPT-4o",
        "Claude-3.5\nSonnet",
        "Gemini\n1.5 Pro",
        "LLaMA-3\n70B",
        "Qwen2.5\n72B",
        "Mistral\nLarge",
    ]
    benchmarks = {
        "MMLU\n(Knowledge)": [88.7, 88.3, 85.9, 82.0, 85.0, 81.2],
        "HumanEval\n(Code)": [90.2, 92.0, 84.1, 81.1, 86.6, 73.2],
        "GSM8K\n(Math)": [95.8, 96.4, 91.7, 93.0, 94.5, 88.7],
        "MT-Bench\n(Dialog)": [9.1, 9.0, 8.9, 8.2, 8.7, 8.1],
    }
    elo_scores = np.array([1314, 1298, 1261, 1207, 1258, 1158], dtype=float)
    colors = ["#1565C0", "#2E7D32", "#E65100", "#6A1B9A", "#00838F", "#795548"]

    normalized = {}
    for bench, scores in benchmarks.items():
        if "MT-Bench" in bench:
            normalized[bench] = np.array(scores, dtype=float) * 10
        else:
            normalized[bench] = np.array(scores, dtype=float)

    sorted_idx = np.argsort(elo_scores)[::-1]
    sorted_models = [models[i].replace("\n", " ") for i in sorted_idx]
    sorted_elo = elo_scores[sorted_idx]
    sorted_colors = [colors[i] for i in sorted_idx]

    return {
        "models": models,
        "benchmarks": benchmarks,
        "normalized": normalized,
        "elo_scores": elo_scores,
        "colors": colors,
        "sorted_models": sorted_models,
        "sorted_elo": sorted_elo,
        "sorted_colors": sorted_colors,
    }


def print_summary(result):
    print("=" * 70)
    print("LLM Evaluation: Benchmark Comparison and ELO Ratings")
    print("=" * 70)
    print(f"Models tracked: {len(result['models'])}")
    print(f"Benchmarks tracked: {len(result['benchmarks'])}")
    print(f"Top ELO model: {result['sorted_models'][0]} ({result['sorted_elo'][0]:.0f})")


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle(
        "LLM Evaluation: Benchmark Comparison and ELO Ratings",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = axes[0]
    bench_names = list(result["normalized"].keys())
    x = np.arange(len(bench_names))
    width = 0.12
    for i, (model, color) in enumerate(zip(result["models"], result["colors"])):
        scores = [result["normalized"][bench][i] for bench in bench_names]
        offset = (i - len(result["models"]) / 2 + 0.5) * width
        ax1.bar(
            x + offset,
            scores,
            width,
            label=model.replace("\n", " "),
            color=color,
            alpha=0.85,
            edgecolor="white",
            linewidth=0.5,
        )
    ax1.set_xticks(x)
    ax1.set_xticklabels(bench_names, fontsize=9.5)
    ax1.set_ylabel("Score (%)", fontsize=10)
    ax1.set_ylim(60, 105)
    ax1.set_title("Benchmark Scores Across Tasks\n(normalized to 0-100%)", fontsize=11)
    ax1.legend(fontsize=8, loc="lower right", ncol=2)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.grid(True, alpha=0.2, axis="y")

    ax2 = axes[1]
    bars = ax2.barh(
        range(len(result["sorted_models"])),
        result["sorted_elo"],
        color=result["sorted_colors"],
        edgecolor="white",
        linewidth=1,
        alpha=0.85,
    )
    ax2.set_yticks(range(len(result["sorted_models"])))
    ax2.set_yticklabels(result["sorted_models"], fontsize=10)
    ax2.set_xlabel("Chatbot Arena ELO Score", fontsize=10)
    ax2.set_title("Chatbot Arena ELO Leaderboard\n(Human preference voting)", fontsize=11)
    ax2.set_xlim(1100, 1360)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar, elo in zip(bars, result["sorted_elo"]):
        ax2.text(
            bar.get_width() + 2,
            bar.get_y() + bar.get_height() / 2,
            str(int(elo)),
            va="center",
            fontsize=9.5,
            fontweight="bold",
        )
    ax2.text(
        0.02,
        0.08,
        "ELO Update Rule:\n"
        "E[A wins] = 1 / (1 + 10^((R_B - R_A)/400))\n"
        "R_A_new = R_A + K × (actual - expected)\n"
        "K=32 (update speed)",
        transform=ax2.transAxes,
        fontsize=8,
        bbox=dict(
            boxstyle="round,pad=0.4",
            facecolor="#FFF9C4",
            edgecolor="#F9A825",
            alpha=0.9,
        ),
        verticalalignment="bottom",
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
