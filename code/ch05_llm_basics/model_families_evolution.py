"""
实验5.3a：主流 LLM 家族演化时间线
对应章节：第5章 5.3 模型家族
目标：可视化 GPT、LLaMA、Qwen、DeepSeek 等主流模型的参数量与发布时间
"""

from pathlib import Path

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch05_model_families_evolution.png")


def run_experiment():
    models = [
        ("GPT-1", 2018.6, 0.117, "GPT", False),
        ("GPT-2", 2019.2, 1.5, "GPT", True),
        ("GPT-3", 2020.5, 175, "GPT", False),
        ("InstructGPT", 2022.1, 175, "GPT", False),
        ("GPT-4", 2023.3, 1000, "GPT", False),
        ("GPT-4o", 2024.5, 200, "GPT", False),
        ("LLaMA-1", 2023.2, 65, "LLaMA", True),
        ("LLaMA-2", 2023.7, 70, "LLaMA", True),
        ("LLaMA-3", 2024.4, 70, "LLaMA", True),
        ("LLaMA-3.1", 2024.7, 405, "LLaMA", True),
        ("Qwen-7B", 2023.9, 7, "Qwen", True),
        ("Qwen-72B", 2023.11, 72, "Qwen", True),
        ("Qwen2.5", 2024.9, 72, "Qwen", True),
        ("DeepSeek-V2", 2024.5, 236, "DeepSeek", True),
        ("DeepSeek-R1", 2025.1, 671, "DeepSeek", True),
        ("Mistral-7B", 2023.9, 7, "Mistral", True),
        ("Mixtral-8x7B", 2023.12, 47, "Mistral", True),
    ]
    family_colors = {
        "GPT": "#1565C0",
        "LLaMA": "#2E7D32",
        "Qwen": "#E65100",
        "DeepSeek": "#6A1B9A",
        "Mistral": "#00838F",
    }

    open_scores = np.array([5, 5, 3, 4, 5], dtype=float)
    closed_scores = np.array([2, 2, 5, 2, 2], dtype=float)
    categories = [
        "Accessibility",
        "Customization",
        "Performance\n(frontier)",
        "Cost\n(inference)",
        "Privacy\nControl",
    ]

    return {
        "models": models,
        "family_colors": family_colors,
        "open_scores": open_scores,
        "closed_scores": closed_scores,
        "categories": categories,
    }


def print_summary(result):
    print("=" * 70)
    print("LLM Family Evolution: Scale and Timeline")
    print("=" * 70)
    print(f"Models tracked: {len(result['models'])}")
    print(f"Families: {sorted(result['family_colors'].keys())}")
    print(f"Open-source score total: {result['open_scores'].sum():.1f}")


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("LLM Family Evolution: Scale and Timeline", fontsize=14, fontweight="bold")

    ax1 = axes[0]
    for name, year, params, family, open_src in result["models"]:
        color = result["family_colors"][family]
        marker = "o" if open_src else "s"
        size = max(30, min(params * 0.8, 600))
        ax1.scatter(
            year,
            np.log10(params + 0.01),
            s=size,
            color=color,
            marker=marker,
            alpha=0.85,
            edgecolors="white",
            linewidth=1,
            zorder=5,
        )
        if params >= 65 or name in {
            "GPT-1",
            "GPT-2",
            "LLaMA-1",
            "Mistral-7B",
            "Qwen-7B",
        }:
            offset_y = 0.15 if params < 10 else -0.25
            ax1.annotate(
                name,
                (year, np.log10(params + 0.01)),
                xytext=(year + 0.05, np.log10(params + 0.01) + offset_y),
                fontsize=7.5,
                color=color,
            )
    ax1.set_xlabel("Year", fontsize=10)
    ax1.set_ylabel("Parameters (log₁₀ scale, B)", fontsize=10)
    ax1.set_title("Model Scale Over Time\n(○=open source, □=closed source)", fontsize=11)
    ax1.set_yticks([np.log10(x) for x in [0.1, 1, 10, 100, 1000]])
    ax1.set_yticklabels(["0.1B", "1B", "10B", "100B", "1000B"])
    ax1.set_xlim(2018, 2025.5)
    ax1.grid(True, alpha=0.3)
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)
    ax1.legend(
        handles=[mpatches.Patch(color=c, label=f) for f, c in result["family_colors"].items()],
        fontsize=9,
        loc="upper left",
    )

    ax2 = axes[1]
    x = np.arange(len(result["categories"]))
    width = 0.35
    bars1 = ax2.bar(
        x - width / 2,
        result["open_scores"],
        width,
        label="Open Source (LLaMA/Qwen)",
        color="#2E7D32",
        alpha=0.85,
        edgecolor="white",
    )
    bars2 = ax2.bar(
        x + width / 2,
        result["closed_scores"],
        width,
        label="Closed Source (GPT-4/Claude)",
        color="#1565C0",
        alpha=0.85,
        edgecolor="white",
    )
    ax2.set_xticks(x)
    ax2.set_xticklabels(result["categories"], fontsize=9)
    ax2.set_ylabel("Score (1-5)", fontsize=10)
    ax2.set_ylim(0, 6.5)
    ax2.set_title(
        "Open vs Closed Source LLMs\nKey Tradeoffs",
        fontsize=11,
    )
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    for bar in bars1:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(int(bar.get_height())),
            ha="center",
            fontsize=9,
            color="#2E7D32",
            fontweight="bold",
        )
    for bar in bars2:
        ax2.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.1,
            str(int(bar.get_height())),
            ha="center",
            fontsize=9,
            color="#1565C0",
            fontweight="bold",
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
