"""
Prompt Engineering Techniques Visualization
Demonstrates Zero-shot / Few-shot / Chain-of-Thought effectiveness
and the iterative prompt optimization workflow.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

OUTPUT_PATH = "assets/ch06_prompt_techniques.png"


def plot_technique_comparison(ax):
    """Bar chart: accuracy of three prompting strategies across task types."""
    tasks = ["Classification", "Math\nReasoning", "Code\nGeneration", "Summarization"]
    zero_shot  = [72, 45, 58, 78]
    few_shot   = [85, 62, 74, 83]
    cot        = [84, 88, 76, 81]

    x = np.arange(len(tasks))
    w = 0.25

    b1 = ax.bar(x - w, zero_shot, w, label="Zero-shot",  color="#4C72B0", alpha=0.85)
    b2 = ax.bar(x,     few_shot,  w, label="Few-shot",   color="#55A868", alpha=0.85)
    b3 = ax.bar(x + w, cot,       w, label="Chain-of-Thought", color="#C44E52", alpha=0.85)

    ax.set_xlabel("Task Type", fontsize=11)
    ax.set_ylabel("Accuracy (%)", fontsize=11)
    ax.set_title("Prompting Strategy vs Task Type", fontsize=13, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(tasks, fontsize=10)
    ax.set_ylim(0, 100)
    ax.legend(fontsize=10)
    ax.grid(axis="y", alpha=0.3)

    # annotate CoT advantage on reasoning
    ax.annotate("+43pts\nvs Zero-shot",
                xy=(x[1] + w, cot[1]), xytext=(x[1] + w + 0.35, cot[1] + 5),
                fontsize=8, color="#C44E52",
                arrowprops=dict(arrowstyle="->", color="#C44E52", lw=1.2))


def plot_cost_vs_effort(ax):
    """Scatter: cost-to-implement vs performance for four optimization methods."""
    methods = ["Zero-shot", "Few-shot", "CoT", "Fine-tuning"]
    effort  = [1,  2,  3,  9]    # relative engineering effort (1-10)
    perf    = [65, 78, 85, 92]   # relative performance (%)
    colors  = ["#4C72B0", "#55A868", "#C44E52", "#8172B2"]
    sizes   = [120, 120, 120, 120]

    for i, (m, e, p, c) in enumerate(zip(methods, effort, perf, colors)):
        ax.scatter(e, p, s=sizes[i], color=c, zorder=3, edgecolors="white", linewidths=1.5)
        offset_x = 0.3 if i < 3 else -1.8
        offset_y = 1.5 if i != 1 else -3.5
        ax.annotate(m, (e, p), xytext=(e + offset_x, p + offset_y), fontsize=10, color=c)

    # "sweet spot" region
    rect = mpatches.FancyBboxPatch((1.5, 74), 2.5, 14,
                                   boxstyle="round,pad=0.3",
                                   linewidth=1.5, edgecolor="#55A868",
                                   facecolor="#55A868", alpha=0.08)
    ax.add_patch(rect)
    ax.text(2.75, 89.5, "Sweet Spot", ha="center", fontsize=9,
            color="#55A868", style="italic")

    ax.set_xlabel("Engineering Effort (relative)", fontsize=11)
    ax.set_ylabel("Performance (%)", fontsize=11)
    ax.set_title("Cost vs Performance Trade-off", fontsize=13, fontweight="bold")
    ax.set_xlim(0, 11)
    ax.set_ylim(55, 98)
    ax.grid(alpha=0.3)


def plot_iteration_workflow(ax):
    """Horizontal flow diagram: iterative prompt optimization loop."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis("off")
    ax.set_title("Iterative Prompt Optimization Workflow", fontsize=13, fontweight="bold")

    steps = [
        (1.0, "Write\nInitial\nPrompt",  "#4C72B0"),
        (3.2, "Test on\n10-20\nSamples", "#55A868"),
        (5.4, "Analyze\nFailure\nCases",  "#C44E52"),
        (7.6, "Refine\nPrompt",          "#8172B2"),
    ]

    for x, label, color in steps:
        circle = plt.Circle((x, 2), 0.75, color=color, alpha=0.85, zorder=3)
        ax.add_patch(circle)
        ax.text(x, 2, label, ha="center", va="center",
                fontsize=8.5, color="white", fontweight="bold", zorder=4)

    # arrows between steps
    for i in range(len(steps) - 1):
        x1 = steps[i][0] + 0.75
        x2 = steps[i + 1][0] - 0.75
        ax.annotate("", xy=(x2, 2), xytext=(x1, 2),
                    arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))

    # feedback loop arrow (back from Refine to Write)
    ax.annotate("", xy=(1.0, 1.1), xytext=(7.6, 1.1),
                arrowprops=dict(arrowstyle="->", lw=1.8, color="#888888",
                                connectionstyle="arc3,rad=0"))
    ax.text(4.3, 0.55, "Iterate until quality target met", ha="center",
            fontsize=9, color="#888888", style="italic")

    # "Done" exit
    ax.annotate("", xy=(9.3, 2), xytext=(8.35, 2),
                arrowprops=dict(arrowstyle="->", lw=2, color="#555555"))
    ax.text(9.6, 2, "Done", ha="center", va="center", fontsize=10,
            color="#555555", fontweight="bold")


def main():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Prompt Engineering: Techniques, Trade-offs & Workflow",
                 fontsize=14, fontweight="bold", y=1.02)

    plot_technique_comparison(axes[0])
    plot_cost_vs_effort(axes[1])
    plot_iteration_workflow(axes[2])

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=150, bbox_inches="tight")
    print(f"Saved: {OUTPUT_PATH}")
    plt.close()


if __name__ == "__main__":
    main()
