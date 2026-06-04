"""
多模态架构说明图：CNN vs ViT 感受野对比 + 温度参数效果

生成：
  assets/ch07_vit_cnn_comparison.png
  assets/ch07_temperature_effect.png

运行方式：
    python code/ch07_multimodal_llm/architecture_diagrams.py
"""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np

OUTPUT_VIT_CNN = Path("assets/ch07_vit_cnn_comparison.png")
OUTPUT_TEMPERATURE = Path("assets/ch07_temperature_effect.png")


def build_vit_cnn_figure():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle("CNN vs ViT: Receptive Field Comparison", fontsize=14, fontweight="bold")

    ax = axes[0]
    ax.set_title("CNN: Local Receptive Field", fontsize=12)
    ax.set_xlim(-0.1, 5.1)
    ax.set_ylim(-0.1, 5.1)
    ax.set_aspect("equal")
    ax.axis("off")

    for i in range(6):
        ax.plot([0, 5], [i, i], color="#cccccc", linewidth=0.8)
        ax.plot([i, i], [0, 5], color="#cccccc", linewidth=0.8)

    for r in range(5):
        for c in range(5):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (c + 0.02, r + 0.02),
                    0.96,
                    0.96,
                    boxstyle="square,pad=0",
                    facecolor="#e3f2fd",
                    edgecolor="none",
                )
            )

    for r in range(1, 4):
        for c in range(1, 4):
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (c + 0.02, r + 0.02),
                    0.96,
                    0.96,
                    boxstyle="square,pad=0",
                    facecolor="#ef9a9a",
                    edgecolor="none",
                )
            )

    ax.add_patch(mpatches.Rectangle((1, 1), 3, 3, linewidth=2.5, edgecolor="#c62828", facecolor="none"))
    ax.text(2.5, 2.5, "3×3\nkernel", ha="center", va="center", fontsize=10, fontweight="bold", color="#b71c1c")
    ax.annotate(
        "Only 9 pixels\nvisible at once",
        xy=(4.0, 1.5),
        xytext=(3.3, 0.2),
        fontsize=9,
        ha="center",
        color="#c62828",
        arrowprops=dict(arrowstyle="->", color="#c62828", lw=1.2),
    )
    ax.set_xlabel("Pixels", fontsize=10, labelpad=4)
    ax.text(
        2.5,
        -0.35,
        "Each output sees only a small local region",
        ha="center",
        fontsize=9,
        color="gray",
        style="italic",
    )

    ax = axes[1]
    ax.set_title("ViT: Global Self-Attention", fontsize=12)
    ax.set_xlim(-0.1, 4.1)
    ax.set_ylim(-0.6, 4.1)
    ax.set_aspect("equal")
    ax.axis("off")

    cmap = plt.cm.Blues
    for r in range(4):
        for c in range(4):
            color = cmap(0.25 + 0.55 * (r * 4 + c) / 15)
            ax.add_patch(
                mpatches.FancyBboxPatch(
                    (c + 0.04, r + 0.04),
                    0.92,
                    0.92,
                    boxstyle="round,pad=0.03",
                    facecolor=color,
                    edgecolor="white",
                    linewidth=1.5,
                )
            )
            ax.text(
                c + 0.5,
                r + 0.5,
                f"P{r*4+c+1}",
                ha="center",
                va="center",
                fontsize=7,
                color="white",
                fontweight="bold",
            )

    qr, qc = 1, 1
    qx, qy = qc + 0.5, qr + 0.5
    for r in range(4):
        for c in range(4):
            if r == qr and c == qc:
                continue
            tx, ty = c + 0.5, r + 0.5
            ax.annotate(
                "",
                xy=(tx, ty),
                xytext=(qx, qy),
                arrowprops=dict(arrowstyle="->", color="#1565c0", alpha=0.35, lw=0.9),
            )

    ax.add_patch(
        mpatches.FancyBboxPatch(
            (qc + 0.04, qr + 0.04),
            0.92,
            0.92,
            boxstyle="round,pad=0.03",
            facecolor="#e53935",
            edgecolor="#b71c1c",
            linewidth=2.5,
        )
    )
    ax.text(qc + 0.5, qr + 0.5, "Q", ha="center", va="center", fontsize=11, fontweight="bold", color="white")
    ax.text(
        2.0,
        -0.45,
        "Each patch attends to ALL other patches",
        ha="center",
        fontsize=9,
        color="#1565c0",
        style="italic",
    )
    ax.set_xlabel("16×16 px patches", fontsize=10, labelpad=4)

    plt.tight_layout()
    return fig


def build_temperature_figure():
    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle("Effect of Temperature τ on Softmax Distribution", fontsize=13, fontweight="bold")

    taus = [0.07, 0.5, 10.0]
    subtitles = ["τ = 0.07  (too small)", "τ = 0.5  (CLIP default ~0.07)", "τ = 10.0  (too large)"]
    notes = ["Sharp → unstable gradient", "Balanced signal", "Flat → weak learning signal"]
    note_colors = ["#c62828", "#2e7d32", "#1565c0"]

    sims = np.array([0.90, 0.35, 0.25, 0.20, 0.30])
    xlabels = ["Correct", "Neg 1", "Neg 2", "Neg 3", "Neg 4"]
    bar_colors = ["#43a047", "#e57373", "#e57373", "#e57373", "#e57373"]

    for i, (tau, subtitle, note, nc) in enumerate(zip(taus, subtitles, notes, note_colors)):
        ax = axes[i]
        logits = sims / tau
        logits -= logits.max()
        probs = np.exp(logits)
        probs /= probs.sum()

        bars = ax.bar(xlabels, probs, color=bar_colors, edgecolor="white", linewidth=0.5, width=0.6)
        bars[0].set_edgecolor("#1b5e20")
        bars[0].set_linewidth(1.5)
        for bar, prob in zip(bars, probs):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height() + 0.005,
                f"{prob:.2f}",
                ha="center",
                va="bottom",
                fontsize=8,
            )
        ax.set_title(subtitle, fontsize=10, fontweight="bold")
        ax.set_ylim(0, 1.15)
        ax.set_ylabel("Probability" if i == 0 else "", fontsize=10)
        ax.tick_params(axis="x", labelsize=8.5)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            0.5,
            0.97,
            note,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=8.5,
            color=nc,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="#fffde7", alpha=0.9),
        )

    plt.tight_layout()
    return fig


def run_experiment():
    return {
        "vit_cnn_path": OUTPUT_VIT_CNN,
        "temperature_path": OUTPUT_TEMPERATURE,
    }


def plot_results(result, output_vit_cnn=OUTPUT_VIT_CNN, output_temperature=OUTPUT_TEMPERATURE):
    fig1 = build_vit_cnn_figure()
    output_vit_cnn = Path(output_vit_cnn)
    output_vit_cnn.parent.mkdir(parents=True, exist_ok=True)
    fig1.savefig(output_vit_cnn, dpi=100, bbox_inches="tight")
    plt.close(fig1)
    print(f"Saved {output_vit_cnn}")

    fig2 = build_temperature_figure()
    output_temperature = Path(output_temperature)
    output_temperature.parent.mkdir(parents=True, exist_ok=True)
    fig2.savefig(output_temperature, dpi=100, bbox_inches="tight")
    plt.close(fig2)
    print(f"Saved {output_temperature}")
    return {"vit_cnn_path": output_vit_cnn, "temperature_path": output_temperature}


def main():
    result = run_experiment()
    plot_results(result)


if __name__ == "__main__":
    main()
