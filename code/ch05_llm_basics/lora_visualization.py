"""
实验5.4a：LoRA 低秩分解可视化
对应章节：第5章 5.4 微调
目标：展示 LoRA 的矩阵分解原理，对比全量微调与 LoRA 的参数量差异
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch05_lora_visualization.png")


def low_rank_approx(w, rank):
    """用 SVD 做低秩近似，模拟 LoRA 的 A×B 分解"""
    u, s, vt = np.linalg.svd(w, full_matrices=False)
    w_approx = u[:, :rank] @ np.diag(s[:rank]) @ vt[:rank, :]
    error = np.linalg.norm(w - w_approx, "fro") / np.linalg.norm(w, "fro")
    return w_approx, float(error)


def run_experiment(seed=42):
    rng = np.random.RandomState(seed)
    d = 64
    w_original = rng.randn(d, d) * 0.1
    rank_true = 4
    a_true = rng.randn(d, rank_true) * 0.3
    b_true = rng.randn(rank_true, d) * 0.3
    w_delta = a_true @ b_true

    ranks = [1, 2, 4, 8, 16, 32]
    errors = []
    param_ratios = []
    for rank in ranks:
        _, err = low_rank_approx(w_delta, rank)
        errors.append(err)
        param_ratios.append(2 * d * rank / (d * d) * 100)

    return {
        "d": d,
        "w_original": w_original,
        "rank_true": rank_true,
        "w_delta": w_delta,
        "ranks": np.array(ranks),
        "errors": np.array(errors),
        "param_ratios": np.array(param_ratios),
    }


def print_summary(result):
    print("=" * 70)
    print("LoRA: Low-Rank Adaptation for Efficient Finetuning")
    print("=" * 70)
    for rank, error, ratio in zip(result["ranks"], result["errors"], result["param_ratios"]):
        print(f"rank={rank:>2}: error={error * 100:.2f}%, params={ratio:.2f}% of full")


def plot_results(result, output_path=OUTPUT_PATH):
    fig = plt.figure(figsize=(16, 7))
    fig.suptitle(
        "LoRA: Low-Rank Adaptation for Efficient Finetuning",
        fontsize=14,
        fontweight="bold",
    )
    gs = plt.GridSpec(1, 3, figure=fig, wspace=0.35)

    ax1 = fig.add_subplot(gs[0])
    d_show = 8
    r_show = 2
    rng = np.random.RandomState(42)
    w_show = rng.randn(d_show, d_show)
    a_show = rng.randn(d_show, r_show)
    b_show = rng.randn(r_show, d_show)
    positions = {
        "W": (0.0, 0.1, 0.35, 0.8),
        "A": (0.45, 0.1, 0.12, 0.8),
        "B": (0.62, 0.35, 0.35, 0.3),
    }
    norm = Normalize(vmin=-2, vmax=2)
    cmap = plt.cm.RdBu_r
    for label, (left, bottom, width, height) in positions.items():
        data = w_show if label == "W" else a_show if label == "A" else b_show
        ax_inset = ax1.inset_axes([left, bottom, width, height])
        ax_inset.imshow(data, cmap=cmap, norm=norm, aspect="auto")
        ax_inset.set_xticks([])
        ax_inset.set_yticks([])
        ax_inset.set_title(
            f"{label}\n({data.shape[0]}×{data.shape[1]})",
            fontsize=9,
            fontweight="bold",
        )
    ax1.text(0.41, 0.5, "≈", transform=ax1.transAxes, fontsize=20, ha="center", va="center")
    ax1.text(0.59, 0.5, "×", transform=ax1.transAxes, fontsize=16, ha="center", va="center")
    ax1.axis("off")
    ax1.set_title(
        (
            f"Matrix Decomposition\nΔW({d_show}×{d_show}) ≈ "
            f"A({d_show}×{r_show}) × B({r_show}×{d_show})"
        ),
        fontsize=11,
    )
    ax1.text(
        0.5,
        0.02,
        f"Full: {d_show}×{d_show}={d_show**2} params\n"
        f"LoRA (r={r_show}): {d_show}×{r_show}+{r_show}×{d_show}={2*d_show*r_show} params\n"
        f"Reduction: {2*d_show*r_show/(d_show**2)*100:.0f}% of original",
        transform=ax1.transAxes,
        fontsize=8.5,
        ha="center",
        va="bottom",
        bbox=dict(boxstyle="round,pad=0.3", facecolor="#E3F2FD", edgecolor="#1565C0", alpha=0.8),
    )

    ax2 = fig.add_subplot(gs[1])
    ax2.plot(
        result["ranks"],
        result["errors"] * 100,
        "o-",
        color="#E53935",
        linewidth=2,
        markersize=7,
        label="Approximation error (%)",
    )
    ax2.fill_between(result["ranks"], result["errors"] * 100, alpha=0.15, color="#E53935")
    ax2.axhline(5, color="#999", linestyle="--", linewidth=1, label="5% error threshold")
    ax2.set_xlabel("LoRA Rank (r)", fontsize=10)
    ax2.set_ylabel("Relative Frobenius Error (%)", fontsize=10)
    ax2.set_title("Approximation Quality vs Rank\n(lower rank = fewer params)", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)
    ax2.grid(True, alpha=0.3)
    r4_idx = int(np.where(result["ranks"] == 4)[0][0])
    ax2.annotate(
        (
            f"r=4: {result['errors'][r4_idx] * 100:.1f}% error\n"
            f"{result['param_ratios'][r4_idx]:.1f}% params"
        ),
        xy=(4, result["errors"][r4_idx] * 100),
        xytext=(10, result["errors"][r4_idx] * 100 + 5),
        fontsize=8,
        color="#2E7D32",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1),
    )

    ax3 = fig.add_subplot(gs[2])
    model_configs = [
        ("Full\nFinetune", 7000, "#E53935"),
        ("LoRA\nr=64", 8.4, "#FF9800"),
        ("LoRA\nr=16", 2.1, "#FFC107"),
        ("LoRA\nr=8", 1.05, "#4CAF50"),
        ("LoRA\nr=4", 0.52, "#2E7D32"),
    ]
    names = [cfg[0] for cfg in model_configs]
    params = [cfg[1] for cfg in model_configs]
    colors = [cfg[2] for cfg in model_configs]
    bars = ax3.bar(names, params, color=colors, edgecolor="white", linewidth=1)
    ax3.set_yscale("log")
    ax3.set_ylabel("Trainable Parameters (M, log scale)", fontsize=10)
    ax3.set_title(
        "Trainable Parameters\nLLaMA-7B (7B total params)",
        fontsize=11,
    )
    ax3.spines["top"].set_visible(False)
    ax3.spines["right"].set_visible(False)
    for bar, val in zip(bars, params):
        label = f"{val:.1f}M" if val < 100 else f"{val:.0f}M"
        ax3.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() * 1.3,
            label,
            ha="center",
            fontsize=8.5,
            fontweight="bold",
        )
    ax3.annotate(
        "0.12% of\nfull model",
        xy=(4, 0.52),
        xytext=(3.2, 0.15),
        fontsize=8,
        color="#2E7D32",
        arrowprops=dict(arrowstyle="->", color="#2E7D32", lw=1),
    )

    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
