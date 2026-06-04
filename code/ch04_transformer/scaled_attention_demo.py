"""
实验4.2：缩放点积注意力演示
对应章节：第4章 - Transformer详解
目标：展示为什么注意力分数需要除以 sqrt(d_k)
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

NUM_SAMPLES = 4000
DIMS = (8, 32, 128, 512)
OUTPUT_PATH = Path("assets/ch04_scaled_attention.png")


def softmax(logits):
    shifted = logits - np.max(logits, axis=1, keepdims=True)
    exp_logits = np.exp(shifted)
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)


def sample_attention_statistics(d_k, num_samples=NUM_SAMPLES, seed=42):
    rng = np.random.RandomState(seed + d_k)
    q = rng.randn(num_samples, d_k)
    k = rng.randn(num_samples, d_k)

    raw_scores = np.sum(q * k, axis=1)
    scaled_scores = raw_scores / np.sqrt(d_k)

    logits_raw = np.vstack([raw_scores[:50], np.zeros(50)]).T
    logits_scaled = np.vstack([scaled_scores[:50], np.zeros(50)]).T

    probs_raw = softmax(logits_raw)
    probs_scaled = softmax(logits_scaled)

    return {
        "d_k": d_k,
        "raw_score_std": float(np.std(raw_scores)),
        "scaled_score_std": float(np.std(scaled_scores)),
        "raw_softmax_peak": float(np.mean(np.max(probs_raw, axis=1))),
        "scaled_softmax_peak": float(np.mean(np.max(probs_scaled, axis=1))),
        "raw_scores": raw_scores,
        "scaled_scores": scaled_scores,
    }


def run_experiment(seed=42, num_samples=NUM_SAMPLES, dims=DIMS):
    statistics = [
        sample_attention_statistics(d_k, num_samples=num_samples, seed=seed)
        for d_k in dims
    ]
    return {
        "dims": np.array(dims),
        "raw_score_stds": np.array([item["raw_score_std"] for item in statistics]),
        "scaled_score_stds": np.array([item["scaled_score_std"] for item in statistics]),
        "raw_softmax_peaks": np.array([item["raw_softmax_peak"] for item in statistics]),
        "scaled_softmax_peaks": np.array([item["scaled_softmax_peak"] for item in statistics]),
        "statistics": statistics,
    }


def print_summary(result):
    print("=" * 70)
    print("Scaled Dot-Product Attention Demo")
    print("=" * 70)
    for d_k, raw_std, scaled_std, raw_peak, scaled_peak in zip(
        result["dims"],
        result["raw_score_stds"],
        result["scaled_score_stds"],
        result["raw_softmax_peaks"],
        result["scaled_softmax_peaks"],
    ):
        print(f"d_k={d_k:>3}: raw std={raw_std:.4f}, scaled std={scaled_std:.4f},")
        print(f"          raw peak={raw_peak:.4f}, scaled peak={scaled_peak:.4f}")
    print("=" * 70)
    print(
        "Conclusion: as dimension grows, unscaled dot products make softmax too sharp; "
        "dividing by sqrt(d_k) keeps it stable."
    )


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))

    ax = axes[0, 0]
    for item in result["statistics"]:
        ax.hist(item["raw_scores"], bins=50, alpha=0.45, label=f"d_k={item['d_k']}")
    ax.set_title("Raw Dot-Product Score Distribution")
    ax.set_xlabel("Score")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[0, 1]
    for item in result["statistics"]:
        ax.hist(item["scaled_scores"], bins=50, alpha=0.45, label=f"d_k={item['d_k']}")
    ax.set_title("Scaled Score Distribution")
    ax.set_xlabel("Score / sqrt(d_k)")
    ax.set_ylabel("Frequency")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.plot(result["dims"], result["raw_score_stds"], marker="o", label="Raw score std")
    ax.plot(result["dims"], result["scaled_score_stds"], marker="o", label="Scaled score std")
    ax.set_title("Score Standard Deviation vs d_k")
    ax.set_xlabel("d_k")
    ax.set_ylabel("Standard deviation")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(result["dims"], result["raw_softmax_peaks"], marker="o", label="Raw softmax peak")
    ax.plot(result["dims"], result["scaled_softmax_peaks"], marker="o", label="Scaled softmax peak")
    ax.set_title("Average Softmax Peak vs d_k")
    ax.set_xlabel("d_k")
    ax.set_ylabel("Average max probability")
    ax.legend()
    ax.grid(True, alpha=0.3)

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
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
