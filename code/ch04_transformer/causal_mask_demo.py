"""
实验4.3：因果掩码演示
对应章节：第4章 - Transformer详解
目标：对比双向注意力和 causal mask 对信息流的影响
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

SEQ_LEN = 8
D_MODEL = 16
OUTPUT_PATH = Path("assets/ch04_causal_mask.png")


def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


def run_experiment(seed=42, seq_len=SEQ_LEN, d_model=D_MODEL):
    rng = np.random.RandomState(seed)
    x = rng.randn(seq_len, d_model)
    w_q = rng.randn(d_model, d_model) * 0.1
    w_k = rng.randn(d_model, d_model) * 0.1
    w_v = rng.randn(d_model, d_model) * 0.1

    q = x @ w_q
    k = x @ w_k
    v = x @ w_v

    scores = q @ k.T / np.sqrt(d_model)
    bidirectional_weights = softmax(scores)

    causal_mask = np.triu(np.full((seq_len, seq_len), -1e9), k=1)
    causal_weights = softmax(scores + causal_mask)

    bidirectional_output = bidirectional_weights @ v
    causal_output = causal_weights @ v

    bidirectional_top = np.argsort(bidirectional_weights, axis=1)[:, -3:][:, ::-1]
    causal_top = np.argsort(causal_weights, axis=1)[:, -3:][:, ::-1]
    output_diffs = np.linalg.norm(bidirectional_output - causal_output, axis=1)

    return {
        "seq_len": seq_len,
        "d_model": d_model,
        "x": x,
        "scores": scores,
        "bidirectional_weights": bidirectional_weights,
        "causal_mask": causal_mask,
        "causal_weights": causal_weights,
        "bidirectional_output": bidirectional_output,
        "causal_output": causal_output,
        "bidirectional_top": bidirectional_top,
        "causal_top": causal_top,
        "output_diffs": output_diffs,
    }


def print_summary(result):
    print("=" * 70)
    print("Causal Mask Demo")
    print("=" * 70)
    print(f"Sequence length: {result['seq_len']}")
    print(f"Model dimension: {result['d_model']}")
    print()
    print("Bidirectional attention: each position can attend to the full sequence.")
    print("Causal attention: each position can only attend to itself and previous positions.")
    print()

    for i in range(result["seq_len"]):
        print(f"Position {i}: bidirectional top attention {result['bidirectional_top'][i]}")
        print(f"             causal top attention {result['causal_top'][i]}")

    print()
    print("Output difference (L2 norm):")
    for i, diff in enumerate(result["output_diffs"]):
        print(f"Position {i}: {diff:.4f}")


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    ax = axes[0, 0]
    im = ax.imshow(result["scores"], cmap="coolwarm", aspect="auto")
    ax.set_title("Raw Attention Scores")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(result["bidirectional_weights"], cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Bidirectional Attention")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    im = ax.imshow(result["causal_mask"], cmap="gray", aspect="auto")
    ax.set_title("Causal Mask")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    im = ax.imshow(result["causal_weights"], cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_title("Causal Attention")
    ax.set_xlabel("Key Position")
    ax.set_ylabel("Query Position")
    plt.colorbar(im, ax=ax)

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
