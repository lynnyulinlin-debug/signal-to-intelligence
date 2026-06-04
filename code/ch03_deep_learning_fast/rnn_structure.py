"""
实验3.2：RNN结构与序列处理
对应章节：第3章 - 深度学习快速通道
目标：展示RNN如何处理序列数据，隐状态的演化过程
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 8
INPUT_SIZE = 3
OUTPUT_SIZE = 1
OUTPUT_PATH = Path("assets/ch03_rnn_structure.png")


def generate_sequence_data(sequence_length=SEQUENCE_LENGTH, input_size=INPUT_SIZE, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(sequence_length, input_size)
    w_true = rng.randn(input_size)
    y = np.array([np.tanh(x[t] @ w_true) for t in range(sequence_length)])
    return x, y


def initialize_rnn_weights(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    output_size=OUTPUT_SIZE,
    seed=42,
):
    rng = np.random.RandomState(seed + 1)
    return {
        "W_h": rng.randn(hidden_size, hidden_size) * 0.1,
        "W_x": rng.randn(input_size, hidden_size) * 0.1,
        "W_y": rng.randn(hidden_size, output_size) * 0.1,
        "b_h": np.zeros((1, hidden_size)),
        "b_y": np.zeros((1, output_size)),
    }


def run_rnn_forward(x, weights):
    """运行简单 RNN 前向传播，返回预测和所有隐状态。"""
    h_states = []
    h = np.zeros((1, weights["W_h"].shape[0]))

    for t in range(len(x)):
        x_t = x[t : t + 1]
        h = np.tanh(x_t @ weights["W_x"] + h @ weights["W_h"] + weights["b_h"])
        h_states.append(h.copy())

    h_states = np.array(h_states).squeeze()
    y_pred = h_states @ weights["W_y"] + weights["b_y"]
    return y_pred.flatten(), h_states


def adjacent_hidden_similarities(h_states):
    similarities = []
    for t in range(1, len(h_states)):
        sim = np.dot(h_states[t], h_states[t - 1]) / (
            np.linalg.norm(h_states[t]) * np.linalg.norm(h_states[t - 1]) + 1e-8
        )
        similarities.append(sim)
    return np.array(similarities)


def run_experiment(seed=42):
    x, y = generate_sequence_data(seed=seed)
    weights = initialize_rnn_weights(seed=seed)
    y_pred, h_states = run_rnn_forward(x, weights)

    mse = np.mean((y - y_pred) ** 2)
    correlation = np.corrcoef(y, y_pred)[0, 1]
    h_norms = np.linalg.norm(h_states, axis=1)
    h_similarities = adjacent_hidden_similarities(h_states)
    h_proj = h_states[:, :2]

    return {
        "X": x,
        "y": y,
        "weights": weights,
        "y_pred": y_pred,
        "h_states": h_states,
        "mse": mse,
        "correlation": correlation,
        "h_norms": h_norms,
        "h_similarities": h_similarities,
        "h_proj": h_proj,
    }


def print_summary(result):
    weights = result["weights"]
    print("=" * 70)
    print("RNN Structure and Sequence Processing")
    print("=" * 70)
    print(f"序列长度: {len(result['X'])}")
    print(f"输入维度: {result['X'].shape[1]}")
    print(f"隐层大小: {weights['W_h'].shape[0]}")
    print(f"输出维度: {weights['W_y'].shape[1]}")
    print()

    print("网络结构:")
    print("-" * 70)
    print(f"W_h (隐->隐): {weights['W_h'].shape}")
    print(f"W_x (输入->隐): {weights['W_x'].shape}")
    print(f"W_y (隐->输出): {weights['W_y'].shape}")
    print()

    print("性能指标:")
    print("-" * 70)
    print(f"预测MSE: {result['mse']:.6f}")
    print(f"预测相关系数: {result['correlation']:.4f}")
    print()
    print("=" * 70)


def plot_rnn_structure(result, output_path=OUTPUT_PATH):
    y = result["y"]
    y_pred = result["y_pred"]
    h_states = result["h_states"]
    h_norms = result["h_norms"]
    h_similarities = result["h_similarities"]
    h_proj = result["h_proj"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

    ax = fig.add_subplot(gs[0, :2])
    ax.plot(y, "k-", linewidth=2, label="True Sequence", alpha=0.7)
    ax.plot(y_pred, "r--", linewidth=1.5, label="RNN Prediction", alpha=0.8)
    ax.fill_between(range(len(y)), y, y_pred, alpha=0.2, color="red")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Value")
    ax.set_title("Sequence Prediction")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    error = y - y_pred
    ax.bar(range(len(y)), error, color="steelblue", alpha=0.7)
    ax.axhline(0, color="k", linestyle="-", linewidth=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Error")
    ax.set_title("Prediction Error")
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, :2])
    im = ax.imshow(h_states.T, aspect="auto", cmap="RdBu_r", interpolation="nearest")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Hidden Unit")
    ax.set_title("Hidden State Evolution (Heatmap)")
    plt.colorbar(im, ax=ax, label="Activation")

    ax = fig.add_subplot(gs[1, 2])
    ax.plot(h_norms, "b-", linewidth=2, marker="o", markersize=4)
    ax.fill_between(range(len(h_norms)), h_norms, alpha=0.3)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Hidden State Norm")
    ax.set_title("Hidden State Magnitude")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 0])
    ax.plot(h_similarities, "g-", linewidth=2, marker="s", markersize=4)
    ax.axhline(0, color="k", linestyle="--", linewidth=0.5)
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Adjacent Hidden State Similarity")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 1])
    scatter = ax.scatter(
        h_proj[:, 0],
        h_proj[:, 1],
        c=range(len(h_proj)),
        cmap="viridis",
        s=100,
        alpha=0.7,
        edgecolors="black",
        linewidth=0.5,
    )
    for t in range(0, len(h_proj) - 1, 5):
        ax.arrow(
            h_proj[t, 0],
            h_proj[t, 1],
            h_proj[t + 1, 0] - h_proj[t, 0],
            h_proj[t + 1, 1] - h_proj[t, 1],
            head_width=0.1,
            head_length=0.1,
            fc="gray",
            ec="gray",
            alpha=0.5,
        )
    ax.set_xlabel("Hidden Unit 1")
    ax.set_ylabel("Hidden Unit 2")
    ax.set_title("Hidden State Trajectory (2D Projection)")
    plt.colorbar(scatter, ax=ax, label="Time Step")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, 2])
    h_flat = h_states.flatten()
    ax.hist(h_flat, bins=30, color="purple", alpha=0.7, edgecolor="black")
    ax.set_xlabel("Activation Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Hidden Unit Activation Distribution")
    ax.grid(True, alpha=0.3, axis="y")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_rnn_structure(result)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
