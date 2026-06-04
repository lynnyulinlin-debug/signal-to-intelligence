"""
实验2.2：LMS vs Adam
对应章节：第2章 - 优化算法与传统机器学习
目标：实现LMS算法求解线性回归，与Adam优化器对比收敛曲线
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

N_SAMPLES = 200
N_FEATURES = 5
LEARNING_RATE_LMS = 0.01
LEARNING_RATE_ADAM = 0.01
EPOCHS = 100
OUTPUT_PATH = Path("assets/ch02_lms_vs_adam.png")


def generate_linear_data(n_samples=N_SAMPLES, n_features=N_FEATURES, noise_std=0.1, seed=42):
    rng = np.random.RandomState(seed)
    w_true = rng.randn(n_features)
    x = rng.randn(n_samples, n_features)
    y = x @ w_true + noise_std * rng.randn(n_samples)
    return x, y, w_true


def train_lms(x, y, learning_rate=LEARNING_RATE_LMS, epochs=EPOCHS):
    w = np.zeros(x.shape[1])
    losses = []

    for _ in range(epochs):
        for i in range(len(x)):
            y_pred = x[i] @ w
            error = y[i] - y_pred
            w += 2 * learning_rate * error * x[i]

        y_pred_all = x @ w
        losses.append(np.mean((y - y_pred_all) ** 2))

    return w, losses


def train_adam(
    x,
    y,
    learning_rate=LEARNING_RATE_ADAM,
    epochs=EPOCHS,
    beta1=0.9,
    beta2=0.999,
    epsilon=1e-8,
):
    w = np.zeros(x.shape[1])
    m = np.zeros(x.shape[1])
    v = np.zeros(x.shape[1])
    losses = []

    for epoch in range(epochs):
        y_pred = x @ w
        error = y_pred - y
        grad = x.T @ error / len(x)

        m = beta1 * m + (1 - beta1) * grad
        v = beta2 * v + (1 - beta2) * (grad**2)

        m_hat = m / (1 - beta1 ** (epoch + 1))
        v_hat = v / (1 - beta2 ** (epoch + 1))

        w -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

        y_pred = x @ w
        losses.append(np.mean((y - y_pred) ** 2))

    return w, losses


def run_experiment(seed=42):
    x, y, w_true = generate_linear_data(seed=seed)
    w_lms, losses_lms = train_lms(x, y)
    w_adam, losses_adam = train_adam(x, y)

    return {
        "X": x,
        "y": y,
        "w_true": w_true,
        "w_lms": w_lms,
        "w_adam": w_adam,
        "losses_lms": losses_lms,
        "losses_adam": losses_adam,
    }


def print_summary(result):
    print("=" * 70)
    print("LMS vs Adam Optimizer Comparison")
    print("=" * 70)
    print(f"样本数: {len(result['X'])}")
    print(f"特征数: {result['X'].shape[1]}")
    print(f"训练轮数: {len(result['losses_lms'])}")
    print()

    print("真实权重:")
    print("-" * 70)
    print(f"w_true: {result['w_true'].round(4)}")
    print()

    print("LMS 算法结果:")
    print("-" * 70)
    print(f"学习到的权重: {result['w_lms'].round(4)}")
    print(f"最终MSE: {result['losses_lms'][-1]:.6f}")
    print(f"权重误差: {np.linalg.norm(result['w_lms'] - result['w_true']):.6f}")
    print()

    print("Adam 优化器结果:")
    print("-" * 70)
    print(f"学习到的权重: {result['w_adam'].round(4)}")
    print(f"最终MSE: {result['losses_adam'][-1]:.6f}")
    print(f"权重误差: {np.linalg.norm(result['w_adam'] - result['w_true']):.6f}")
    print()

    print("收敛对比:")
    print("-" * 70)
    print(
        f"LMS 初始MSE: {result['losses_lms'][0]:.6f} "
        f"-> 最终MSE: {result['losses_lms'][-1]:.6f}"
    )
    print(
        f"Adam 初始MSE: {result['losses_adam'][0]:.6f} "
        f"-> 最终MSE: {result['losses_adam'][-1]:.6f}"
    )
    print()
    print("=" * 70)


def plot_results(result, output_path=OUTPUT_PATH):
    x = result["X"]
    y = result["y"]
    w_true = result["w_true"]
    w_lms = result["w_lms"]
    w_adam = result["w_adam"]
    losses_lms = result["losses_lms"]
    losses_adam = result["losses_adam"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.plot(losses_lms, "b-", linewidth=2, label="LMS", alpha=0.8)
    ax.plot(losses_adam, "r-", linewidth=2, label="Adam", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Convergence Comparison")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(losses_lms, "b-", linewidth=2, label="LMS", alpha=0.8)
    ax.semilogy(losses_adam, "r-", linewidth=2, label="Adam", alpha=0.8)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Convergence Comparison (Log Scale)")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 0]
    weight_error_lms = np.abs(w_lms - w_true)
    weight_error_adam = np.abs(w_adam - w_true)
    x_pos = np.arange(len(w_true))
    width = 0.35
    ax.bar(x_pos - width / 2, weight_error_lms, width, label="LMS", alpha=0.8)
    ax.bar(x_pos + width / 2, weight_error_adam, width, label="Adam", alpha=0.8)
    ax.set_xlabel("Feature Index")
    ax.set_ylabel("Absolute Error")
    ax.set_title("Weight Estimation Error")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    y_pred_lms_final = x @ w_lms
    y_pred_adam_final = x @ w_adam
    ax.scatter(y, y_pred_lms_final, alpha=0.5, s=30, label="LMS Prediction")
    ax.scatter(y, y_pred_adam_final, alpha=0.5, s=30, label="Adam Prediction")
    ax.plot([y.min(), y.max()], [y.min(), y.max()], "k--", linewidth=1, label="Perfect Prediction")
    ax.set_xlabel("True Value")
    ax.set_ylabel("Predicted Value")
    ax.set_title("Prediction Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
