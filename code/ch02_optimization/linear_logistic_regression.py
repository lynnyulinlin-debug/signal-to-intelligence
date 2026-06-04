"""
实验2.3：线性回归与逻辑回归
对应章节：第2章 - 优化算法与传统机器学习
目标：用梯度下降实现线性回归和逻辑回归，理解ML本质是优化问题
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

N_SAMPLES = 200
LEARNING_RATE = 0.01
EPOCHS = 100
OUTPUT_PATH = Path("assets/ch02_linear_logistic_regression.png")


def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))


def generate_linear_regression_data(n_samples=N_SAMPLES, noise_std=0.5, seed=42):
    rng = np.random.RandomState(seed)
    x = rng.randn(n_samples, 1)
    y = 2 * x + 1 + noise_std * rng.randn(n_samples, 1)
    return x, y


def train_linear_regression(x, y, learning_rate=LEARNING_RATE, epochs=EPOCHS, seed=42):
    rng = np.random.RandomState(seed + 1)
    w = rng.randn(1, 1) * 0.1
    b = 0.0
    losses = []

    for _ in range(epochs):
        y_pred = x @ w + b
        losses.append(np.mean((y_pred - y) ** 2))

        dw = 2 * x.T @ (y_pred - y) / len(x)
        db = 2 * np.mean(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, losses


def generate_logistic_regression_data(n_samples=N_SAMPLES, seed=42):
    rng = np.random.RandomState(seed + 2)
    x = rng.randn(n_samples, 2)
    y = (x[:, 0] + x[:, 1] > 0).astype(int).reshape(-1, 1)
    return x, y


def train_logistic_regression(x, y, learning_rate=LEARNING_RATE, epochs=EPOCHS, seed=42):
    rng = np.random.RandomState(seed + 3)
    w = rng.randn(x.shape[1], 1) * 0.1
    b = 0.0
    losses = []

    for _ in range(epochs):
        z = x @ w + b
        y_pred = sigmoid(z)
        loss = -np.mean(y * np.log(y_pred + 1e-8) + (1 - y) * np.log(1 - y_pred + 1e-8))
        losses.append(loss)

        dw = x.T @ (y_pred - y) / len(x)
        db = np.mean(y_pred - y)

        w -= learning_rate * dw
        b -= learning_rate * db

    return w, b, losses


def classification_accuracy(x, y, w, b):
    y_pred = sigmoid(x @ w + b)
    y_pred_class = (y_pred > 0.5).astype(int)
    return np.mean(y_pred_class == y)


def run_experiment(seed=42):
    x_lr, y_lr = generate_linear_regression_data(seed=seed)
    w_lr, b_lr, losses_lr = train_linear_regression(x_lr, y_lr, seed=seed)

    x_logistic, y_logistic = generate_logistic_regression_data(seed=seed)
    w_logistic, b_logistic, losses_logistic = train_logistic_regression(
        x_logistic,
        y_logistic,
        seed=seed,
    )
    accuracy = classification_accuracy(x_logistic, y_logistic, w_logistic, b_logistic)

    return {
        "X_lr": x_lr,
        "y_lr": y_lr,
        "w_lr": w_lr,
        "b_lr": b_lr,
        "losses_lr": losses_lr,
        "X_logistic": x_logistic,
        "y_logistic": y_logistic,
        "w_logistic": w_logistic,
        "b_logistic": b_logistic,
        "losses_logistic": losses_logistic,
        "accuracy": accuracy,
    }


def print_summary(result):
    print("=" * 70)
    print("Linear regression: minimize MSE with gradient descent")
    print("=" * 70)
    print(f"学习到的参数: w={result['w_lr'][0, 0]:.4f}, b={result['b_lr']:.4f}")
    print("真实参数: w=2.0000, b=1.0000")
    print(f"初始损失: {result['losses_lr'][0]:.6f}")
    print(f"最终损失: {result['losses_lr'][-1]:.6f}")
    print()

    print("=" * 70)
    print("Logistic regression: minimize cross-entropy with gradient descent")
    print("=" * 70)
    print(f"学习到的参数: w={result['w_logistic'].flatten()}")
    print(f"初始损失: {result['losses_logistic'][0]:.6f}")
    print(f"最终损失: {result['losses_logistic'][-1]:.6f}")
    print(f"分类准确率: {result['accuracy']:.4f}")
    print()


def plot_results(result, output_path=OUTPUT_PATH):
    x_lr = result["X_lr"]
    y_lr = result["y_lr"]
    w_lr = result["w_lr"]
    b_lr = result["b_lr"]
    x_logistic = result["X_logistic"]
    y_logistic = result["y_logistic"]
    w_logistic = result["w_logistic"]
    b_logistic = result["b_logistic"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(x_lr, y_lr, alpha=0.5, s=30, label="Data")
    x_line = np.linspace(x_lr.min(), x_lr.max(), 100).reshape(-1, 1)
    y_line = x_line @ w_lr + b_lr
    ax.plot(x_line, y_line, "r-", linewidth=2, label="Fitted Line")
    ax.set_xlabel("X")
    ax.set_ylabel("y")
    ax.set_title("Linear Regression")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(result["losses_lr"], "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("Linear Regression: Training Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    h = 0.02
    x_min, x_max = x_logistic[:, 0].min() - 1, x_logistic[:, 0].max() + 1
    y_min, y_max = x_logistic[:, 1].min() - 1, x_logistic[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w_logistic + b_logistic)
    z = z.reshape(xx.shape)

    ax.contourf(xx, yy, z, levels=20, cmap="RdBu", alpha=0.6)
    ax.scatter(
        x_logistic[y_logistic.flatten() == 0, 0],
        x_logistic[y_logistic.flatten() == 0, 1],
        c="blue",
        marker="o",
        s=30,
        label="Class 0",
    )
    ax.scatter(
        x_logistic[y_logistic.flatten() == 1, 0],
        x_logistic[y_logistic.flatten() == 1, 1],
        c="red",
        marker="x",
        s=30,
        label="Class 1",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Logistic Regression: Decision Boundary")
    ax.legend()

    ax = axes[1, 1]
    ax.plot(result["losses_logistic"], "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Cross-Entropy Loss")
    ax.set_title("Logistic Regression: Training Loss")
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
