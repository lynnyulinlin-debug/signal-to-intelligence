"""
实验2.4：SVM与核方法
对应章节：第2章 - 优化算法与传统机器学习
目标：理解SVM的优化问题和核方法
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

N_SAMPLES = 100
OUTPUT_PATH = Path("assets/ch02_svm_kernel.png")


def generate_linear_svm_data(n_samples=N_SAMPLES, seed=42):
    rng = np.random.RandomState(seed)
    x_class0 = rng.randn(n_samples // 2, 2) + np.array([2, 2])
    x_class1 = rng.randn(n_samples // 2, 2) + np.array([-2, -2])
    x = np.vstack([x_class0, x_class1])
    y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])
    return x, y


def train_linear_svm(x, y, learning_rate=0.01, c=1.0, epochs=100, seed=42):
    rng = np.random.RandomState(seed + 1)
    w = rng.randn(x.shape[1]) * 0.1
    b = 0.0
    losses = []

    for _ in range(epochs):
        z = x @ w + b
        margins = 1 - y * z
        hinge_loss = np.maximum(0, margins)
        losses.append(np.mean(hinge_loss) + c * np.sum(w**2) / 2)

        mask = margins > 0
        grad_w = -x[mask].T @ y[mask] / len(x) + c * w
        grad_b = -np.sum(y[mask]) / len(x)

        w -= learning_rate * grad_w
        b -= learning_rate * grad_b

    return w, b, losses


def rbf_kernel(x1, x2, gamma=1.0):
    """计算RBF核矩阵"""
    sq_distances = np.sum((x1[:, np.newaxis, :] - x2[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sq_distances)


def generate_ring_data(n_per_class=50):
    theta = np.linspace(0, 2 * np.pi, n_per_class)
    r_inner = 1
    r_outer = 3
    x_inner = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
    x_outer = np.column_stack([r_outer * np.cos(theta), r_outer * np.sin(theta)])
    x = np.vstack([x_inner, x_outer])
    y = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])
    return x, y


def train_kernel_svm(
    x,
    y,
    gamma=0.5,
    learning_rate=0.001,
    epochs=100,
    regularization=0.01,
    seed=42,
):
    rng = np.random.RandomState(seed + 2)
    kernel = rbf_kernel(x, x, gamma=gamma)
    alpha = rng.randn(len(x)) * 0.01
    b = 0.0
    losses = []

    for _ in range(epochs):
        f = kernel @ alpha + b
        margins = 1 - y * f
        hinge_loss = np.maximum(0, margins)
        losses.append(np.mean(hinge_loss) + regularization * np.sum(alpha**2))

        mask = margins > 0
        dalpha = -kernel.T @ (y * mask) / len(x) + regularization * alpha
        db = -np.sum(y * mask) / len(x)

        alpha -= learning_rate * dalpha
        b -= learning_rate * db

    return alpha, b, losses, kernel


def classification_accuracy(x, y, w, b):
    y_pred = np.sign(x @ w + b)
    return np.mean(y_pred == y)


def kernel_accuracy(kernel, y, alpha, b):
    y_pred = np.sign(kernel @ alpha + b)
    return np.mean(y_pred == y)


def run_experiment(seed=42):
    x_svm, y_svm = generate_linear_svm_data(seed=seed)
    w, b, losses_svm = train_linear_svm(x_svm, y_svm, seed=seed)
    accuracy_svm = classification_accuracy(x_svm, y_svm, w, b)

    x_kernel, y_kernel = generate_ring_data()
    alpha, b_kernel, losses_kernel, kernel = train_kernel_svm(x_kernel, y_kernel, seed=seed)
    accuracy_kernel = kernel_accuracy(kernel, y_kernel, alpha, b_kernel)

    return {
        "X_svm": x_svm,
        "y_svm": y_svm,
        "w": w,
        "b": b,
        "losses_svm": losses_svm,
        "accuracy_svm": accuracy_svm,
        "X_kernel": x_kernel,
        "y_kernel": y_kernel,
        "alpha": alpha,
        "b_kernel": b_kernel,
        "losses_kernel": losses_kernel,
        "accuracy_kernel": accuracy_kernel,
    }


def print_summary(result):
    print("=" * 70)
    print("Support Vector Machine: Maximum-Margin Classification")
    print("=" * 70)
    print(f"学习到的权重: w={result['w']}")
    print(f"偏置: b={result['b']:.4f}")
    print(f"分类准确率: {result['accuracy_svm']:.4f}")
    print()

    print("=" * 70)
    print("Kernel Method: Nonlinear SVM")
    print("=" * 70)
    print(f"分类准确率: {result['accuracy_kernel']:.4f}")
    print()


def plot_results(result, output_path=OUTPUT_PATH):
    x_svm = result["X_svm"]
    y_svm = result["y_svm"]
    w = result["w"]
    b = result["b"]
    x_kernel = result["X_kernel"]
    y_kernel = result["y_kernel"]
    alpha = result["alpha"]
    b_kernel = result["b_kernel"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    h = 0.02
    x_min, x_max = x_svm[:, 0].min() - 1, x_svm[:, 0].max() + 1
    y_min, y_max = x_svm[:, 1].min() - 1, x_svm[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z = np.sign(np.c_[xx.ravel(), yy.ravel()] @ w + b).reshape(xx.shape)

    ax.contourf(xx, yy, z, levels=1, colors=["lightblue", "lightcoral"], alpha=0.6)
    ax.scatter(
        x_svm[y_svm == 1, 0],
        x_svm[y_svm == 1, 1],
        c="blue",
        marker="o",
        s=30,
        label="Class +1",
    )
    ax.scatter(
        x_svm[y_svm == -1, 0],
        x_svm[y_svm == -1, 1],
        c="red",
        marker="x",
        s=30,
        label="Class -1",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Linear SVM")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.plot(result["losses_svm"], "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hinge Loss")
    ax.set_title("Linear SVM: Training Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    h = 0.05
    x_min, x_max = x_kernel[:, 0].min() - 1, x_kernel[:, 0].max() + 1
    y_min, y_max = x_kernel[:, 1].min() - 1, x_kernel[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    z_points = np.c_[xx.ravel(), yy.ravel()]
    k_test = rbf_kernel(z_points, x_kernel, gamma=0.5)
    z = np.sign(k_test @ alpha + b_kernel).reshape(xx.shape)

    ax.contourf(xx, yy, z, levels=1, colors=["lightblue", "lightcoral"], alpha=0.6)
    ax.scatter(
        x_kernel[y_kernel == 1, 0],
        x_kernel[y_kernel == 1, 1],
        c="blue",
        marker="o",
        s=30,
        label="Class +1",
    )
    ax.scatter(
        x_kernel[y_kernel == -1, 0],
        x_kernel[y_kernel == -1, 1],
        c="red",
        marker="x",
        s=30,
        label="Class -1",
    )
    ax.set_xlabel("Feature 1")
    ax.set_ylabel("Feature 2")
    ax.set_title("Kernel SVM (RBF)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(result["losses_kernel"], "g-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Hinge Loss")
    ax.set_title("Kernel SVM: Training Loss")
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
