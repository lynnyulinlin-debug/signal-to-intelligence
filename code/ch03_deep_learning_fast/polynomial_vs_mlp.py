"""
实验3.1：多项式拟合 vs MLP
对应章节：第3章 - 深度学习快速通道
目标：对比传统多项式拟合和神经网络的表达能力
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

N_TRAIN = 20
N_TEST = 100
POLY_DEGREES = [1, 3, 5, 9]
LAMBDA_REG = 0.01
OUTPUT_PATH = Path("assets/ch03_polynomial_vs_mlp.png")


def generate_data(n_train=N_TRAIN, n_test=N_TEST, seed=42):
    rng = np.random.RandomState(seed)
    x_train = np.linspace(0, 2 * np.pi, n_train)
    y_train = np.sin(x_train) + 0.3 * rng.randn(n_train)
    x_test = np.linspace(0, 2 * np.pi, n_test)
    y_test = np.sin(x_test)
    return x_train, y_train, x_test, y_test


def fit_polynomials(x_train, y_train, x_test, y_test, degrees=POLY_DEGREES, lambda_reg=LAMBDA_REG):
    poly_results = {}
    for degree in degrees:
        x_train_poly = np.vstack([x_train**i for i in range(degree + 1)]).T
        x_test_poly = np.vstack([x_test**i for i in range(degree + 1)]).T

        xtx = x_train_poly.T @ x_train_poly
        xty = x_train_poly.T @ y_train
        w = np.linalg.solve(xtx + lambda_reg * np.eye(degree + 1), xty)

        y_pred_train = x_train_poly @ w
        y_pred_test = x_test_poly @ w

        poly_results[degree] = {
            "w": w,
            "y_pred_train": y_pred_train,
            "y_pred_test": y_pred_test,
            "mse_train": np.mean((y_pred_train - y_train) ** 2),
            "mse_test": np.mean((y_pred_test - y_test) ** 2),
        }
    return poly_results


class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01, seed=None):
        rng = np.random.RandomState(seed) if seed is not None else np.random
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))
        self.lr = learning_rate

    def relu(self, x):
        return np.maximum(0, x)

    def relu_grad(self, x):
        return (x > 0).astype(float)

    def forward(self, X):
        self.z1 = X @ self.W1 + self.b1
        self.a1 = self.relu(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def backward(self, X, y, y_pred):
        m = X.shape[0]
        dz2 = y_pred - y
        dW2 = self.a1.T @ dz2 / m
        db2 = np.sum(dz2, axis=0, keepdims=True) / m

        da1 = dz2 @ self.W2.T
        dz1 = da1 * self.relu_grad(self.z1)
        dW1 = X.T @ dz1 / m
        db1 = np.sum(dz1, axis=0, keepdims=True) / m

        self.W1 -= self.lr * dW1
        self.b1 -= self.lr * db1
        self.W2 -= self.lr * dW2
        self.b2 -= self.lr * db2

    def train(self, X, y, epochs):
        losses = []
        for _ in range(epochs):
            y_pred = self.forward(X)
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)
            self.backward(X, y, y_pred)
        return losses


def run_experiment(seed=42):
    x_train, y_train, x_test, y_test = generate_data(seed=seed)
    poly_results = fit_polynomials(x_train, y_train, x_test, y_test)

    x_train_nn = x_train.reshape(-1, 1)
    x_test_nn = x_test.reshape(-1, 1)
    y_train_nn = y_train.reshape(-1, 1)
    y_test_nn = y_test.reshape(-1, 1)

    nn = SimpleNN(input_dim=1, hidden_dim=10, output_dim=1, learning_rate=0.1, seed=seed)
    nn_losses = nn.train(x_train_nn, y_train_nn, epochs=200)

    y_pred_nn_train = nn.forward(x_train_nn)
    y_pred_nn_test = nn.forward(x_test_nn)

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "poly_results": poly_results,
        "nn": nn,
        "nn_losses": nn_losses,
        "y_pred_nn_train": y_pred_nn_train,
        "y_pred_nn_test": y_pred_nn_test,
        "mse_nn_train": np.mean((y_pred_nn_train - y_train_nn) ** 2),
        "mse_nn_test": np.mean((y_pred_nn_test - y_test_nn) ** 2),
    }


def print_summary(result):
    print("=" * 70)
    print("Polynomial Fit vs MLP")
    print("=" * 70)
    print(f"训练样本数: {len(result['x_train'])}")
    print(f"测试样本数: {len(result['x_test'])}")
    print()

    print("多项式拟合结果:")
    print("-" * 70)
    for degree in POLY_DEGREES:
        item = result["poly_results"][degree]
        print(
            f"阶数 {degree}: 训练MSE={item['mse_train']:.6f}, "
            f"测试MSE={item['mse_test']:.6f}"
        )
    print()

    print("MLP 结果:")
    print("-" * 70)
    print("隐层维度: 10")
    print(f"训练MSE: {result['mse_nn_train']:.6f}")
    print(f"测试MSE: {result['mse_nn_test']:.6f}")
    print()
    print("=" * 70)


def plot_results(result, output_path=OUTPUT_PATH):
    x_train = result["x_train"]
    y_train = result["y_train"]
    x_test = result["x_test"]
    y_test = result["y_test"]
    poly_results = result["poly_results"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    ax.scatter(x_train, y_train, color="red", s=50, alpha=0.7, label="Training Data")
    ax.plot(x_test, y_test, "k-", linewidth=2, label="True Function")
    for degree in [1, 3, 9]:
        ax.plot(
            x_test,
            poly_results[degree]["y_pred_test"],
            linewidth=1.5,
            label=f"Poly Degree {degree}",
            alpha=0.8,
        )
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("Polynomial Fitting (Different Degrees)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    degrees_list = list(POLY_DEGREES)
    train_errors = [poly_results[d]["mse_train"] for d in degrees_list]
    test_errors = [poly_results[d]["mse_test"] for d in degrees_list]
    x_pos = np.arange(len(degrees_list))
    width = 0.35
    ax.bar(x_pos - width / 2, train_errors, width, label="Train MSE", alpha=0.8)
    ax.bar(x_pos + width / 2, test_errors, width, label="Test MSE", alpha=0.8)
    ax.set_xlabel("Polynomial Degree")
    ax.set_ylabel("MSE")
    ax.set_title("Overfitting in Polynomial Fitting")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(degrees_list)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 0]
    ax.scatter(x_train, y_train, color="red", s=50, alpha=0.7, label="Training Data")
    ax.plot(x_test, y_test, "k-", linewidth=2, label="True Function")
    ax.plot(x_test, result["y_pred_nn_test"], "b-", linewidth=1.5, label="MLP Prediction")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_title("MLP Fitting (10 Hidden Units)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(result["nn_losses"], "b-", linewidth=1.5)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.set_title("MLP Training Loss")
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
