"""
实验2.1：MMSE vs 神经网络
对应章节：第2章 - 优化算法与传统机器学习
目标：对比最小均方误差（MMSE）和神经网络在信号估计中的性能
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

N_SAMPLES = 500
SNR_DB = 10
NOISE_POWER = 10 ** (-SNR_DB / 10)
HIDDEN_SIZE = 16
LEARNING_RATE = 0.01
EPOCHS = 200
OUTPUT_PATH = Path("assets/ch02_mmse_vs_nn.png")


def generate_signal(n_samples=N_SAMPLES, snr_db=SNR_DB, seed=42):
    rng = np.random.RandomState(seed)
    noise_power = 10 ** (-snr_db / 10)
    s_true = rng.randn(n_samples)
    noise = np.sqrt(noise_power) * rng.randn(n_samples)
    y = s_true + noise
    return s_true, noise, y, noise_power


def mmse_estimate(s_true, y):
    sigma_s_sq = np.var(s_true)
    sigma_n_sq = np.var(y - s_true)
    gain = sigma_s_sq / (sigma_s_sq + sigma_n_sq)
    s_mmse = gain * y
    mse = np.mean((s_true - s_mmse) ** 2)
    snr_out = np.var(s_mmse) / np.mean((s_true - s_mmse) ** 2)
    return {
        "sigma_s_sq": sigma_s_sq,
        "sigma_n_sq": sigma_n_sq,
        "gain": gain,
        "estimate": s_mmse,
        "mse": mse,
        "snr_out": snr_out,
    }


class SimpleEstimatorNN:
    def __init__(self, input_dim=1, hidden_dim=HIDDEN_SIZE, output_dim=1, seed=42):
        rng = np.random.RandomState(seed)
        self.W1 = rng.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = rng.randn(hidden_dim, output_dim) * 0.1
        self.b2 = np.zeros((1, output_dim))

    def forward(self, x):
        self.z1 = x @ self.W1 + self.b1
        self.a1 = np.tanh(self.z1)
        self.z2 = self.a1 @ self.W2 + self.b2
        return self.z2

    def train(self, x, y, learning_rate=LEARNING_RATE, epochs=EPOCHS):
        losses = []
        for _ in range(epochs):
            pred = self.forward(x).flatten()
            loss = np.mean((y - pred) ** 2)
            losses.append(loss)

            dz2 = (pred - y).reshape(-1, 1) / len(x)
            dW2 = self.a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ self.W2.T
            dz1 = da1 * (1 - self.a1**2)
            dW1 = x.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            self.W1 -= learning_rate * dW1
            self.b1 -= learning_rate * db1
            self.W2 -= learning_rate * dW2
            self.b2 -= learning_rate * db2

        return losses

    def predict(self, x):
        return self.forward(x).flatten()


def neural_estimate(
    s_true,
    y,
    hidden_size=HIDDEN_SIZE,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    seed=42,
):
    x = y.reshape(-1, 1)
    target = s_true
    model = SimpleEstimatorNN(input_dim=1, hidden_dim=hidden_size, output_dim=1, seed=seed)
    losses = model.train(x, target, learning_rate=learning_rate, epochs=epochs)
    s_nn = model.predict(x)
    mse = np.mean((target - s_nn) ** 2)
    snr_out = np.var(s_nn) / np.mean((target - s_nn) ** 2)
    return {
        "model": model,
        "estimate": s_nn,
        "losses": losses,
        "mse": mse,
        "snr_out": snr_out,
    }


def run_experiment(seed=42):
    s_true, noise, y, noise_power = generate_signal(seed=seed)
    mmse = mmse_estimate(s_true, y)
    nn = neural_estimate(s_true, y, seed=seed)

    return {
        "s_true": s_true,
        "noise": noise,
        "y": y,
        "noise_power": noise_power,
        "sigma_s_sq": mmse["sigma_s_sq"],
        "sigma_n_sq": mmse["sigma_n_sq"],
        "mmse_gain": mmse["gain"],
        "s_mmse": mmse["estimate"],
        "mse_mmse": mmse["mse"],
        "snr_out_mmse": mmse["snr_out"],
        "hidden_size": HIDDEN_SIZE,
        "learning_rate": LEARNING_RATE,
        "epochs": EPOCHS,
        "s_nn": nn["estimate"],
        "losses_nn": nn["losses"],
        "mse_nn": nn["mse"],
        "snr_out_nn": nn["snr_out"],
    }


def print_summary(result):
    print("=" * 70)
    print("MMSE vs Neural Network: Estimation Performance")
    print("=" * 70)
    print(f"样本数: {len(result['s_true'])}")
    print(f"输入信噪比: {SNR_DB} dB")
    print(f"信号方差: {result['sigma_s_sq']:.4f}")
    print(f"噪声方差: {result['sigma_n_sq']:.4f}")
    print()

    print("MMSE 估计器:")
    print("-" * 70)
    print(f"MMSE增益: {result['mmse_gain']:.4f}")
    print(f"输出MSE: {result['mse_mmse']:.6f}")
    print(f"输出信噪比: {10 * np.log10(result['snr_out_mmse']):.2f} dB")
    print()

    print("神经网络估计器:")
    print("-" * 70)
    print(f"隐层大小: {result['hidden_size']}")
    print(f"训练轮数: {result['epochs']}")
    print(f"学习率: {result['learning_rate']}")
    print(f"输出MSE: {result['mse_nn']:.6f}")
    print(f"输出信噪比: {10 * np.log10(result['snr_out_nn']):.2f} dB")
    print()

    print("性能对比:")
    print("-" * 70)
    mse_improvement = (result["mse_mmse"] - result["mse_nn"]) / result["mse_mmse"] * 100
    print(
        f"MSE改进: {mse_improvement:.2f}% "
        f"{'(NN更好)' if mse_improvement > 0 else '(MMSE更好)'}"
    )
    snr_improvement = (result["snr_out_nn"] - result["snr_out_mmse"]) / result["snr_out_mmse"] * 100
    print(
        f"SNR改进: {snr_improvement:.2f}% "
        f"{'(NN更好)' if snr_improvement > 0 else '(MMSE更好)'}"
    )
    print()
    print("=" * 70)


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    idx = slice(0, 100)
    ax.plot(result["s_true"][idx], "k-", linewidth=2, label="True Signal", alpha=0.7)
    ax.plot(result["y"][idx], "gray", linewidth=0.8, label="Noisy Signal", alpha=0.5)
    ax.plot(result["s_mmse"][idx], "b--", linewidth=1.5, label="MMSE Estimate", alpha=0.8)
    ax.plot(result["s_nn"][idx], "r--", linewidth=1.5, label="NN Estimate", alpha=0.8)
    ax.set_xlabel("Sample Index")
    ax.set_ylabel("Amplitude")
    ax.set_title("Signal Estimation (First 100 Samples)")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.semilogy(result["losses_nn"], "r-", linewidth=2, label="NN Training Loss")
    ax.axhline(
        result["mse_mmse"],
        color="b",
        linestyle="--",
        linewidth=2,
        label=f"MMSE MSE ({result['mse_mmse']:.6f})",
    )
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss (log scale)")
    ax.set_title("Training Convergence")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")

    ax = axes[1, 0]
    error_mmse = result["s_true"] - result["s_mmse"]
    error_nn = result["s_true"] - result["s_nn"]
    ax.hist(error_mmse, bins=30, alpha=0.6, label="MMSE Error", color="b", density=True)
    ax.hist(error_nn, bins=30, alpha=0.6, label="NN Error", color="r", density=True)
    ax.set_xlabel("Estimation Error")
    ax.set_ylabel("Probability Density")
    ax.set_title("Error Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    methods = ["MMSE", "Neural Network"]
    mse_values = [result["mse_mmse"], result["mse_nn"]]
    snr_values = [10 * np.log10(result["snr_out_mmse"]), 10 * np.log10(result["snr_out_nn"])]

    x_pos = np.arange(len(methods))
    width = 0.35

    ax2 = ax.twinx()
    ax.bar(x_pos - width / 2, mse_values, width, label="MSE", color="steelblue", alpha=0.8)
    ax2.bar(x_pos + width / 2, snr_values, width, label="Output SNR (dB)", color="coral", alpha=0.8)

    ax.set_ylabel("MSE", color="steelblue")
    ax2.set_ylabel("Output SNR (dB)", color="coral")
    ax.set_title("Performance Comparison")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(methods)
    ax.tick_params(axis="y", labelcolor="steelblue")
    ax2.tick_params(axis="y", labelcolor="coral")
    ax.grid(True, alpha=0.3, axis="y")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, loc="upper left")

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
