"""
实验3.2：CNN结构与特征提取
对应章节：第3章 - 深度学习快速通道
目标：展示CNN如何从图像中逐层提取特征并完成分类
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
IMG_SIZE = 28
NUM_CLASSES = 10
OUTPUT_PATH = Path("assets/ch03_mnist_cnn.png")


class SimpleCNN:
    def __init__(self, num_filters=16, kernel_size=3, seed=42):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        rng = np.random.RandomState(seed)
        self.conv1_filters = rng.randn(num_filters, kernel_size, kernel_size) * 0.01
        self.fc_weights = rng.randn(num_filters * 13 * 13, NUM_CLASSES) * 0.01
        self.fc_bias = np.zeros((1, NUM_CLASSES))

    def relu(self, x):
        return np.maximum(0, x)

    def conv2d(self, x, filters):
        """简化的2D卷积（步长=1，无padding）"""
        batch_size, height, width = x.shape
        num_filters, k_h, k_w = filters.shape
        out_h = height - k_h + 1
        out_w = width - k_w + 1
        output = np.zeros((batch_size, num_filters, out_h, out_w))

        for b in range(batch_size):
            for f in range(num_filters):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = x[b, i : i + k_h, j : j + k_w]
                        output[b, f, i, j] = np.sum(patch * filters[f])

        return output

    def max_pool(self, x, pool_size=2):
        """最大池化"""
        batch_size, channels, height, width = x.shape
        out_h = height // pool_size
        out_w = width // pool_size
        output = np.zeros((batch_size, channels, out_h, out_w))

        for b in range(batch_size):
            for c in range(channels):
                for i in range(out_h):
                    for j in range(out_w):
                        patch = x[
                            b,
                            c,
                            i * pool_size : (i + 1) * pool_size,
                            j * pool_size : (j + 1) * pool_size,
                        ]
                        output[b, c, i, j] = np.max(patch)

        return output

    def forward(self, x):
        self.conv1_out = self.conv2d(x, self.conv1_filters)
        self.conv1_out = self.relu(self.conv1_out)
        self.pool_out = self.max_pool(self.conv1_out, pool_size=2)
        batch_size = x.shape[0]
        self.flat = self.pool_out.reshape(batch_size, -1)
        self.logits = self.flat @ self.fc_weights + self.fc_bias
        return self.logits

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)


def generate_mnist_like_data(num_samples=1000, seed=42):
    """生成类似MNIST的合成数据"""
    rng = np.random.RandomState(seed)
    x = rng.randn(num_samples, IMG_SIZE, IMG_SIZE) * 0.5 + 0.5
    x = np.clip(x, 0, 1)
    y = rng.randint(0, NUM_CLASSES, num_samples)
    return x, y


def train_cnn(
    x_train,
    y_train,
    num_filters=16,
    kernel_size=3,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    seed=42,
):
    model = SimpleCNN(num_filters=num_filters, kernel_size=kernel_size, seed=seed)
    losses = []

    print("训练CNN...")
    for epoch in range(epochs):
        epoch_loss = 0
        num_batches = len(x_train) // batch_size

        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size
            end_idx = start_idx + batch_size

            x_batch = x_train[start_idx:end_idx]
            y_batch = y_train[start_idx:end_idx]

            logits = model.forward(x_batch)
            probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
            batch_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))
            epoch_loss += batch_loss

            # 这里只做演示，不实现完整反向传播。
            _ = learning_rate

        epoch_loss /= num_batches
        losses.append(epoch_loss)
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {epoch_loss:.4f}")

    return model, losses


def evaluate(model, x_train, y_train, x_test, y_test):
    y_pred_train = model.predict(x_train)
    y_pred_test = model.predict(x_test)
    train_acc = np.mean(y_pred_train == y_train)
    test_acc = np.mean(y_pred_test == y_test)
    return train_acc, test_acc


def run_experiment(
    seed=42,
    train_samples=1000,
    test_samples=200,
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    num_filters=16,
    kernel_size=3,
):
    x_train, y_train = generate_mnist_like_data(num_samples=train_samples, seed=seed)
    x_test, y_test = generate_mnist_like_data(num_samples=test_samples, seed=seed + 1)
    model, losses = train_cnn(
        x_train,
        y_train,
        num_filters=num_filters,
        kernel_size=kernel_size,
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
    )
    train_acc, test_acc = evaluate(model, x_train, y_train, x_test, y_test)

    return {
        "X_train": x_train,
        "y_train": y_train,
        "X_test": x_test,
        "y_test": y_test,
        "model": model,
        "losses": losses,
        "train_acc": train_acc,
        "test_acc": test_acc,
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "train_samples": train_samples,
        "test_samples": test_samples,
    }


def print_summary(result):
    model = result["model"]
    print()
    print("=" * 70)
    print("MNIST CNN Training Results")
    print("=" * 70)
    print("模型配置:")
    print(f"  卷积核数: {model.num_filters}")
    print(f"  卷积核大小: {model.kernel_size}x{model.kernel_size}")
    print(f"  训练轮数: {result['epochs']}")
    print(f"  批大小: {result['batch_size']}")
    print()
    print("性能指标:")
    print(f"  训练准确率: {result['train_acc']:.4f}")
    print(f"  测试准确率: {result['test_acc']:.4f}")
    print(f"  最终损失: {result['losses'][-1]:.4f}")
    print()
    print("卷积核统计:")
    print(f"  卷积核形状: {model.conv1_filters.shape}")
    print(
        f"  卷积核范围: "
        f"[{model.conv1_filters.min():.4f}, {model.conv1_filters.max():.4f}]"
    )
    print(f"  卷积核均值: {model.conv1_filters.mean():.4f}")
    print()
    print("=" * 70)


def plot_results(result, output_path=OUTPUT_PATH):
    model = result["model"]
    x_train = result["X_train"]
    losses = result["losses"]

    fig, axes = plt.subplots(3, 2, figsize=(12, 12))

    ax = axes[0, 0]
    ax.plot(losses, "b-", linewidth=2)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.set_title("Training Loss")
    ax.grid(True, alpha=0.3)

    ax = axes[0, 1]
    ax.imshow(x_train[0], cmap="gray")
    ax.set_title("Sample Training Image")
    ax.axis("off")

    for idx in range(4):
        row = 1 + idx // 2
        col = idx % 2
        ax = axes[row, col]
        kernel = model.conv1_filters[idx]
        ax.imshow(kernel, cmap="RdBu", vmin=-0.1, vmax=0.1)
        ax.set_title(f"Filter {idx}")
        ax.axis("off")

    ax = axes[2, 1]
    ax.hist(
        model.conv1_filters.flatten(),
        bins=50,
        alpha=0.7,
        color="blue",
        edgecolor="black",
    )
    ax.set_xlabel("Weight Value")
    ax.set_ylabel("Frequency")
    ax.set_title("Conv Filter Weight Distribution")
    ax.grid(True, alpha=0.3, axis="y")

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
