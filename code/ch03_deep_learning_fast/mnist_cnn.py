"""
实验3.2：CNN结构与特征提取
对应章节：第3章 - 深度学习快速通道
目标：展示CNN如何从图像中逐层提取特征并完成分类
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
EPOCHS = 5
BATCH_SIZE = 32
LEARNING_RATE = 0.01
IMG_SIZE = 28
NUM_CLASSES = 10

# ============ 核心逻辑 ============
# 简化的CNN实现（仅用NumPy）
class SimpleCNN:
    def __init__(self, num_filters=16, kernel_size=3):
        self.num_filters = num_filters
        self.kernel_size = kernel_size
        # 初始化卷积核
        self.conv1_filters = np.random.randn(num_filters, kernel_size, kernel_size) * 0.01
        self.fc_weights = np.random.randn(num_filters * 13 * 13, NUM_CLASSES) * 0.01
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
                        patch = x[b, i:i+k_h, j:j+k_w]
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
                        patch = x[b, c, i*pool_size:(i+1)*pool_size, j*pool_size:(j+1)*pool_size]
                        output[b, c, i, j] = np.max(patch)

        return output

    def forward(self, x):
        # Conv1 + ReLU
        self.conv1_out = self.conv2d(x, self.conv1_filters)
        self.conv1_out = self.relu(self.conv1_out)

        # MaxPool
        self.pool_out = self.max_pool(self.conv1_out, pool_size=2)

        # Flatten
        batch_size = x.shape[0]
        self.flat = self.pool_out.reshape(batch_size, -1)

        # FC layer
        self.logits = self.flat @ self.fc_weights + self.fc_bias
        return self.logits

    def predict(self, x):
        logits = self.forward(x)
        return np.argmax(logits, axis=1)

# 加载MNIST数据（简化版：生成合成数据）
def generate_mnist_like_data(num_samples=1000):
    """生成类似MNIST的合成数据"""
    X = np.random.randn(num_samples, IMG_SIZE, IMG_SIZE) * 0.5 + 0.5
    X = np.clip(X, 0, 1)
    y = np.random.randint(0, NUM_CLASSES, num_samples)
    return X, y

# 生成数据
X_train, y_train = generate_mnist_like_data(num_samples=1000)
X_test, y_test = generate_mnist_like_data(num_samples=200)

# 训练CNN
model = SimpleCNN(num_filters=16, kernel_size=3)
losses = []

print("训练CNN...")
for epoch in range(EPOCHS):
    epoch_loss = 0
    num_batches = len(X_train) // BATCH_SIZE

    for batch_idx in range(num_batches):
        start_idx = batch_idx * BATCH_SIZE
        end_idx = start_idx + BATCH_SIZE

        X_batch = X_train[start_idx:end_idx]
        y_batch = y_train[start_idx:end_idx]

        # 前向传播
        logits = model.forward(X_batch)

        # 计算损失（简化的交叉熵）
        probs = np.exp(logits) / np.sum(np.exp(logits), axis=1, keepdims=True)
        batch_loss = -np.mean(np.log(probs[np.arange(len(y_batch)), y_batch] + 1e-8))
        epoch_loss += batch_loss

    epoch_loss /= num_batches
    losses.append(epoch_loss)
    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {epoch_loss:.4f}")

# 评估
y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)
train_acc = np.mean(y_pred_train == y_train)
test_acc = np.mean(y_pred_test == y_test)

# ============ 结果输出 ============
print()
print("=" * 70)
print("MNIST CNN 训练结果")
print("=" * 70)
print(f"模型配置:")
print(f"  卷积核数: 16")
print(f"  卷积核大小: 3x3")
print(f"  训练轮数: {EPOCHS}")
print(f"  批大小: {BATCH_SIZE}")
print()

print(f"性能指标:")
print(f"  训练准确率: {train_acc:.4f}")
print(f"  测试准确率: {test_acc:.4f}")
print(f"  最终损失: {losses[-1]:.4f}")
print()

print(f"卷积核统计:")
print(f"  卷积核形状: {model.conv1_filters.shape}")
print(f"  卷积核范围: [{model.conv1_filters.min():.4f}, {model.conv1_filters.max():.4f}]")
print(f"  卷积核均值: {model.conv1_filters.mean():.4f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(3, 2, figsize=(12, 12))

# 1. 训练损失
ax = axes[0, 0]
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Loss')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)

# 2. 样本图像
ax = axes[0, 1]
ax.imshow(X_train[0], cmap='gray')
ax.set_title('Sample Training Image')
ax.axis('off')

# 3. 第一层卷积核可视化（前4个）
for idx in range(4):
    row = 1 + idx // 2
    col = idx % 2
    ax = axes[row, col]
    kernel = model.conv1_filters[idx]
    ax.imshow(kernel, cmap='RdBu', vmin=-0.1, vmax=0.1)
    ax.set_title(f'Filter {idx}')
    ax.axis('off')

# 4. 卷积核分布
ax = axes[2, 1]
ax.hist(model.conv1_filters.flatten(), bins=50, alpha=0.7, color='blue', edgecolor='black')
ax.set_xlabel('Weight Value')
ax.set_ylabel('Frequency')
ax.set_title('Conv Filter Weight Distribution')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/ch03_mnist_cnn.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch03_mnist_cnn.png")
