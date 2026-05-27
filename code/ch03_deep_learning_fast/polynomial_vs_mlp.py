"""
实验3.1：多项式拟合 vs MLP
对应章节：第3章 - 深度学习快速通道
目标：对比传统多项式拟合和神经网络的表达能力
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
N_TRAIN = 20
N_TEST = 100
POLY_DEGREES = [1, 3, 5, 9]
LAMBDA_REG = 0.01  # 正则化系数

# ============ 核心逻辑 ============
# 生成数据：y = sin(x) + noise
x_train = np.linspace(0, 2*np.pi, N_TRAIN)
y_train = np.sin(x_train) + 0.3 * np.random.randn(N_TRAIN)

x_test = np.linspace(0, 2*np.pi, N_TEST)
y_test = np.sin(x_test)

# 方法1：多项式拟合（不同阶数）
poly_results = {}
for degree in POLY_DEGREES:
    # 构造特征矩阵
    X_train_poly = np.vstack([x_train**i for i in range(degree+1)]).T
    X_test_poly = np.vstack([x_test**i for i in range(degree+1)]).T

    # 岭回归（带L2正则化）
    # w = (X^T X + λI)^{-1} X^T y
    XtX = X_train_poly.T @ X_train_poly
    Xty = X_train_poly.T @ y_train
    w = np.linalg.solve(XtX + LAMBDA_REG * np.eye(degree+1), Xty)

    # 预测
    y_pred_train = X_train_poly @ w
    y_pred_test = X_test_poly @ w

    # 计算MSE
    mse_train = np.mean((y_pred_train - y_train) ** 2)
    mse_test = np.mean((y_pred_test - y_test) ** 2)

    poly_results[degree] = {
        'w': w,
        'y_pred_train': y_pred_train,
        'y_pred_test': y_pred_test,
        'mse_train': mse_train,
        'mse_test': mse_test
    }

# 方法2：小型MLP（2层隐层）
class SimpleNN:
    def __init__(self, input_dim, hidden_dim, output_dim, learning_rate=0.01):
        self.W1 = np.random.randn(input_dim, hidden_dim) * 0.1
        self.b1 = np.zeros((1, hidden_dim))
        self.W2 = np.random.randn(hidden_dim, output_dim) * 0.1
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

# 训练MLP
X_train_nn = x_train.reshape(-1, 1)
X_test_nn = x_test.reshape(-1, 1)
y_train_nn = y_train.reshape(-1, 1)
y_test_nn = y_test.reshape(-1, 1)

nn = SimpleNN(input_dim=1, hidden_dim=10, output_dim=1, learning_rate=0.1)
nn_losses = nn.train(X_train_nn, y_train_nn, epochs=200)

y_pred_nn_train = nn.forward(X_train_nn)
y_pred_nn_test = nn.forward(X_test_nn)

mse_nn_train = np.mean((y_pred_nn_train - y_train_nn) ** 2)
mse_nn_test = np.mean((y_pred_nn_test - y_test_nn) ** 2)

# ============ 结果输出 ============
print("=" * 70)
print("多项式拟合 vs MLP")
print("=" * 70)
print(f"训练样本数: {N_TRAIN}")
print(f"测试样本数: {N_TEST}")
print()

print("多项式拟合结果:")
print("-" * 70)
for degree in POLY_DEGREES:
    result = poly_results[degree]
    print(f"阶数 {degree}: 训练MSE={result['mse_train']:.6f}, 测试MSE={result['mse_test']:.6f}")
print()

print("MLP 结果:")
print("-" * 70)
print(f"隐层维度: 10")
print(f"训练MSE: {mse_nn_train:.6f}")
print(f"测试MSE: {mse_nn_test:.6f}")
print()

print("过拟合分析:")
print("-" * 70)
for degree in POLY_DEGREES:
    result = poly_results[degree]
    overfitting = result['mse_test'] - result['mse_train']
    print(f"多项式阶数{degree}: 过拟合程度={overfitting:.6f}")
overfitting_nn = mse_nn_test - mse_nn_train
print(f"MLP: 过拟合程度={overfitting_nn:.6f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 不同阶数的多项式拟合
ax = axes[0, 0]
ax.scatter(x_train, y_train, color='red', s=50, alpha=0.7, label='Training Data')
ax.plot(x_test, y_test, 'k-', linewidth=2, label='True Function')
for degree in [1, 3, 9]:
    ax.plot(x_test, poly_results[degree]['y_pred_test'], linewidth=1.5,
            label=f'Poly Degree {degree}', alpha=0.8)
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('Polynomial Fitting (Different Degrees)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 训练 vs 测试误差
ax = axes[0, 1]
degrees_list = list(POLY_DEGREES)
train_errors = [poly_results[d]['mse_train'] for d in degrees_list]
test_errors = [poly_results[d]['mse_test'] for d in degrees_list]
x_pos = np.arange(len(degrees_list))
width = 0.35
ax.bar(x_pos - width/2, train_errors, width, label='Train MSE', alpha=0.8)
ax.bar(x_pos + width/2, test_errors, width, label='Test MSE', alpha=0.8)
ax.set_xlabel('Polynomial Degree')
ax.set_ylabel('MSE')
ax.set_title('Overfitting in Polynomial Fitting')
ax.set_xticks(x_pos)
ax.set_xticklabels(degrees_list)
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 3. MLP 拟合
ax = axes[1, 0]
ax.scatter(x_train, y_train, color='red', s=50, alpha=0.7, label='Training Data')
ax.plot(x_test, y_test, 'k-', linewidth=2, label='True Function')
ax.plot(x_test, y_pred_nn_test, 'b-', linewidth=1.5, label='MLP Prediction')
ax.set_xlabel('x')
ax.set_ylabel('y')
ax.set_title('MLP Fitting (10 Hidden Units)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. MLP 训练损失
ax = axes[1, 1]
ax.plot(nn_losses, 'b-', linewidth=1.5)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('MLP Training Loss')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch03_polynomial_vs_mlp.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch03_polynomial_vs_mlp.png")
