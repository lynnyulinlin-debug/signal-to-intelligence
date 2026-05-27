"""
实验2.3：线性回归与逻辑回归
对应章节：第2章 - 优化算法与传统机器学习
目标：用梯度下降实现线性回归和逻辑回归，理解ML本质是优化问题
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
N_SAMPLES = 200
LEARNING_RATE = 0.01
EPOCHS = 100

# ============ 线性回归 ============
print("=" * 70)
print("线性回归：用梯度下降最小化均方误差")
print("=" * 70)

# 生成数据：y = 2x + 1 + noise
X_lr = np.random.randn(N_SAMPLES, 1)
y_lr = 2 * X_lr + 1 + 0.5 * np.random.randn(N_SAMPLES, 1)

# 初始化参数
w_lr = np.random.randn(1, 1) * 0.1
b_lr = 0.0
losses_lr = []

# 梯度下降
for epoch in range(EPOCHS):
    # 前向传播
    y_pred = X_lr @ w_lr + b_lr

    # 计算损失（MSE）
    loss = np.mean((y_pred - y_lr) ** 2)
    losses_lr.append(loss)

    # 反向传播
    dw = 2 * X_lr.T @ (y_pred - y_lr) / N_SAMPLES
    db = 2 * np.mean(y_pred - y_lr)

    # 更新参数
    w_lr -= LEARNING_RATE * dw
    b_lr -= LEARNING_RATE * db

print(f"学习到的参数: w={w_lr[0,0]:.4f}, b={b_lr:.4f}")
print(f"真实参数: w=2.0000, b=1.0000")
print(f"初始损失: {losses_lr[0]:.6f}")
print(f"最终损失: {losses_lr[-1]:.6f}")
print()

# ============ 逻辑回归 ============
print("=" * 70)
print("逻辑回归：用梯度下降最小化交叉熵")
print("=" * 70)

# 生成二分类数据
X_logistic = np.random.randn(N_SAMPLES, 2)
y_logistic = (X_logistic[:, 0] + X_logistic[:, 1] > 0).astype(int).reshape(-1, 1)

# 初始化参数
w_logistic = np.random.randn(2, 1) * 0.1
b_logistic = 0.0
losses_logistic = []

# Sigmoid函数
def sigmoid(z):
    return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

# 梯度下降
for epoch in range(EPOCHS):
    # 前向传播
    z = X_logistic @ w_logistic + b_logistic
    y_pred = sigmoid(z)

    # 计算损失（交叉熵）
    loss = -np.mean(y_logistic * np.log(y_pred + 1e-8) +
                    (1 - y_logistic) * np.log(1 - y_pred + 1e-8))
    losses_logistic.append(loss)

    # 反向传播
    dw = X_logistic.T @ (y_pred - y_logistic) / N_SAMPLES
    db = np.mean(y_pred - y_logistic)

    # 更新参数
    w_logistic -= LEARNING_RATE * dw
    b_logistic -= LEARNING_RATE * db

# 计算准确率
y_pred_final = sigmoid(X_logistic @ w_logistic + b_logistic)
y_pred_class = (y_pred_final > 0.5).astype(int)
accuracy = np.mean(y_pred_class == y_logistic)

print(f"学习到的参数: w={w_logistic.flatten()}")
print(f"初始损失: {losses_logistic[0]:.6f}")
print(f"最终损失: {losses_logistic[-1]:.6f}")
print(f"分类准确率: {accuracy:.4f}")
print()

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 线性回归：数据和拟合线
ax = axes[0, 0]
ax.scatter(X_lr, y_lr, alpha=0.5, s=30, label='Data')
X_line = np.linspace(X_lr.min(), X_lr.max(), 100).reshape(-1, 1)
y_line = X_line @ w_lr + b_lr
ax.plot(X_line, y_line, 'r-', linewidth=2, label='Fitted Line')
ax.set_xlabel('X')
ax.set_ylabel('y')
ax.set_title('Linear Regression')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 线性回归：损失曲线
ax = axes[0, 1]
ax.plot(losses_lr, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Linear Regression: Training Loss')
ax.grid(True, alpha=0.3)

# 3. 逻辑回归：决策边界
ax = axes[1, 0]
h = 0.02
x_min, x_max = X_logistic[:, 0].min() - 1, X_logistic[:, 0].max() + 1
y_min, y_max = X_logistic[:, 1].min() - 1, X_logistic[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = sigmoid(np.c_[xx.ravel(), yy.ravel()] @ w_logistic + b_logistic)
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=20, cmap='RdBu', alpha=0.6)
ax.scatter(X_logistic[y_logistic.flatten() == 0, 0],
          X_logistic[y_logistic.flatten() == 0, 1],
          c='blue', marker='o', s=30, label='Class 0')
ax.scatter(X_logistic[y_logistic.flatten() == 1, 0],
          X_logistic[y_logistic.flatten() == 1, 1],
          c='red', marker='x', s=30, label='Class 1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Logistic Regression: Decision Boundary')
ax.legend()

# 4. 逻辑回归：损失曲线
ax = axes[1, 1]
ax.plot(losses_logistic, 'g-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Cross-Entropy Loss')
ax.set_title('Logistic Regression: Training Loss')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch02_linear_logistic_regression.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch02_linear_logistic_regression.png")
