"""
实验2.4：SVM与核方法
对应章节：第2章 - 优化算法与传统机器学习
目标：理解SVM的优化问题和核方法
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
N_SAMPLES = 100

# ============ 线性SVM ============
print("=" * 70)
print("支持向量机（SVM）：最大间隔分类")
print("=" * 70)

# 生成线性可分数据
X_class0 = np.random.randn(N_SAMPLES // 2, 2) + np.array([2, 2])
X_class1 = np.random.randn(N_SAMPLES // 2, 2) + np.array([-2, -2])
X_svm = np.vstack([X_class0, X_class1])
y_svm = np.hstack([np.ones(N_SAMPLES // 2), -np.ones(N_SAMPLES // 2)])

# 简化的SVM实现（使用梯度下降）
w = np.random.randn(2) * 0.1
b = 0.0
learning_rate = 0.01
C = 1.0  # 正则化参数
epochs = 100
losses_svm = []

for epoch in range(epochs):
    # 计算预测
    z = X_svm @ w + b

    # Hinge损失
    margins = 1 - y_svm * z
    hinge_loss = np.maximum(0, margins)
    loss = np.mean(hinge_loss) + C * np.sum(w ** 2) / 2
    losses_svm.append(loss)

    # 梯度
    mask = margins > 0
    grad_w = -X_svm[mask].T @ y_svm[mask] / N_SAMPLES + C * w
    grad_b = -np.sum(y_svm[mask]) / N_SAMPLES

    # 更新
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

# 计算准确率
y_pred_svm = np.sign(X_svm @ w + b)
accuracy_svm = np.mean(y_pred_svm == y_svm)

print(f"学习到的权重: w={w}")
print(f"偏置: b={b:.4f}")
print(f"分类准确率: {accuracy_svm:.4f}")
print()

# ============ 核方法（RBF核） ============
print("=" * 70)
print("核方法：非线性SVM")
print("=" * 70)

# 生成非线性可分数据
n_per_class = 50
theta = np.linspace(0, 2 * np.pi, n_per_class)
r_inner = 1
r_outer = 3

X_inner = np.column_stack([r_inner * np.cos(theta), r_inner * np.sin(theta)])
X_outer = np.column_stack([r_outer * np.cos(theta), r_outer * np.sin(theta)])
X_kernel = np.vstack([X_inner, X_outer])
y_kernel = np.hstack([np.ones(n_per_class), -np.ones(n_per_class)])

# RBF核函数
def rbf_kernel(X1, X2, gamma=1.0):
    """计算RBF核矩阵"""
    sq_distances = np.sum((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]) ** 2, axis=2)
    return np.exp(-gamma * sq_distances)

# 计算核矩阵
K = rbf_kernel(X_kernel, X_kernel, gamma=0.5)

# 简化的核SVM（使用梯度下降在特征空间中）
alpha = np.random.randn(len(X_kernel)) * 0.01
b_kernel = 0.0
learning_rate_kernel = 0.001
epochs_kernel = 100
losses_kernel = []

for epoch in range(epochs_kernel):
    # 预测
    f = K @ alpha + b_kernel

    # Hinge损失
    margins = 1 - y_kernel * f
    hinge_loss = np.maximum(0, margins)
    loss = np.mean(hinge_loss) + 0.01 * np.sum(alpha ** 2)
    losses_kernel.append(loss)

    # 梯度
    mask = margins > 0
    dalpha = -K.T @ (y_kernel * mask) / len(X_kernel) + 0.01 * alpha
    db_kernel = -np.sum(y_kernel * mask) / len(X_kernel)

    # 更新
    alpha -= learning_rate_kernel * dalpha
    b_kernel -= learning_rate_kernel * db_kernel

# 计算准确率
f_pred = K @ alpha + b_kernel
y_pred_kernel = np.sign(f_pred)
accuracy_kernel = np.mean(y_pred_kernel == y_kernel)

print(f"分类准确率: {accuracy_kernel:.4f}")
print()

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 线性SVM：数据和决策边界
ax = axes[0, 0]
h = 0.02
x_min, x_max = X_svm[:, 0].min() - 1, X_svm[:, 0].max() + 1
y_min, y_max = X_svm[:, 1].min() - 1, X_svm[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z = np.sign(np.c_[xx.ravel(), yy.ravel()] @ w + b)
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=1, colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.scatter(X_svm[y_svm == 1, 0], X_svm[y_svm == 1, 1],
          c='blue', marker='o', s=30, label='Class +1')
ax.scatter(X_svm[y_svm == -1, 0], X_svm[y_svm == -1, 1],
          c='red', marker='x', s=30, label='Class -1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Linear SVM')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 线性SVM：损失曲线
ax = axes[0, 1]
ax.plot(losses_svm, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Hinge Loss')
ax.set_title('Linear SVM: Training Loss')
ax.grid(True, alpha=0.3)

# 3. 核SVM：数据和决策边界
ax = axes[1, 0]
h = 0.05
x_min, x_max = X_kernel[:, 0].min() - 1, X_kernel[:, 0].max() + 1
y_min, y_max = X_kernel[:, 1].min() - 1, X_kernel[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                     np.arange(y_min, y_max, h))
Z_points = np.c_[xx.ravel(), yy.ravel()]
K_test = rbf_kernel(Z_points, X_kernel, gamma=0.5)
Z = np.sign(K_test @ alpha + b_kernel)
Z = Z.reshape(xx.shape)

ax.contourf(xx, yy, Z, levels=1, colors=['lightblue', 'lightcoral'], alpha=0.6)
ax.scatter(X_kernel[y_kernel == 1, 0], X_kernel[y_kernel == 1, 1],
          c='blue', marker='o', s=30, label='Class +1')
ax.scatter(X_kernel[y_kernel == -1, 0], X_kernel[y_kernel == -1, 1],
          c='red', marker='x', s=30, label='Class -1')
ax.set_xlabel('Feature 1')
ax.set_ylabel('Feature 2')
ax.set_title('Kernel SVM (RBF)')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 核SVM：损失曲线
ax = axes[1, 1]
ax.plot(losses_kernel, 'g-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('Hinge Loss')
ax.set_title('Kernel SVM: Training Loss')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch02_svm_kernel.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch02_svm_kernel.png")
