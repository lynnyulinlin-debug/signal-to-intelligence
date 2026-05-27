"""
实验2.2：LMS vs Adam
对应章节：第2章 - 优化算法与传统机器学习
目标：实现LMS算法求解线性回归，与Adam优化器对比收敛曲线
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
N_SAMPLES = 200
N_FEATURES = 5
LEARNING_RATE_LMS = 0.01
LEARNING_RATE_ADAM = 0.01
EPOCHS = 100

# ============ 核心逻辑 ============
# 生成合成数据：y = w_true * x + noise
w_true = np.random.randn(N_FEATURES)
X = np.random.randn(N_SAMPLES, N_FEATURES)
y = X @ w_true + 0.1 * np.random.randn(N_SAMPLES)

# 方法1：LMS（最小均方）算法
# w_{n+1} = w_n + 2 * μ * e_n * x_n
# 其中 e_n = y_n - w_n^T * x_n 是误差，μ 是步长
w_lms = np.zeros(N_FEATURES)
losses_lms = []

for epoch in range(EPOCHS):
    # 遍历每个样本（在线学习）
    for i in range(N_SAMPLES):
        x_i = X[i]  # (N_FEATURES,)
        y_i = y[i]  # scalar

        # 预测和误差
        y_pred = x_i @ w_lms
        error = y_i - y_pred

        # LMS更新
        w_lms += 2 * LEARNING_RATE_LMS * error * x_i

    # 计算整个数据集的MSE
    y_pred_lms = X @ w_lms
    loss_lms = np.mean((y - y_pred_lms) ** 2)
    losses_lms.append(loss_lms)

# 方法2：Adam优化器
# 维护一阶矩估计 m 和二阶矩估计 v
# m_t = β1 * m_{t-1} + (1 - β1) * g_t
# v_t = β2 * v_{t-1} + (1 - β2) * g_t^2
# w_t = w_{t-1} - α * m_t / (sqrt(v_t) + ε)
w_adam = np.zeros(N_FEATURES)
m = np.zeros(N_FEATURES)  # 一阶矩
v = np.zeros(N_FEATURES)  # 二阶矩
beta1 = 0.9
beta2 = 0.999
epsilon = 1e-8
losses_adam = []

for epoch in range(EPOCHS):
    # 计算梯度（批量）
    y_pred_adam = X @ w_adam
    error = y_pred_adam - y
    grad = X.T @ error / N_SAMPLES

    # Adam更新
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * (grad ** 2)

    # 偏差修正
    m_hat = m / (1 - beta1 ** (epoch + 1))
    v_hat = v / (1 - beta2 ** (epoch + 1))

    # 参数更新
    w_adam -= LEARNING_RATE_ADAM * m_hat / (np.sqrt(v_hat) + epsilon)

    # 计算MSE
    y_pred_adam = X @ w_adam
    loss_adam = np.mean((y - y_pred_adam) ** 2)
    losses_adam.append(loss_adam)

# ============ 结果输出 ============
print("=" * 70)
print("LMS vs Adam 优化器对比")
print("=" * 70)
print(f"样本数: {N_SAMPLES}")
print(f"特征数: {N_FEATURES}")
print(f"训练轮数: {EPOCHS}")
print()

print("真实权重:")
print("-" * 70)
print(f"w_true: {w_true.round(4)}")
print()

print("LMS 算法结果:")
print("-" * 70)
print(f"学习到的权重: {w_lms.round(4)}")
print(f"最终MSE: {losses_lms[-1]:.6f}")
print(f"权重误差: {np.linalg.norm(w_lms - w_true):.6f}")
print()

print("Adam 优化器结果:")
print("-" * 70)
print(f"学习到的权重: {w_adam.round(4)}")
print(f"最终MSE: {losses_adam[-1]:.6f}")
print(f"权重误差: {np.linalg.norm(w_adam - w_true):.6f}")
print()

print("收敛对比:")
print("-" * 70)
print(f"LMS 初始MSE: {losses_lms[0]:.6f} → 最终MSE: {losses_lms[-1]:.6f}")
print(f"Adam 初始MSE: {losses_adam[0]:.6f} → 最终MSE: {losses_adam[-1]:.6f}")
print(f"LMS 收敛速度: {(losses_lms[0] - losses_lms[-1]) / losses_lms[0] * 100:.2f}%")
print(f"Adam 收敛速度: {(losses_adam[0] - losses_adam[-1]) / losses_adam[0] * 100:.2f}%")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 收敛曲线对比
ax = axes[0, 0]
ax.plot(losses_lms, 'b-', linewidth=2, label='LMS', alpha=0.8)
ax.plot(losses_adam, 'r-', linewidth=2, label='Adam', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Convergence Comparison')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 对数尺度收敛曲线
ax = axes[0, 1]
ax.semilogy(losses_lms, 'b-', linewidth=2, label='LMS', alpha=0.8)
ax.semilogy(losses_adam, 'r-', linewidth=2, label='Adam', alpha=0.8)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Convergence Comparison (Log Scale)')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# 3. 权重估计误差
ax = axes[1, 0]
weight_error_lms = np.abs(w_lms - w_true)
weight_error_adam = np.abs(w_adam - w_true)
x_pos = np.arange(N_FEATURES)
width = 0.35
ax.bar(x_pos - width/2, weight_error_lms, width, label='LMS', alpha=0.8)
ax.bar(x_pos + width/2, weight_error_adam, width, label='Adam', alpha=0.8)
ax.set_xlabel('Feature Index')
ax.set_ylabel('Absolute Error')
ax.set_title('Weight Estimation Error')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. 预测值 vs 真实值
ax = axes[1, 1]
y_pred_lms_final = X @ w_lms
y_pred_adam_final = X @ w_adam
ax.scatter(y, y_pred_lms_final, alpha=0.5, s=30, label='LMS Prediction')
ax.scatter(y, y_pred_adam_final, alpha=0.5, s=30, label='Adam Prediction')
ax.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', linewidth=1, label='Perfect Prediction')
ax.set_xlabel('True Value')
ax.set_ylabel('Predicted Value')
ax.set_title('Prediction Accuracy')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch02_lms_vs_adam.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch02_lms_vs_adam.png")
