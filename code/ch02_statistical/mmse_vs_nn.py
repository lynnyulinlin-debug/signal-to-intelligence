"""
实验2.1：MMSE vs 神经网络
对应章节：第2章 - 统计信号处理
目标：对加噪正弦波，分别用MMSE公式和单层神经网络估计原始信号
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
SIGNAL_LENGTH = 500
SIGNAL_FREQ = 5  # Hz
SAMPLING_RATE = 100  # Hz
NOISE_VARIANCE = 0.5
SIGNAL_VARIANCE = 1.0

# ============ 核心逻辑 ============
# 生成时间轴和原始信号
t = np.arange(SIGNAL_LENGTH) / SAMPLING_RATE
signal_clean = np.sin(2 * np.pi * SIGNAL_FREQ * t)

# 添加高斯噪声
noise = np.sqrt(NOISE_VARIANCE) * np.random.randn(SIGNAL_LENGTH)
signal_noisy = signal_clean + noise

# 方法1：MMSE估计（最小均方误差）
# 假设信号和噪声都是高斯分布
# MMSE估计：x_hat = (σ_s^2 / (σ_s^2 + σ_n^2)) * y
# 其中 σ_s^2 是信号方差，σ_n^2 是噪声方差，y 是观测值
mmse_gain = SIGNAL_VARIANCE / (SIGNAL_VARIANCE + NOISE_VARIANCE)
signal_mmse = mmse_gain * signal_noisy

# 方法2：单层神经网络（线性回归）
# 使用梯度下降训练一个简单的线性模型：y_hat = w * x + b
# 这实际上会学到类似MMSE的增益
learning_rate = 0.01
epochs = 100
w = np.random.randn() * 0.1
b = np.random.randn() * 0.1

# 训练过程
losses = []
for epoch in range(epochs):
    # 前向传播
    signal_pred = w * signal_noisy + b

    # 计算MSE损失
    loss = np.mean((signal_pred - signal_clean) ** 2)
    losses.append(loss)

    # 反向传播（梯度下降）
    grad_w = -2 * np.mean((signal_clean - signal_pred) * signal_noisy)
    grad_b = -2 * np.mean(signal_clean - signal_pred)

    # 更新参数
    w -= learning_rate * grad_w
    b -= learning_rate * grad_b

signal_nn = w * signal_noisy + b

# ============ 结果输出 ============
print("=" * 70)
print("MMSE vs 神经网络估计")
print("=" * 70)
print(f"信号长度: {SIGNAL_LENGTH}")
print(f"信号频率: {SIGNAL_FREQ} Hz")
print(f"信号方差: {SIGNAL_VARIANCE}")
print(f"噪声方差: {NOISE_VARIANCE}")
print()

# 计算性能指标
mse_noisy = np.mean((signal_noisy - signal_clean) ** 2)
mse_mmse = np.mean((signal_mmse - signal_clean) ** 2)
mse_nn = np.mean((signal_nn - signal_clean) ** 2)

print("性能对比：")
print("-" * 70)
print(f"原始观测 MSE: {mse_noisy:.4f}")
print(f"MMSE 估计 MSE: {mse_mmse:.4f}")
print(f"神经网络 MSE: {mse_nn:.4f}")
print()

print("MMSE 参数：")
print("-" * 70)
print(f"理论增益: {mmse_gain:.4f}")
print()

print("神经网络参数：")
print("-" * 70)
print(f"学习到的权重 w: {w:.4f}")
print(f"学习到的偏置 b: {b:.4f}")
print(f"最终训练损失: {losses[-1]:.4f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 时域信号对比
ax = axes[0, 0]
ax.plot(t[:100], signal_clean[:100], 'k-', linewidth=2, label='Clean Signal')
ax.plot(t[:100], signal_noisy[:100], 'r.', markersize=4, alpha=0.6, label='Noisy Signal')
ax.plot(t[:100], signal_mmse[:100], 'b-', linewidth=1.5, label='MMSE Estimate')
ax.plot(t[:100], signal_nn[:100], 'g--', linewidth=1.5, label='NN Estimate')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Signal Estimation (first 1 second)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 估计误差对比
ax = axes[0, 1]
error_mmse = signal_mmse - signal_clean
error_nn = signal_nn - signal_clean
ax.plot(error_mmse, 'b-', linewidth=1, alpha=0.7, label='MMSE Error')
ax.plot(error_nn, 'g-', linewidth=1, alpha=0.7, label='NN Error')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Sample')
ax.set_ylabel('Error')
ax.set_title('Estimation Error')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 训练损失曲线
ax = axes[1, 0]
ax.plot(losses, 'b-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Neural Network Training Loss')
ax.grid(True, alpha=0.3)

# 4. MSE对比柱状图
ax = axes[1, 1]
methods = ['Noisy\nObservation', 'MMSE\nEstimate', 'NN\nEstimate']
mses = [mse_noisy, mse_mmse, mse_nn]
colors = ['red', 'blue', 'green']
bars = ax.bar(methods, mses, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
ax.set_ylabel('MSE')
ax.set_title('Performance Comparison')
ax.grid(True, alpha=0.3, axis='y')

# 添加数值标签
for bar, mse in zip(bars, mses):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{mse:.4f}', ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.savefig('assets/ch02_mmse_vs_nn.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch02_mmse_vs_nn.png")
