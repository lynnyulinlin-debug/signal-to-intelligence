"""
实验2.1：MMSE vs 神经网络
对应章节：第2章 - 优化算法与传统机器学习
目标：对比最小均方误差（MMSE）和神经网络在信号估计中的性能
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
np.random.seed(42)
N_SAMPLES = 500
SNR_DB = 10  # 信噪比（dB）
NOISE_POWER = 10 ** (-SNR_DB / 10)

# ============ 核心逻辑 ============
# 生成真实信号：高斯随机信号
s_true = np.random.randn(N_SAMPLES)

# 添加噪声
noise = np.sqrt(NOISE_POWER) * np.random.randn(N_SAMPLES)
y = s_true + noise

# 方法1：MMSE（最小均方误差）估计
# 对于高斯信号+高斯噪声，MMSE估计是线性的：
# s_hat = (σ_s^2 / (σ_s^2 + σ_n^2)) * y
# 其中 σ_s^2 是信号方差，σ_n^2 是噪声方差
sigma_s_sq = np.var(s_true)
sigma_n_sq = NOISE_POWER
mmse_gain = sigma_s_sq / (sigma_s_sq + sigma_n_sq)
s_mmse = mmse_gain * y

# 计算MMSE性能
mse_mmse = np.mean((s_true - s_mmse) ** 2)
snr_out_mmse = np.var(s_mmse) / np.mean((s_true - s_mmse) ** 2)

# 方法2：简单神经网络（1层隐层）
# 网络结构：输入 -> 隐层(16) -> 输出
# 使用梯度下降训练
np.random.seed(42)
hidden_size = 16
learning_rate = 0.01
epochs = 200

# 初始化权重
W1 = np.random.randn(1, hidden_size) * 0.1
b1 = np.zeros((1, hidden_size))
W2 = np.random.randn(hidden_size, 1) * 0.1
b2 = np.zeros((1, 1))

losses_nn = []

for epoch in range(epochs):
    # 前向传播
    z1 = y.reshape(-1, 1) @ W1 + b1  # (N, 16)
    a1 = np.tanh(z1)  # 激活函数
    z2 = a1 @ W2 + b2  # (N, 1)
    s_pred = z2.flatten()  # (N,)

    # 计算损失
    loss = np.mean((s_true - s_pred) ** 2)
    losses_nn.append(loss)

    # 反向传播
    dz2 = (s_pred - s_true).reshape(-1, 1) / N_SAMPLES  # (N, 1)
    dW2 = a1.T @ dz2  # (16, 1)
    db2 = np.sum(dz2, axis=0, keepdims=True)  # (1, 1)

    da1 = dz2 @ W2.T  # (N, 16)
    dz1 = da1 * (1 - a1 ** 2)  # tanh导数
    dW1 = y.reshape(-1, 1).T @ dz1  # (1, 16)
    db1 = np.sum(dz1, axis=0, keepdims=True)  # (1, 16)

    # 更新权重
    W1 -= learning_rate * dW1
    b1 -= learning_rate * db1
    W2 -= learning_rate * dW2
    b2 -= learning_rate * db2

# 最终预测
z1 = y.reshape(-1, 1) @ W1 + b1
a1 = np.tanh(z1)
z2 = a1 @ W2 + b2
s_nn = z2.flatten()

# 计算NN性能
mse_nn = np.mean((s_true - s_nn) ** 2)
snr_out_nn = np.var(s_nn) / np.mean((s_true - s_nn) ** 2)

# ============ 结果输出 ============
print("=" * 70)
print("MMSE vs Neural Network: Estimation Performance")
print("=" * 70)
print(f"样本数: {N_SAMPLES}")
print(f"输入信噪比: {SNR_DB} dB")
print(f"信号方差: {sigma_s_sq:.4f}")
print(f"噪声方差: {sigma_n_sq:.4f}")
print()

print("MMSE 估计器:")
print("-" * 70)
print(f"MMSE增益: {mmse_gain:.4f}")
print(f"输出MSE: {mse_mmse:.6f}")
print(f"输出信噪比: {10 * np.log10(snr_out_mmse):.2f} dB")
print()

print("神经网络估计器:")
print("-" * 70)
print(f"隐层大小: {hidden_size}")
print(f"训练轮数: {epochs}")
print(f"学习率: {learning_rate}")
print(f"输出MSE: {mse_nn:.6f}")
print(f"输出信噪比: {10 * np.log10(snr_out_nn):.2f} dB")
print()

print("性能对比:")
print("-" * 70)
mse_improvement = (mse_mmse - mse_nn) / mse_mmse * 100
print(f"MSE改进: {mse_improvement:.2f}% {'(NN更好)' if mse_improvement > 0 else '(MMSE更好)'}")
snr_improvement = (snr_out_nn - snr_out_mmse) / snr_out_mmse * 100
print(f"SNR改进: {snr_improvement:.2f}% {'(NN更好)' if snr_improvement > 0 else '(MMSE更好)'}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 信号对比（前100个样本）
ax = axes[0, 0]
idx = slice(0, 100)
ax.plot(s_true[idx], 'k-', linewidth=2, label='True Signal', alpha=0.7)
ax.plot(y[idx], 'gray', linewidth=0.8, label='Noisy Signal', alpha=0.5)
ax.plot(s_mmse[idx], 'b--', linewidth=1.5, label='MMSE Estimate', alpha=0.8)
ax.plot(s_nn[idx], 'r--', linewidth=1.5, label='NN Estimate', alpha=0.8)
ax.set_xlabel('Sample Index')
ax.set_ylabel('Amplitude')
ax.set_title('Signal Estimation (First 100 Samples)')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 训练曲线
ax = axes[0, 1]
ax.semilogy(losses_nn, 'r-', linewidth=2, label='NN Training Loss')
ax.axhline(mse_mmse, color='b', linestyle='--', linewidth=2, label=f'MMSE MSE ({mse_mmse:.6f})')
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss (log scale)')
ax.set_title('Training Convergence')
ax.legend()
ax.grid(True, alpha=0.3, which='both')

# 3. 误差分布
ax = axes[1, 0]
error_mmse = s_true - s_mmse
error_nn = s_true - s_nn
ax.hist(error_mmse, bins=30, alpha=0.6, label='MMSE Error', color='b', density=True)
ax.hist(error_nn, bins=30, alpha=0.6, label='NN Error', color='r', density=True)
ax.set_xlabel('Estimation Error')
ax.set_ylabel('Probability Density')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. 性能指标对比
ax = axes[1, 1]
methods = ['MMSE', 'Neural Network']
mse_values = [mse_mmse, mse_nn]
snr_values = [10 * np.log10(snr_out_mmse), 10 * np.log10(snr_out_nn)]

x_pos = np.arange(len(methods))
width = 0.35

ax2 = ax.twinx()
bars1 = ax.bar(x_pos - width/2, mse_values, width, label='MSE', color='steelblue', alpha=0.8)
bars2 = ax2.bar(x_pos + width/2, snr_values, width, label='Output SNR (dB)', color='coral', alpha=0.8)

ax.set_ylabel('MSE', color='steelblue')
ax2.set_ylabel('Output SNR (dB)', color='coral')
ax.set_title('Performance Comparison')
ax.set_xticks(x_pos)
ax.set_xticklabels(methods)
ax.tick_params(axis='y', labelcolor='steelblue')
ax2.tick_params(axis='y', labelcolor='coral')
ax.grid(True, alpha=0.3, axis='y')

# 添加图例
lines1, labels1 = ax.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.savefig('assets/ch02_mmse_vs_nn.png', dpi=100, bbox_inches='tight')
print("Figure saved to: assets/ch02_mmse_vs_nn.png")
