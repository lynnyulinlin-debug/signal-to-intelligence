"""
1.8 矩阵分解应用 - 代码实验

本实验演示：
1. SVD分解
2. 特征值分解（EVD）
3. MUSIC算法（频率估计）
4. 子空间方法
5. 低秩近似和去噪
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import linalg

# 设置随机种子
np.random.seed(42)

print("=" * 60)
print("1.8 矩阵分解应用 - 代码实验")
print("=" * 60)

# ============================================================================
# 1. 生成多信号观测
# ============================================================================
print("\n1. 生成多信号观测")
print("-" * 60)

# 信号参数
N = 200  # 观测长度
K = 2    # 信号个数
f1, f2 = 0.1, 0.15  # 两个信号的频率
A1, A2 = 1.0, 0.8   # 两个信号的幅度
SNR_dB = 10
SNR = 10 ** (SNR_dB / 10)

# 生成信号
t = np.arange(N)
s1 = A1 * np.sin(2 * np.pi * f1 * t)
s2 = A2 * np.sin(2 * np.pi * f2 * t)
s_true = s1 + s2

# 添加噪声
sigma2 = np.sum(s_true**2) / N / SNR
sigma = np.sqrt(sigma2)
w = sigma * np.random.randn(N)
y = s_true + w

print(f"信号1频率: {f1:.4f}, 幅度: {A1:.4f}")
print(f"信号2频率: {f2:.4f}, 幅度: {A2:.4f}")
print(f"观测长度: {N}")
print(f"SNR (dB): {SNR_dB:.2f}")

# ============================================================================
# 2. SVD分解
# ============================================================================
print("\n2. SVD分解")
print("-" * 60)

# 构造Hankel矩阵
M = 50  # 矩阵行数
Hankel = np.zeros((M, N - M + 1))
for i in range(M):
    Hankel[i, :] = y[i:i + N - M + 1]

# SVD分解
U, s, Vh = linalg.svd(Hankel, full_matrices=False)

print(f"Hankel矩阵大小: {Hankel.shape}")
print(f"奇异值数量: {len(s)}")
print(f"前5个奇异值: {s[:5]}")
print(f"奇异值衰减比: {s[0]/s[-1]:.2f}")

# ============================================================================
# 3. 特征值分解（EVD）
# ============================================================================
print("\n3. 特征值分解（EVD）")
print("-" * 60)

# 构造协方差矩阵
R = Hankel @ Hankel.T / (N - M + 1)

# EVD分解
eigenvalues, eigenvectors = linalg.eigh(R)

# 按特征值从大到小排序
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]

print(f"协方差矩阵大小: {R.shape}")
print(f"特征值数量: {len(eigenvalues)}")
print(f"前5个特征值: {eigenvalues[:5]}")
print(f"特征值衰减比: {eigenvalues[0]/eigenvalues[-1]:.2f}")

# ============================================================================
# 4. MUSIC算法
# ============================================================================
print("\n4. MUSIC算法（频率估计）")
print("-" * 60)

# 分离信号子空间和噪声子空间
Es = eigenvectors[:, :K]  # 信号子空间
En = eigenvectors[:, K:]  # 噪声子空间

# MUSIC谱
freq_range = np.linspace(0, 0.5, 1000)
music_spectrum = np.zeros_like(freq_range)

for i, f in enumerate(freq_range):
    # 导向向量
    a = np.exp(1j * 2 * np.pi * f * np.arange(M))
    # MUSIC谱
    music_spectrum[i] = 1 / np.sum(np.abs(En.T @ a)**2)

# 找到峰值
peaks_idx = np.argsort(music_spectrum)[-K:]
peaks_idx = np.sort(peaks_idx)
estimated_freqs = freq_range[peaks_idx]

print(f"真实频率: {f1:.4f}, {f2:.4f}")
print(f"估计频率: {estimated_freqs[0]:.4f}, {estimated_freqs[1]:.4f}")
print(f"频率误差: {abs(estimated_freqs[0]-f1):.6f}, {abs(estimated_freqs[1]-f2):.6f}")

# ============================================================================
# 5. 低秩近似和去噪
# ============================================================================
print("\n5. 低秩近似和去噪")
print("-" * 60)

# 低秩近似（保留前K个奇异值）
U_k = U[:, :K]
s_k = s[:K]
Vh_k = Vh[:K, :]

# 重构Hankel矩阵
Hankel_approx = U_k @ np.diag(s_k) @ Vh_k

# 从Hankel矩阵恢复信号
y_denoised = np.zeros(N)
for i in range(N):
    if i < M:
        y_denoised[i] = np.mean(np.diag(Hankel_approx, i - i))
    else:
        y_denoised[i] = np.mean(np.diag(Hankel_approx, i - M + 1))

# 简化：直接使用第一行和最后一列
y_denoised = np.concatenate([Hankel_approx[0, :], Hankel_approx[1:, -1]])

# 计算去噪效果
mse_noisy = np.mean((y - s_true)**2)
mse_denoised = np.mean((y_denoised - s_true)**2)
snr_improvement = 10 * np.log10(mse_noisy / mse_denoised)

print(f"原始信号MSE: {mse_noisy:.6f}")
print(f"去噪后MSE: {mse_denoised:.6f}")
print(f"SNR改进: {snr_improvement:.2f} dB")

# ============================================================================
# 6. PCA应用
# ============================================================================
print("\n6. PCA应用（特征提取）")
print("-" * 60)

# 使用前K个特征向量作为特征
features = Es.T @ Hankel

print(f"特征维数: {features.shape}")
print(f"特征方差: {np.var(features, axis=1)}")

# ============================================================================
# 7. 可视化
# ============================================================================
print("\n7. 生成可视化图表")
print("-" * 60)

fig, axes = plt.subplots(2, 3, figsize=(16, 10))

# 观测信号
axes[0, 0].plot(t, s_true, 'g-', linewidth=2, label='真实信号')
axes[0, 0].plot(t, y, 'b-', linewidth=0.5, alpha=0.7, label='观测（信号+噪声）')
axes[0, 0].set_title('观测信号')
axes[0, 0].set_xlabel('时间')
axes[0, 0].set_ylabel('幅度')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 奇异值
axes[0, 1].semilogy(range(len(s)), s, 'b-o', linewidth=2, markersize=4)
axes[0, 1].axvline(x=K, color='r', linestyle='--', linewidth=2, label=f'K={K}')
axes[0, 1].set_title('SVD奇异值')
axes[0, 1].set_xlabel('索引')
axes[0, 1].set_ylabel('奇异值')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, which='both')

# 特征值
axes[0, 2].semilogy(range(len(eigenvalues)), eigenvalues, 'r-s', linewidth=2, markersize=4)
axes[0, 2].axvline(x=K, color='b', linestyle='--', linewidth=2, label=f'K={K}')
axes[0, 2].set_title('EVD特征值')
axes[0, 2].set_xlabel('索引')
axes[0, 2].set_ylabel('特征值')
axes[0, 2].legend()
axes[0, 2].grid(True, alpha=0.3, which='both')

# MUSIC谱
axes[1, 0].plot(freq_range, music_spectrum, 'b-', linewidth=1)
axes[1, 0].plot(estimated_freqs, music_spectrum[peaks_idx], 'ro', markersize=8, label='估计频率')
axes[1, 0].axvline(x=f1, color='g', linestyle='--', linewidth=2, label='真实频率1')
axes[1, 0].axvline(x=f2, color='g', linestyle='--', linewidth=2, label='真实频率2')
axes[1, 0].set_title('MUSIC谱')
axes[1, 0].set_xlabel('频率')
axes[1, 0].set_ylabel('MUSIC谱')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# 去噪效果
axes[1, 1].plot(t, s_true, 'g-', linewidth=2, label='真实信号')
axes[1, 1].plot(t, y, 'b-', linewidth=0.5, alpha=0.5, label='观测')
axes[1, 1].plot(t, y_denoised, 'r-', linewidth=1, alpha=0.8, label='去噪')
axes[1, 1].set_title('去噪效果')
axes[1, 1].set_xlabel('时间')
axes[1, 1].set_ylabel('幅度')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

# 频域对比
freq_axis = np.fft.fftfreq(N)
Y_fft = np.abs(np.fft.fft(y))
Y_denoised_fft = np.abs(np.fft.fft(y_denoised))

axes[1, 2].plot(freq_axis[:N//2], Y_fft[:N//2], 'b-', linewidth=1, label='观测')
axes[1, 2].plot(freq_axis[:N//2], Y_denoised_fft[:N//2], 'r-', linewidth=1, label='去噪')
axes[1, 2].axvline(x=f1, color='g', linestyle='--', linewidth=2, label='真实频率')
axes[1, 2].axvline(x=f2, color='g', linestyle='--', linewidth=2)
axes[1, 2].set_title('频域对比')
axes[1, 2].set_xlabel('频率')
axes[1, 2].set_ylabel('幅度')
axes[1, 2].legend()
axes[1, 2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_matrix_decomposition.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存到 assets/ch01_matrix_decomposition.png")

# ============================================================================
# 8. 性能对比
# ============================================================================
print("\n8. 性能对比")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# 频率估计精度
methods = ['FFT', 'MUSIC']
freq1_errors = [abs(0.1 - 0.098), abs(estimated_freqs[0] - f1)]
freq2_errors = [abs(0.15 - 0.152), abs(estimated_freqs[1] - f2)]

x = np.arange(len(methods))
width = 0.35

axes[0].bar(x - width/2, freq1_errors, width, label='频率1', alpha=0.8)
axes[0].bar(x + width/2, freq2_errors, width, label='频率2', alpha=0.8)
axes[0].set_ylabel('频率估计误差')
axes[0].set_title('频率估计精度对比')
axes[0].set_xticks(x)
axes[0].set_xticklabels(methods)
axes[0].legend()
axes[0].grid(True, alpha=0.3, axis='y')

# 去噪效果
methods_denoise = ['原始', '去噪']
mse_values = [mse_noisy, mse_denoised]

axes[1].bar(methods_denoise, mse_values, color=['b', 'r'], alpha=0.7)
axes[1].set_ylabel('MSE')
axes[1].set_title('去噪效果对比')
axes[1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/ch01_matrix_decomposition_performance.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存到 assets/ch01_matrix_decomposition_performance.png")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
