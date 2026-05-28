"""
1.5 随机信号理论 - 代码实验

本实验演示：
1. 随机信号的生成和统计特性
2. 自相关函数和功率谱密度
3. 白噪声和有色噪声的区别
4. 平稳性和遍历性
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
from scipy import signal
from scipy.fft import fft, fftfreq

# 设置随机种子以保证可重复性
np.random.seed(42)

print("=" * 60)
print("1.5 随机信号理论 - 代码实验")
print("=" * 60)

# ============================================================================
# 1. 生成随机信号
# ============================================================================
print("\n1. 生成随机信号")
print("-" * 60)

N = 1000  # 信号长度
t = np.arange(N)

# 白噪声
white_noise = np.random.randn(N)

# 有色噪声（一阶AR过程）
colored_noise = np.zeros(N)
alpha = 0.8  # AR系数
for n in range(1, N):
    colored_noise[n] = alpha * colored_noise[n-1] + np.random.randn()

print(f"信号长度: {N}")
print(f"白噪声均值: {np.mean(white_noise):.4f}")
print(f"白噪声方差: {np.var(white_noise):.4f}")
print(f"有色噪声均值: {np.mean(colored_noise):.4f}")
print(f"有色噪声方差: {np.var(colored_noise):.4f}")

# ============================================================================
# 2. 计算统计特性
# ============================================================================
print("\n2. 计算统计特性")
print("-" * 60)

# 自相关函数
def compute_acf(x, max_lag=50):
    """计算自相关函数"""
    mean = np.mean(x)
    c0 = np.sum((x - mean) ** 2) / len(x)
    acf = np.zeros(max_lag + 1)
    for k in range(max_lag + 1):
        acf[k] = np.sum((x[:-k or None] - mean) * (x[k:] - mean)) / len(x) / c0
    return acf

acf_white = compute_acf(white_noise)
acf_colored = compute_acf(colored_noise)

print(f"白噪声ACF(0): {acf_white[0]:.4f}")
print(f"白噪声ACF(1): {acf_white[1]:.4f}")
print(f"有色噪声ACF(0): {acf_colored[0]:.4f}")
print(f"有色噪声ACF(1): {acf_colored[1]:.4f}")

# ============================================================================
# 3. 计算功率谱密度（PSD）
# ============================================================================
print("\n3. 计算功率谱密度")
print("-" * 60)

# 使用Welch方法估计PSD
freqs_white, psd_white = signal.welch(white_noise, nperseg=256)
freqs_colored, psd_colored = signal.welch(colored_noise, nperseg=256)

print(f"白噪声平均功率: {np.mean(psd_white):.4f}")
print(f"有色噪声平均功率: {np.mean(psd_colored):.4f}")
print(f"有色噪声低频功率: {np.mean(psd_colored[:10]):.4f}")
print(f"有色噪声高频功率: {np.mean(psd_colored[-10:]):.4f}")

# ============================================================================
# 4. 卷积对随机信号的影响
# ============================================================================
print("\n4. 卷积对随机信号的影响")
print("-" * 60)

# 设计低通滤波器
b, a = signal.butter(4, 0.2)  # 4阶Butterworth低通滤波器

# 对白噪声进行滤波
filtered_white = signal.filtfilt(b, a, white_noise)

# 对有色噪声进行滤波
filtered_colored = signal.filtfilt(b, a, colored_noise)

print(f"原始白噪声方差: {np.var(white_noise):.4f}")
print(f"滤波后白噪声方差: {np.var(filtered_white):.4f}")
print(f"原始有色噪声方差: {np.var(colored_noise):.4f}")
print(f"滤波后有色噪声方差: {np.var(filtered_colored):.4f}")

# ============================================================================
# 5. 可视化
# ============================================================================
print("\n5. 生成可视化图表")
print("-" * 60)

fig, axes = plt.subplots(3, 2, figsize=(14, 10))

# 时域信号
axes[0, 0].plot(t[:200], white_noise[:200], 'b-', linewidth=0.5)
axes[0, 0].set_title('White Noise (Time Domain)')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t[:200], colored_noise[:200], 'r-', linewidth=0.5)
axes[0, 1].set_title('Colored Noise (Time Domain)')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].grid(True, alpha=0.3)

# 自相关函数
lags = np.arange(len(acf_white))
axes[1, 0].stem(lags, acf_white, basefmt=' ')
axes[1, 0].set_title('White Noise Autocorrelation')
axes[1, 0].set_xlabel('Lag')
axes[1, 0].set_ylabel('ACF')
axes[1, 0].grid(True, alpha=0.3)

axes[1, 1].stem(lags, acf_colored, basefmt=' ')
axes[1, 1].set_title('Colored Noise Autocorrelation')
axes[1, 1].set_xlabel('Lag')
axes[1, 1].set_ylabel('ACF')
axes[1, 1].grid(True, alpha=0.3)

# 功率谱密度
axes[2, 0].semilogy(freqs_white, psd_white, 'b-', linewidth=1)
axes[2, 0].set_title('White Noise Power Spectral Density')
axes[2, 0].set_xlabel('Frequency')
axes[2, 0].set_ylabel('Power')
axes[2, 0].grid(True, alpha=0.3, which='both')

axes[2, 1].semilogy(freqs_colored, psd_colored, 'r-', linewidth=1)
axes[2, 1].set_title('Colored Noise Power Spectral Density')
axes[2, 1].set_xlabel('Frequency')
axes[2, 1].set_ylabel('Power')
axes[2, 1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('assets/ch01_random_signals.png', dpi=150, bbox_inches='tight')
print("Figure saved to assets/ch01_random_signals.png")

# ============================================================================
# 6. 卷积效果可视化
# ============================================================================
print("\n6. 卷积效果可视化")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 8))

# 原始信号
axes[0, 0].plot(t[:200], white_noise[:200], 'b-', linewidth=0.5, label='Original')
axes[0, 0].plot(t[:200], filtered_white[:200], 'r-', linewidth=1, label='Filtered')
axes[0, 0].set_title('White Noise: Low-Pass Filtering')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

axes[0, 1].plot(t[:200], colored_noise[:200], 'b-', linewidth=0.5, label='Original')
axes[0, 1].plot(t[:200], filtered_colored[:200], 'r-', linewidth=1, label='Filtered')
axes[0, 1].set_title('Colored Noise: Low-Pass Filtering')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Amplitude')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 频域对比
freqs_orig_w, psd_orig_w = signal.welch(white_noise, nperseg=256)
freqs_filt_w, psd_filt_w = signal.welch(filtered_white, nperseg=256)

axes[1, 0].semilogy(freqs_orig_w, psd_orig_w, 'b-', linewidth=1, label='Original')
axes[1, 0].semilogy(freqs_filt_w, psd_filt_w, 'r-', linewidth=1, label='Filtered')
axes[1, 0].set_title('White Noise: Frequency-Domain Comparison')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Power')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, which='both')

freqs_orig_c, psd_orig_c = signal.welch(colored_noise, nperseg=256)
freqs_filt_c, psd_filt_c = signal.welch(filtered_colored, nperseg=256)

axes[1, 1].semilogy(freqs_orig_c, psd_orig_c, 'b-', linewidth=1, label='Original')
axes[1, 1].semilogy(freqs_filt_c, psd_filt_c, 'r-', linewidth=1, label='Filtered')
axes[1, 1].set_title('Colored Noise: Frequency-Domain Comparison')
axes[1, 1].set_xlabel('Frequency')
axes[1, 1].set_ylabel('Power')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3, which='both')

plt.tight_layout()
plt.savefig('assets/ch01_convolution_effect.png', dpi=150, bbox_inches='tight')
print("Figure saved to assets/ch01_convolution_effect.png")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
