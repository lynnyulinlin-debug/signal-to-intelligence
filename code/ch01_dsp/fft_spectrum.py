"""
实验1.1：FFT频谱分析
对应章节：第1章 - 数字信号处理基础
目标：生成正弦波+噪声，用FFT观察频谱
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
SIGNAL_LENGTH = 1000
SAMPLING_RATE = 100  # Hz
SIGNAL_FREQ = [5, 10]  # Hz
NOISE_LEVEL = 0.1

# ============ 核心逻辑 ============
# 生成时间轴
t = np.arange(SIGNAL_LENGTH) / SAMPLING_RATE

# 生成多频率正弦波
signal = np.zeros(SIGNAL_LENGTH)
for freq in SIGNAL_FREQ:
    signal += np.sin(2 * np.pi * freq * t)

# 添加噪声
signal_noisy = signal + NOISE_LEVEL * np.random.randn(SIGNAL_LENGTH)

# 计算FFT
fft_result = np.fft.fft(signal_noisy)
frequencies = np.fft.fftfreq(SIGNAL_LENGTH, 1 / SAMPLING_RATE)
magnitude = np.abs(fft_result)

# 只取正频率部分
positive_freq_idx = frequencies > 0
frequencies_positive = frequencies[positive_freq_idx]
magnitude_positive = magnitude[positive_freq_idx]

# 找到主要频率分量（幅度最大的前2个）
top_indices = np.argsort(magnitude_positive)[-2:][::-1]
detected_freqs = frequencies_positive[top_indices]

# ============ 结果输出 ============
print("=" * 50)
print("FFT 频谱分析")
print("=" * 50)
print(f"信号长度: {SIGNAL_LENGTH}")
print(f"采样率: {SAMPLING_RATE} Hz")
print(f"真实频率: {SIGNAL_FREQ}")
print(f"检测到的频率: {detected_freqs.round(1)}")
print(f"噪声水平: {NOISE_LEVEL}")
print("=" * 50)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 1, figsize=(12, 8))

# 时域信号
axes[0].plot(t[:200], signal_noisy[:200], 'b-', linewidth=0.8, label='Noisy Signal')
axes[0].plot(t[:200], signal[:200], 'r--', linewidth=1, label='Clean Signal')
axes[0].set_xlabel('Time (s)')
axes[0].set_ylabel('Amplitude')
axes[0].set_title('Time Domain Signal (first 2 seconds)')
axes[0].legend()
axes[0].grid(True, alpha=0.3)

# 频域信号
axes[1].plot(frequencies_positive[:100], magnitude_positive[:100], 'b-', linewidth=1)
axes[1].scatter(detected_freqs, magnitude_positive[top_indices], color='r', s=100,
                label=f'Detected: {detected_freqs.round(1)} Hz', zorder=5)
axes[1].set_xlabel('Frequency (Hz)')
axes[1].set_ylabel('Magnitude')
axes[1].set_title('Frequency Domain (FFT)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_fft_spectrum.png', dpi=100, bbox_inches='tight')
print("\n图表已保存到: assets/ch01_fft_spectrum.png")
