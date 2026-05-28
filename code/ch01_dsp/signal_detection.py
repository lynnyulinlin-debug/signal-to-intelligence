"""
1.6 信号检测 - 代码实验

本实验演示：
1. 能量检测器
2. 相关检测器
3. 匹配滤波器
4. ROC曲线
5. 检测性能分析
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["axes.unicode_minus"] = False
from scipy import signal
from scipy.special import erfc

# 设置随机种子
np.random.seed(42)

print("=" * 60)
print("1.6 信号检测 - 代码实验")
print("=" * 60)

# ============================================================================
# 1. 生成测试信号
# ============================================================================
print("\n1. 生成测试信号")
print("-" * 60)

N = 40  # 信号长度
SNR_dB = 5  # 信噪比（dB）
SNR = 10 ** (SNR_dB / 10)  # 转换为线性

# 已知信号（正弦波）
f0 = 0.1  # 归一化频率
t = np.arange(N)
s = np.sin(2 * np.pi * f0 * t)

# 噪声功率
sigma2 = np.sum(s**2) / N / SNR
sigma = np.sqrt(sigma2)

# 生成观测信号（H1：信号+噪声）
w = sigma * np.random.randn(N)
y = s + w

print(f"信号长度: {N}")
print(f"信号功率: {np.sum(s**2)/N:.4f}")
print(f"噪声功率: {sigma2:.4f}")
print(f"SNR (dB): {SNR_dB:.2f}")
print(f"SNR (线性): {SNR:.4f}")

# ============================================================================
# 2. 能量检测器
# ============================================================================
print("\n2. 能量检测器")
print("-" * 60)

def energy_detector(y, threshold):
    """能量检测器"""
    T = np.sum(y**2)
    return T > threshold

# 计算能量
energy = np.sum(y**2)
print(f"观测信号能量: {energy:.4f}")

# 设置阈值（基于噪声功率）
threshold_energy = N * sigma2 * 2  # 简单阈值
decision_energy = energy_detector(y, threshold_energy)
print(f"能量检测阈值: {threshold_energy:.4f}")
print(f"能量检测决策: {'H1 (信号存在)' if decision_energy else 'H0 (无信号)'}")

# ============================================================================
# 3. 相关检测器
# ============================================================================
print("\n3. 相关检测器")
print("-" * 60)

def correlation_detector(y, s, threshold):
    """相关检测器"""
    T = np.sum(y * s)
    return T > threshold

# 计算相关性
correlation = np.sum(y * s)
print(f"相关统计量: {correlation:.4f}")

# 设置阈值
threshold_corr = np.sum(s**2) * sigma * 2  # 简单阈值
decision_corr = correlation_detector(y, s, threshold_corr)
print(f"相关检测阈值: {threshold_corr:.4f}")
print(f"相关检测决策: {'H1 (信号存在)' if decision_corr else 'H0 (无信号)'}")

# ============================================================================
# 4. 匹配滤波器
# ============================================================================
print("\n4. 匹配滤波器")
print("-" * 60)

def matched_filter(y, s):
    """匹配滤波器"""
    # 冲激响应：h[n] = s[N-1-n]
    h = s[::-1]
    # 卷积
    output = signal.correlate(y, h, mode='same')
    return output

# 应用匹配滤波器
mf_output = matched_filter(y, s)
mf_max = np.max(mf_output)
mf_max_idx = np.argmax(mf_output)

print(f"匹配滤波器最大输出: {mf_max:.4f}")
print(f"最大输出位置: {mf_max_idx}")
print(f"匹配滤波器决策: {'H1 (信号存在)' if mf_max > threshold_corr else 'H0 (无信号)'}")

# ============================================================================
# 5. ROC曲线
# ============================================================================
print("\n5. ROC曲线分析")
print("-" * 60)

# 检测性能随SNR变化：对每个SNR用“平均值 + 固定噪声裕量”设阈值，
# 这样Pd会随SNR提升，而Pfa也会保持在可见但不过高的区间。
SNR_range = np.linspace(-10, 12, 20)
Pd_list = []
Pfa_list = []
num_trials = 3000
margin_scale = 1.15
signal_energy = np.sum(s**2)

for snr_db in SNR_range:
    snr = 10 ** (snr_db / 10)
    sigma_temp = np.sqrt(signal_energy / N / snr)
    threshold_temp = margin_scale * sigma_temp * np.sqrt(signal_energy)
    detections_h1 = 0
    detections_h0 = 0

    for _ in range(num_trials):
        w_h1 = sigma_temp * np.random.randn(N)
        y_h1 = s + w_h1
        T_h1 = np.sum(y_h1 * s)
        if T_h1 > threshold_temp:
            detections_h1 += 1

        w_h0 = sigma_temp * np.random.randn(N)
        T_h0 = np.sum(w_h0 * s)
        if T_h0 > threshold_temp:
            detections_h0 += 1

    Pd_list.append(detections_h1 / num_trials)
    Pfa_list.append(detections_h0 / num_trials)

# 真实ROC：固定SNR，扫描不同阈值
roc_snr_db = 5
roc_snr = 10 ** (roc_snr_db / 10)
roc_sigma = np.sqrt(np.sum(s**2) / N / roc_snr)
roc_trials = 4000
T_h1_samples = []
T_h0_samples = []

for _ in range(roc_trials):
    w_h1 = roc_sigma * np.random.randn(N)
    y_h1 = s + w_h1
    T_h1_samples.append(np.sum(y_h1 * s))

    w_h0 = roc_sigma * np.random.randn(N)
    T_h0_samples.append(np.sum(w_h0 * s))

T_h1_samples = np.array(T_h1_samples)
T_h0_samples = np.array(T_h0_samples)
thresholds_roc = np.linspace(T_h0_samples.min(), T_h1_samples.max(), 80)
roc_pd_list = []
roc_pfa_list = []

for threshold in thresholds_roc:
    roc_pd_list.append(np.mean(T_h1_samples > threshold))
    roc_pfa_list.append(np.mean(T_h0_samples > threshold))

print(f"SNR range: {SNR_range[0]:.1f} - {SNR_range[-1]:.1f} dB")
print(f"Pd at lowest SNR: {Pd_list[0]:.4f}")
print(f"Pd at highest SNR: {Pd_list[-1]:.4f}")
print(f"Average Pfa across SNR sweep: {np.mean(Pfa_list):.4f}")
print(f"ROC SNR: {roc_snr_db:.1f} dB")
print(f"ROC Pfa range: {min(roc_pfa_list):.4f} - {max(roc_pfa_list):.4f}")

# ============================================================================
# 6. 可视化
# ============================================================================
print("\n6. 生成可视化图表")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 时域信号
axes[0, 0].plot(t, s, 'g-', linewidth=2, label='Signal')
axes[0, 0].plot(t, y, 'b-', linewidth=0.5, alpha=0.7, label='Observation (Signal + Noise)')
axes[0, 0].plot(t, w, 'r-', linewidth=0.5, alpha=0.5, label='Noise')
axes[0, 0].set_title('Time-Domain Signals')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 匹配滤波器输出
axes[0, 1].plot(t, mf_output, 'b-', linewidth=1)
axes[0, 1].axhline(y=threshold_corr, color='r', linestyle='--', linewidth=2, label='Threshold')
axes[0, 1].plot(mf_max_idx, mf_max, 'ro', markersize=8, label='Peak')
axes[0, 1].set_title('Matched Filter Output')
axes[0, 1].set_xlabel('Time')
axes[0, 1].set_ylabel('Output')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 频域对比
freq = np.fft.fftfreq(N)
S_fft = np.abs(np.fft.fft(s))
Y_fft = np.abs(np.fft.fft(y))

axes[1, 0].plot(freq[:N//2], S_fft[:N//2], 'g-', linewidth=2, label='Signal')
axes[1, 0].plot(freq[:N//2], Y_fft[:N//2], 'b-', linewidth=1, alpha=0.7, label='Observation')
axes[1, 0].set_title('Frequency-Domain Comparison')
axes[1, 0].set_xlabel('Frequency')
axes[1, 0].set_ylabel('Amplitude')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# ROC曲线
axes[1, 1].plot(roc_pfa_list, roc_pd_list, 'b-', linewidth=2, label='Correlation Detector ROC')
axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='Random Guess')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_xlabel('False Alarm Rate (Pfa)')
axes[1, 1].set_ylabel('Detection Probability (Pd)')
axes[1, 1].set_title(f'ROC Curve at {roc_snr_db:.0f} dB SNR')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_signal_detection.png', dpi=150, bbox_inches='tight')
print("Figure saved to assets/ch01_signal_detection.png")

# ============================================================================
# 7. 检测器性能对比
# ============================================================================
print("\n7. 检测器性能对比")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pd vs SNR
axes[0].plot(SNR_range, Pd_list, 'b-o', linewidth=2, markersize=6, label='Correlation Detector')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('Detection Probability (Pd)')
axes[0].set_title('Detection Probability vs SNR')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Pfa vs SNR
axes[1].plot(SNR_range, Pfa_list, 'r-s', linewidth=2, markersize=6, label='False Alarm Rate')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('False Alarm Rate (Pfa)')
axes[1].set_title('False Alarm Rate vs SNR')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('assets/ch01_detection_performance.png', dpi=150, bbox_inches='tight')
print("Figure saved to assets/ch01_detection_performance.png")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
