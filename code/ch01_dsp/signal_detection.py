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

N = 100  # 信号长度
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

# 生成多个SNR下的检测性能
SNR_range = np.linspace(-5, 15, 20)
Pd_list = []
Pfa_list = []

for snr_db in SNR_range:
    snr = 10 ** (snr_db / 10)
    sigma_temp = np.sqrt(np.sum(s**2) / N / snr)

    # 生成多个样本
    num_trials = 100
    detections_h1 = 0
    detections_h0 = 0

    for _ in range(num_trials):
        # H1：信号+噪声
        w_h1 = sigma_temp * np.random.randn(N)
        y_h1 = s + w_h1
        T_h1 = np.sum(y_h1 * s)
        if T_h1 > threshold_corr:
            detections_h1 += 1

        # H0：仅噪声
        w_h0 = sigma_temp * np.random.randn(N)
        T_h0 = np.sum(w_h0 * s)
        if T_h0 > threshold_corr:
            detections_h0 += 1

    Pd = detections_h1 / num_trials
    Pfa = detections_h0 / num_trials
    Pd_list.append(Pd)
    Pfa_list.append(Pfa)

print(f"SNR范围: {SNR_range[0]:.1f} - {SNR_range[-1]:.1f} dB")
print(f"最低SNR下的Pd: {Pd_list[0]:.4f}")
print(f"最高SNR下的Pd: {Pd_list[-1]:.4f}")
print(f"平均虚警率: {np.mean(Pfa_list):.4f}")

# ============================================================================
# 6. 可视化
# ============================================================================
print("\n6. 生成可视化图表")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 时域信号
axes[0, 0].plot(t, s, 'g-', linewidth=2, label='信号')
axes[0, 0].plot(t, y, 'b-', linewidth=0.5, alpha=0.7, label='观测（信号+噪声）')
axes[0, 0].plot(t, w, 'r-', linewidth=0.5, alpha=0.5, label='噪声')
axes[0, 0].set_title('时域信号')
axes[0, 0].set_xlabel('时间')
axes[0, 0].set_ylabel('幅度')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 匹配滤波器输出
axes[0, 1].plot(t, mf_output, 'b-', linewidth=1)
axes[0, 1].axhline(y=threshold_corr, color='r', linestyle='--', linewidth=2, label='阈值')
axes[0, 1].plot(mf_max_idx, mf_max, 'ro', markersize=8, label='最大值')
axes[0, 1].set_title('匹配滤波器输出')
axes[0, 1].set_xlabel('时间')
axes[0, 1].set_ylabel('输出')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3)

# 频域对比
freq = np.fft.fftfreq(N)
S_fft = np.abs(np.fft.fft(s))
Y_fft = np.abs(np.fft.fft(y))

axes[1, 0].plot(freq[:N//2], S_fft[:N//2], 'g-', linewidth=2, label='信号')
axes[1, 0].plot(freq[:N//2], Y_fft[:N//2], 'b-', linewidth=1, alpha=0.7, label='观测')
axes[1, 0].set_title('频域对比')
axes[1, 0].set_xlabel('频率')
axes[1, 0].set_ylabel('幅度')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# ROC曲线
axes[1, 1].plot(Pfa_list, Pd_list, 'b-o', linewidth=2, markersize=4, label='检测器')
axes[1, 1].plot([0, 1], [0, 1], 'r--', linewidth=1, label='随机检测')
axes[1, 1].set_xlim([0, 1])
axes[1, 1].set_ylim([0, 1])
axes[1, 1].set_xlabel('虚警率 (Pfa)')
axes[1, 1].set_ylabel('检测概率 (Pd)')
axes[1, 1].set_title('ROC曲线')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_signal_detection.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存到 assets/ch01_signal_detection.png")

# ============================================================================
# 7. 检测器性能对比
# ============================================================================
print("\n7. 检测器性能对比")
print("-" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Pd vs SNR
axes[0].plot(SNR_range, Pd_list, 'b-o', linewidth=2, markersize=6, label='相关检测器')
axes[0].set_xlabel('SNR (dB)')
axes[0].set_ylabel('检测概率 (Pd)')
axes[0].set_title('检测概率 vs SNR')
axes[0].grid(True, alpha=0.3)
axes[0].legend()

# Pfa vs SNR
axes[1].plot(SNR_range, Pfa_list, 'r-s', linewidth=2, markersize=6, label='虚警率')
axes[1].set_xlabel('SNR (dB)')
axes[1].set_ylabel('虚警率 (Pfa)')
axes[1].set_title('虚警率 vs SNR')
axes[1].grid(True, alpha=0.3)
axes[1].legend()

plt.tight_layout()
plt.savefig('assets/ch01_detection_performance.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存到 assets/ch01_detection_performance.png")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
