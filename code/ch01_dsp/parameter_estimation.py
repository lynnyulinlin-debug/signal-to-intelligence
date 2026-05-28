"""
1.7 信号估计 - 代码实验

本实验演示：
1. 最大似然估计（MLE）
2. 最小二乘估计（LSE）
3. 贝叶斯估计
4. Cramér-Rao界
5. 估计器性能对比
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
from scipy.optimize import minimize
from scipy import stats

# 设置随机种子
np.random.seed(42)

print("=" * 60)
print("1.7 信号估计 - 代码实验")
print("=" * 60)

# ============================================================================
# 1. 生成观测数据
# ============================================================================
print("\n1. 生成观测数据")
print("-" * 60)

# 参数设置
true_amplitude = 2.0  # 真实幅度
true_frequency = 0.1  # 真实频率
true_phase = np.pi / 4  # 真实相位
N = 100  # 观测长度
SNR_dB = 10  # 信噪比
SNR = 10 ** (SNR_dB / 10)

# 生成信号
t = np.arange(N)
s_true = true_amplitude * np.sin(2 * np.pi * true_frequency * t + true_phase)

# 添加噪声
sigma2 = np.sum(s_true**2) / N / SNR
sigma = np.sqrt(sigma2)
w = sigma * np.random.randn(N)
y = s_true + w

print(f"真实幅度: {true_amplitude:.4f}")
print(f"真实频率: {true_frequency:.4f}")
print(f"真实相位: {true_phase:.4f}")
print(f"噪声功率: {sigma2:.4f}")
print(f"SNR (dB): {SNR_dB:.2f}")

# ============================================================================
# 2. 最小二乘估计（LSE）
# ============================================================================
print("\n2. 最小二乘估计（LSE）")
print("-" * 60)

# 线性模型：y = A*sin(2πft + φ) ≈ A1*sin(2πft) + A2*cos(2πft)
# 设计矩阵
X = np.column_stack([
    np.sin(2 * np.pi * true_frequency * t),
    np.cos(2 * np.pi * true_frequency * t)
])

# LSE求解
theta_lse = np.linalg.lstsq(X, y, rcond=None)[0]
A1_lse, A2_lse = theta_lse
amplitude_lse = np.sqrt(A1_lse**2 + A2_lse**2)
phase_lse = np.arctan2(A2_lse, A1_lse)

print(f"LSE幅度估计: {amplitude_lse:.4f}")
print(f"LSE相位估计: {phase_lse:.4f}")
print(f"幅度误差: {abs(amplitude_lse - true_amplitude):.4f}")
print(f"相位误差: {abs(phase_lse - true_phase):.4f}")

# ============================================================================
# 3. 最大似然估计（MLE）
# ============================================================================
print("\n3. 最大似然估计（MLE）")
print("-" * 60)

def negative_log_likelihood(params, y, t):
    """负对数似然函数"""
    A, f, phi = params
    s = A * np.sin(2 * np.pi * f * t + phi)
    residual = y - s
    nll = 0.5 * np.sum(residual**2) / sigma2
    return nll

# MLE求解
initial_guess = [1.5, 0.1, 0.5]
result_mle = minimize(negative_log_likelihood, initial_guess,
                      args=(y, t), method='Nelder-Mead')
amplitude_mle, frequency_mle, phase_mle = result_mle.x

print(f"MLE幅度估计: {amplitude_mle:.4f}")
print(f"MLE频率估计: {frequency_mle:.4f}")
print(f"MLE相位估计: {phase_mle:.4f}")
print(f"幅度误差: {abs(amplitude_mle - true_amplitude):.4f}")
print(f"频率误差: {abs(frequency_mle - true_frequency):.4f}")
print(f"相位误差: {abs(phase_mle - true_phase):.4f}")

# ============================================================================
# 4. 贝叶斯估计
# ============================================================================
print("\n4. 贝叶斯估计")
print("-" * 60)

# 使用LSE结果作为贝叶斯估计的初值
# 假设先验为高斯分布
prior_mean_A = 2.0
prior_std_A = 0.5

# 后验均值（简化的贝叶斯估计）
posterior_var = 1 / (1/sigma2 + 1/(prior_std_A**2))
posterior_mean = posterior_var * (amplitude_lse/sigma2 + prior_mean_A/(prior_std_A**2))

print(f"先验均值: {prior_mean_A:.4f}")
print(f"先验标准差: {prior_std_A:.4f}")
print(f"后验均值: {posterior_mean:.4f}")
print(f"后验方差: {posterior_var:.4f}")

# ============================================================================
# 5. Cramér-Rao界
# ============================================================================
print("\n5. Cramér-Rao界（CRB）")
print("-" * 60)

# Fisher信息矩阵（对于幅度参数）
# I(A) = (1/σ²) * Σ (∂s/∂A)²
ds_dA = np.sin(2 * np.pi * true_frequency * t + true_phase)
fisher_info_A = np.sum(ds_dA**2) / sigma2

# CRB
crb_A = 1 / fisher_info_A

print(f"Fisher信息矩阵（幅度）: {fisher_info_A:.4f}")
print(f"Cramér-Rao界（幅度）: {crb_A:.4f}")
print(f"LSE方差: {np.var([amplitude_lse]):.4f}")
print(f"LSE是否达到CRB: {np.var([amplitude_lse]) >= crb_A}")

# ============================================================================
# 6. 性能对比
# ============================================================================
print("\n6. 估计器性能对比")
print("-" * 60)

# 多次实验
num_trials = 100
amplitude_estimates_lse = []
amplitude_estimates_mle = []
amplitude_estimates_bayes = []

for trial in range(num_trials):
    # 生成新的观测数据
    w_trial = sigma * np.random.randn(N)
    y_trial = s_true + w_trial

    # LSE
    theta_lse_trial = np.linalg.lstsq(X, y_trial, rcond=None)[0]
    A1, A2 = theta_lse_trial
    amplitude_estimates_lse.append(np.sqrt(A1**2 + A2**2))

    # MLE
    result_mle_trial = minimize(negative_log_likelihood, initial_guess,
                                args=(y_trial, t), method='Nelder-Mead')
    amplitude_estimates_mle.append(result_mle_trial.x[0])

    # Bayes
    posterior_mean_trial = posterior_var * (amplitude_estimates_lse[-1]/sigma2 +
                                            prior_mean_A/(prior_std_A**2))
    amplitude_estimates_bayes.append(posterior_mean_trial)

amplitude_estimates_lse = np.array(amplitude_estimates_lse)
amplitude_estimates_mle = np.array(amplitude_estimates_mle)
amplitude_estimates_bayes = np.array(amplitude_estimates_bayes)

# 计算性能指标
mse_lse = np.mean((amplitude_estimates_lse - true_amplitude)**2)
mse_mle = np.mean((amplitude_estimates_mle - true_amplitude)**2)
mse_bayes = np.mean((amplitude_estimates_bayes - true_amplitude)**2)

bias_lse = np.mean(amplitude_estimates_lse - true_amplitude)
bias_mle = np.mean(amplitude_estimates_mle - true_amplitude)
bias_bayes = np.mean(amplitude_estimates_bayes - true_amplitude)

print(f"LSE - MSE: {mse_lse:.6f}, 偏差: {bias_lse:.6f}")
print(f"MLE - MSE: {mse_mle:.6f}, 偏差: {bias_mle:.6f}")
print(f"Bayes - MSE: {mse_bayes:.6f}, 偏差: {bias_bayes:.6f}")
print(f"CRB: {crb_A:.6f}")

# ============================================================================
# 7. 可视化
# ============================================================================
print("\n7. 生成可视化图表")
print("-" * 60)

fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 观测信号
axes[0, 0].plot(t, s_true, 'g-', linewidth=2, label='True Signal')
axes[0, 0].plot(t, y, 'b-', linewidth=0.5, alpha=0.7, label='Observation (Signal + Noise)')
axes[0, 0].set_title('Observed Signal')
axes[0, 0].set_xlabel('Time')
axes[0, 0].set_ylabel('Amplitude')
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

# 估计值对比
estimators = ['LSE', 'MLE', 'Bayes']
estimates = [amplitude_lse, amplitude_mle, posterior_mean]
colors = ['b', 'r', 'g']

axes[0, 1].bar(estimators, estimates, color=colors, alpha=0.7)
axes[0, 1].axhline(y=true_amplitude, color='k', linestyle='--', linewidth=2, label='True Value')
axes[0, 1].set_ylabel('Amplitude Estimate')
axes[0, 1].set_title('Single Trial: Amplitude Estimates')
axes[0, 1].legend()
axes[0, 1].grid(True, alpha=0.3, axis='y')

# 多次实验的估计分布
axes[1, 0].hist(amplitude_estimates_lse, bins=20, alpha=0.5, label='LSE', color='b')
axes[1, 0].hist(amplitude_estimates_mle, bins=20, alpha=0.5, label='MLE', color='r')
axes[1, 0].hist(amplitude_estimates_bayes, bins=20, alpha=0.5, label='Bayes', color='g')
axes[1, 0].axvline(x=true_amplitude, color='k', linestyle='--', linewidth=2, label='True Value')
axes[1, 0].set_xlabel('Amplitude Estimate')
axes[1, 0].set_ylabel('Count')
axes[1, 0].set_title(f'Estimation Distribution Across {num_trials} Trials')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3, axis='y')

# MSE对比
mse_values = [mse_lse, mse_mle, mse_bayes, crb_A]
labels = ['LSE', 'MLE', 'Bayes', 'CRB']
colors_mse = ['b', 'r', 'g', 'k']

axes[1, 1].bar(labels, mse_values, color=colors_mse, alpha=0.7)
axes[1, 1].set_ylabel('MSE')
axes[1, 1].set_title('Estimator Performance (MSE)')
axes[1, 1].grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/ch01_parameter_estimation.png', dpi=150, bbox_inches='tight')
print("Figure saved to assets/ch01_parameter_estimation.png")

print("\n" + "=" * 60)
print("实验完成！")
print("=" * 60)
