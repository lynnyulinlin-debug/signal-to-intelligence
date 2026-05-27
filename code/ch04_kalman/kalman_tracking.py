"""
实验4.1：卡尔曼滤波跟踪
对应章节：第4章 - 卡尔曼滤波与状态空间
目标：用卡尔曼滤波跟踪匀速运动目标
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
TIME_STEPS = 100
DT = 0.1  # 时间步长
PROCESS_NOISE = 0.01  # 过程噪声方差
MEASUREMENT_NOISE = 1.0  # 测量噪声方差

# ============ 核心逻辑 ============
# 状态向量：[位置, 速度]
# 状态转移矩阵：x_{k+1} = F * x_k + w_k
F = np.array([[1, DT],
              [0, 1]])

# 测量矩阵：z_k = H * x_k + v_k（只能测量位置）
H = np.array([[1, 0]])

# 过程噪声协方差
Q = PROCESS_NOISE * np.eye(2)

# 测量噪声协方差
R = np.array([[MEASUREMENT_NOISE]])

# 初始状态和协方差
x = np.array([[0], [1]])  # 初始位置=0，初始速度=1
P = np.eye(2)

# 生成真实轨迹和测量值
true_positions = []
measurements = []
x_true = np.array([[0], [1]])

for t in range(TIME_STEPS):
    # 真实状态演化
    x_true = F @ x_true + np.random.randn(2, 1) * np.sqrt(PROCESS_NOISE)
    true_positions.append(x_true[0, 0])

    # 生成测量值
    z = H @ x_true + np.random.randn(1, 1) * np.sqrt(MEASUREMENT_NOISE)
    measurements.append(z[0, 0])

# 卡尔曼滤波
x = np.array([[0], [1]])
P = np.eye(2)
estimates = []
covariances = []

for t in range(TIME_STEPS):
    # 预测步骤
    x_pred = F @ x
    P_pred = F @ P @ F.T + Q

    # 更新步骤
    z = np.array([[measurements[t]]])
    y = z - H @ x_pred  # 创新（观测残差）
    S = H @ P_pred @ H.T + R  # 创新协方差
    K = P_pred @ H.T @ np.linalg.inv(S)  # 卡尔曼增益

    x = x_pred + K @ y
    P = (np.eye(2) - K @ H) @ P_pred

    estimates.append(x[0, 0])
    covariances.append(P[0, 0])

# ============ 结果输出 ============
print("=" * 70)
print("卡尔曼滤波跟踪")
print("=" * 70)
print(f"时间步数: {TIME_STEPS}")
print(f"时间步长: {DT} s")
print(f"过程噪声方差: {PROCESS_NOISE}")
print(f"测量噪声方差: {MEASUREMENT_NOISE}")
print()

# 计算性能指标
mse_measurement = np.mean((np.array(measurements) - np.array(true_positions)) ** 2)
mse_kalman = np.mean((np.array(estimates) - np.array(true_positions)) ** 2)

print("性能对比:")
print("-" * 70)
print(f"测量值 MSE: {mse_measurement:.6f}")
print(f"卡尔曼滤波 MSE: {mse_kalman:.6f}")
print(f"改进比例: {(1 - mse_kalman / mse_measurement) * 100:.2f}%")
print()

print("最后5个时间步的状态:")
print("-" * 70)
for t in range(-5, 0):
    print(f"t={TIME_STEPS+t}: 真实={true_positions[t]:.4f}, "
          f"测量={measurements[t]:.4f}, 估计={estimates[t]:.4f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 位置跟踪
ax = axes[0, 0]
t_array = np.arange(TIME_STEPS) * DT
ax.plot(t_array, true_positions, 'k-', linewidth=2, label='True Position')
ax.plot(t_array, measurements, 'r.', markersize=4, alpha=0.6, label='Measurements')
ax.plot(t_array, estimates, 'b-', linewidth=1.5, label='Kalman Estimate')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Position')
ax.set_title('Position Tracking')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 跟踪误差
ax = axes[0, 1]
error_measurement = np.array(measurements) - np.array(true_positions)
error_kalman = np.array(estimates) - np.array(true_positions)
ax.plot(error_measurement, 'r-', linewidth=1, alpha=0.7, label='Measurement Error')
ax.plot(error_kalman, 'b-', linewidth=1, alpha=0.7, label='Kalman Error')
ax.axhline(y=0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Error')
ax.set_title('Tracking Error')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 估计不确定性（协方差）
ax = axes[1, 0]
ax.plot(covariances, 'g-', linewidth=2)
ax.fill_between(range(TIME_STEPS), 0, covariances, alpha=0.3, color='green')
ax.set_xlabel('Time Step')
ax.set_ylabel('Position Variance')
ax.set_title('Kalman Filter Uncertainty')
ax.grid(True, alpha=0.3)

# 4. 误差分布
ax = axes[1, 1]
ax.hist(error_measurement, bins=20, alpha=0.6, label='Measurement Error', color='red')
ax.hist(error_kalman, bins=20, alpha=0.6, label='Kalman Error', color='blue')
ax.set_xlabel('Error')
ax.set_ylabel('Frequency')
ax.set_title('Error Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/ch04_kalman_tracking.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch04_kalman_tracking.png")
