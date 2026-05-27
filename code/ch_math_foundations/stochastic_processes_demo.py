"""
随机过程进阶 - 代码实验

本实验演示：
1. 马尔可夫链的模拟和分析
2. AR(自回归)模型的生成和拟合
3. Kalman 滤波的应用
4. 平稳过程的性质
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal, stats

# 设置随机种子
np.random.seed(42)

print("=" * 70)
print("随机过程进阶 - 代码实验")
print("=" * 70)

# ============================================================================
# 1. 马尔可夫链
# ============================================================================
print("\n1. 马尔可夫链")
print("-" * 70)

# 定义转移矩阵（天气模型）
# 状态：0=晴天，1=阴天，2=下雨
transition_matrix = np.array([
    [0.7, 0.2, 0.1],  # 晴天 → 晴天、阴天、下雨
    [0.3, 0.4, 0.3],  # 阴天 → 晴天、阴天、下雨
    [0.2, 0.3, 0.5]   # 下雨 → 晴天、阴天、下雨
])

state_names = ['晴天', '阴天', '下雨']

print("转移矩阵（天气模型）:")
print("       晴天  阴天  下雨")
for i, state in enumerate(state_names):
    print(f"{state:3s}  {transition_matrix[i]}")

# 模拟马尔可夫链
def simulate_markov_chain(P, initial_state, n_steps):
    """模拟马尔可夫链"""
    states = [initial_state]
    current_state = initial_state

    for _ in range(n_steps):
        # 根据转移概率选择下一个状态
        next_state = np.random.choice(len(P), p=P[current_state])
        states.append(next_state)
        current_state = next_state

    return np.array(states)

# 从不同初始状态开始模拟
n_steps = 100
initial_states = [0, 1, 2]  # 晴天、阴天、下雨

print(f"\n模拟 {n_steps} 步的马尔可夫链:")
simulations = []
for initial_state in initial_states:
    states = simulate_markov_chain(transition_matrix, initial_state, n_steps)
    simulations.append(states)

    # 统计状态分布
    unique, counts = np.unique(states, return_counts=True)
    print(f"\n初始状态：{state_names[initial_state]}")
    for state_idx, count in zip(unique, counts):
        print(f"  {state_names[state_idx]}: {count/len(states)*100:.1f}%")

# 计算平稳分布
print("\n计算平稳分布:")
# 求解 π = π * P，即 π * (P - I) = 0
eigenvalues, eigenvectors = np.linalg.eig(transition_matrix.T)
stationary_idx = np.argmax(np.abs(eigenvalues - 1) < 1e-10)
stationary_dist = np.real(eigenvectors[:, stationary_idx])
stationary_dist = stationary_dist / np.sum(stationary_dist)

print("平稳分布:")
for state, prob in zip(state_names, stationary_dist):
    print(f"  {state}: {prob:.4f}")

# ============================================================================
# 2. AR(自回归)模型
# ============================================================================
print("\n2. AR(自回归)模型")
print("-" * 70)

# AR(1) 模型：X_t = φ * X_{t-1} + ε_t
def generate_ar_process(phi, n_samples, sigma=1.0):
    """生成 AR(1) 过程"""
    x = np.zeros(n_samples)
    x[0] = np.random.randn() * sigma

    for t in range(1, n_samples):
        x[t] = phi * x[t-1] + np.random.randn() * sigma

    return x

# 生成不同参数的 AR 过程
phi_values = [0.3, 0.7, 0.95, -0.7]
n_samples = 500

print("生成 AR(1) 过程（不同参数）:")
ar_processes = []
for phi in phi_values:
    x = generate_ar_process(phi, n_samples)
    ar_processes.append(x)

    # 计算统计特性
    mean = np.mean(x)
    var = np.var(x)
    acf_lag1 = np.corrcoef(x[:-1], x[1:])[0, 1]

    print(f"\nφ = {phi:6.2f}:")
    print(f"  均值: {mean:7.4f}")
    print(f"  方差: {var:7.4f}")
    print(f"  ACF(1): {acf_lag1:7.4f} (理论值: {phi:.4f})")

# ============================================================================
# 3. 自相关函数 (ACF)
# ============================================================================
print("\n3. 自相关函数 (ACF)")
print("-" * 70)

def compute_acf(x, max_lag=20):
    """计算自相关函数"""
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)
    acf = [1.0]

    for lag in range(1, max_lag + 1):
        c_lag = np.dot(x[:-lag], x[lag:]) / len(x)
        acf.append(c_lag / c0)

    return np.array(acf)

print("AR(1) 过程的 ACF:")
for phi, x in zip(phi_values, ar_processes):
    acf = compute_acf(x, max_lag=10)
    print(f"\nφ = {phi:6.2f}:")
    print(f"  ACF: {acf[:6]}")

# ============================================================================
# 4. Kalman 滤波
# ============================================================================
print("\n4. Kalman 滤波")
print("-" * 70)

# 状态空间模型
# x_{t+1} = A * x_t + w_t  (状态方程)
# y_t = C * x_t + v_t      (观测方程)

# 简单的一维模型
A = np.array([[0.9]])  # 状态转移
C = np.array([[1.0]])  # 观测矩阵
Q = np.array([[0.1]])  # 过程噪声方差
R = np.array([[1.0]])  # 观测噪声方差

# 生成真实状态和观测
n_time = 100
true_states = np.zeros(n_time)
observations = np.zeros(n_time)

true_states[0] = np.random.randn()
for t in range(1, n_time):
    true_states[t] = A[0, 0] * true_states[t-1] + np.random.randn() * np.sqrt(Q[0, 0])
    observations[t] = C[0, 0] * true_states[t] + np.random.randn() * np.sqrt(R[0, 0])

print("状态空间模型:")
print(f"  状态转移系数 A: {A[0, 0]}")
print(f"  观测系数 C: {C[0, 0]}")
print(f"  过程噪声方差 Q: {Q[0, 0]}")
print(f"  观测噪声方差 R: {R[0, 0]}")

# Kalman 滤波
def kalman_filter(observations, A, C, Q, R, x0=0, P0=1):
    """Kalman 滤波"""
    n = len(observations)
    x_filtered = np.zeros(n)
    P_filtered = np.zeros(n)
    x_predicted = np.zeros(n)
    P_predicted = np.zeros(n)

    x = x0
    P = P0

    for t in range(n):
        # 预测
        x_pred = A @ x
        P_pred = A @ P @ A.T + Q

        # 更新
        y = observations[t] - C @ x_pred
        S = C @ P_pred @ C.T + R
        K = P_pred @ C.T / S
        x = x_pred + K * y
        P = (1 - K @ C) @ P_pred

        x_filtered[t] = x[0]
        P_filtered[t] = P[0, 0]
        x_predicted[t] = x_pred[0]
        P_predicted[t] = P_pred[0, 0]

    return x_filtered, P_filtered, x_predicted, P_predicted

x_filtered, P_filtered, x_predicted, P_predicted = kalman_filter(
    observations, A, C, Q, R
)

print(f"\nKalman 滤波结果:")
print(f"  真实状态均值: {np.mean(true_states):.4f}")
print(f"  观测均值: {np.mean(observations):.4f}")
print(f"  滤波估计均值: {np.mean(x_filtered):.4f}")
print(f"  真实状态方差: {np.var(true_states):.4f}")
print(f"  观测方差: {np.var(observations):.4f}")
print(f"  滤波估计方差: {np.var(x_filtered):.4f}")

# ============================================================================
# 5. 可视化
# ============================================================================
print("\n5. 生成可视化图表")
print("-" * 70)

fig = plt.figure(figsize=(16, 12))

# 子图1：马尔可夫链模拟
ax = fig.add_subplot(3, 3, 1)
colors_mc = ['gold', 'gray', 'blue']
for i, (initial_state, states) in enumerate(zip(initial_states, simulations)):
    ax.plot(states[:50], 'o-', color=colors_mc[i], alpha=0.7,
            label=f'初始: {state_names[initial_state]}', markersize=4)
ax.set_xlabel('时间步')
ax.set_ylabel('状态')
ax.set_yticks([0, 1, 2])
ax.set_yticklabels(state_names)
ax.set_title('马尔可夫链模拟', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图2：马尔可夫链的状态分布收敛
ax = fig.add_subplot(3, 3, 2)
state_dist = np.zeros((n_steps, 3))
for t in range(n_steps):
    for i in range(3):
        state_dist[t, i] = np.sum(simulations[0][:t+1] == i) / (t + 1)

for i, state in enumerate(state_names):
    ax.plot(state_dist[:, i], label=state, linewidth=2)
ax.axhline(y=stationary_dist[0], color='gold', linestyle='--', alpha=0.5)
ax.axhline(y=stationary_dist[1], color='gray', linestyle='--', alpha=0.5)
ax.axhline(y=stationary_dist[2], color='blue', linestyle='--', alpha=0.5)
ax.set_xlabel('时间步')
ax.set_ylabel('状态概率')
ax.set_title('马尔可夫链：状态分布收敛', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图3：转移矩阵的热力图
ax = fig.add_subplot(3, 3, 3)
im = ax.imshow(transition_matrix, cmap='YlOrRd', aspect='auto')
ax.set_xticks([0, 1, 2])
ax.set_yticks([0, 1, 2])
ax.set_xticklabels(state_names)
ax.set_yticklabels(state_names)
ax.set_xlabel('下一个状态')
ax.set_ylabel('当前状态')
ax.set_title('转移矩阵', fontsize=12, fontweight='bold')
for i in range(3):
    for j in range(3):
        ax.text(j, i, f'{transition_matrix[i, j]:.2f}',
                ha='center', va='center', fontsize=10)
plt.colorbar(im, ax=ax)

# 子图4-6：AR 过程
for idx, (phi, x) in enumerate(zip(phi_values[:3], ar_processes[:3])):
    ax = fig.add_subplot(3, 3, 4 + idx)
    ax.plot(x[:100], linewidth=1, alpha=0.7)
    ax.set_xlabel('时间')
    ax.set_ylabel('值')
    ax.set_title(f'AR(1) 过程，φ={phi}', fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3)

# 子图7：ACF 对比
ax = fig.add_subplot(3, 3, 7)
for phi, x in zip(phi_values[:3], ar_processes[:3]):
    acf = compute_acf(x, max_lag=20)
    ax.plot(acf, marker='o', label=f'φ={phi}', markersize=4)
ax.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('滞后')
ax.set_ylabel('自相关')
ax.set_title('AR(1) 过程的 ACF', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图8：Kalman 滤波结果
ax = fig.add_subplot(3, 3, 8)
ax.plot(true_states, 'g-', linewidth=2, label='真实状态', alpha=0.7)
ax.plot(observations, 'b.', markersize=4, label='观测', alpha=0.5)
ax.plot(x_filtered, 'r-', linewidth=1.5, label='Kalman 滤波', alpha=0.8)
ax.set_xlabel('时间')
ax.set_ylabel('值')
ax.set_title('Kalman 滤波', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# 子图9：滤波误差
ax = fig.add_subplot(3, 3, 9)
error = true_states - x_filtered
ax.plot(error, linewidth=1, alpha=0.7)
ax.axhline(y=0, color='k', linestyle='--', linewidth=1)
ax.fill_between(range(len(error)), -2*np.sqrt(P_filtered), 2*np.sqrt(P_filtered),
                alpha=0.3, label='±2σ 置信区间')
ax.set_xlabel('时间')
ax.set_ylabel('误差')
ax.set_title('Kalman 滤波误差', fontsize=12, fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch_math_stochastic_processes.png', dpi=150, bbox_inches='tight')
print("✓ 图表已保存到 assets/ch_math_stochastic_processes.png")

# ============================================================================
# 6. 平稳性检验
# ============================================================================
print("\n6. 平稳性检验")
print("-" * 70)

# 生成平稳和非平稳过程
stationary_process = generate_ar_process(0.5, 500)
non_stationary_process = np.cumsum(np.random.randn(500))  # 随机游走

print("平稳过程 (AR(1), φ=0.5):")
print(f"  均值: {np.mean(stationary_process):.4f}")
print(f"  方差: {np.var(stationary_process):.4f}")
acf_stat = compute_acf(stationary_process, max_lag=5)
print(f"  ACF: {acf_stat}")

print("\n非平稳过程 (随机游走):")
print(f"  均值: {np.mean(non_stationary_process):.4f}")
print(f"  方差: {np.var(non_stationary_process):.4f}")
acf_non_stat = compute_acf(non_stationary_process, max_lag=5)
print(f"  ACF: {acf_non_stat}")

print("\n" + "=" * 70)
print("实验完成！")
print("=" * 70)
