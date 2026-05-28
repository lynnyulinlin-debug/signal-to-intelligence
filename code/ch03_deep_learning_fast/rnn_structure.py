"""
实验3.2：RNN结构与序列处理
对应章节：第3章 - 深度学习快速通道
目标：展示RNN如何处理序列数据，隐状态的演化过程
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
np.random.seed(42)
SEQUENCE_LENGTH = 50
HIDDEN_SIZE = 8
INPUT_SIZE = 3
OUTPUT_SIZE = 1

# ============ 核心逻辑 ============
# 生成合成序列数据
# 任务：给定前N个值，预测下一个值
X = np.random.randn(SEQUENCE_LENGTH, INPUT_SIZE)

# 生成目标序列：简单的线性组合 + 非线性变换
w_true = np.random.randn(INPUT_SIZE)
y = np.array([np.tanh(X[t] @ w_true) for t in range(SEQUENCE_LENGTH)])

# 简单RNN实现
# h_t = tanh(W_h * h_{t-1} + W_x * x_t + b_h)
# y_t = W_y * h_t + b_y

# 初始化权重
W_h = np.random.randn(HIDDEN_SIZE, HIDDEN_SIZE) * 0.1  # 隐状态到隐状态
W_x = np.random.randn(INPUT_SIZE, HIDDEN_SIZE) * 0.1   # 输入到隐状态
W_y = np.random.randn(HIDDEN_SIZE, OUTPUT_SIZE) * 0.1  # 隐状态到输出
b_h = np.zeros((1, HIDDEN_SIZE))
b_y = np.zeros((1, OUTPUT_SIZE))

# 前向传播：记录所有隐状态
h_states = []
h = np.zeros((1, HIDDEN_SIZE))  # 初始隐状态

for t in range(SEQUENCE_LENGTH):
    x_t = X[t:t+1]  # (1, INPUT_SIZE)
    h = np.tanh(x_t @ W_x + h @ W_h + b_h)  # (1, HIDDEN_SIZE)
    h_states.append(h.copy())

h_states = np.array(h_states).squeeze()  # (SEQUENCE_LENGTH, HIDDEN_SIZE)

# 计算输出
y_pred = h_states @ W_y + b_y  # (SEQUENCE_LENGTH, OUTPUT_SIZE)
y_pred = y_pred.flatten()

# 计算性能指标
mse = np.mean((y - y_pred) ** 2)
correlation = np.corrcoef(y, y_pred)[0, 1]

# 分析隐状态的演化
# 计算隐状态的范数（大小）
h_norms = np.linalg.norm(h_states, axis=1)

# 计算相邻隐状态之间的相似度
h_similarities = []
for t in range(1, SEQUENCE_LENGTH):
    sim = np.dot(h_states[t], h_states[t-1]) / (
        np.linalg.norm(h_states[t]) * np.linalg.norm(h_states[t-1]) + 1e-8
    )
    h_similarities.append(sim)

# 计算隐状态的主成分（用于可视化）
# 简单的2D投影：使用前两个隐单元
h_proj = h_states[:, :2]

# ============ 结果输出 ============
print("=" * 70)
print("RNN Structure and Sequence Processing")
print("=" * 70)
print(f"序列长度: {SEQUENCE_LENGTH}")
print(f"输入维度: {INPUT_SIZE}")
print(f"隐层大小: {HIDDEN_SIZE}")
print(f"输出维度: {OUTPUT_SIZE}")
print()

print("网络结构:")
print("-" * 70)
print(f"W_h (隐->隐): {W_h.shape}")
print(f"W_x (输入->隐): {W_x.shape}")
print(f"W_y (隐->输出): {W_y.shape}")
print()

print("性能指标:")
print("-" * 70)
print(f"预测MSE: {mse:.6f}")
print(f"预测相关系数: {correlation:.4f}")
print()

print("隐状态分析:")
print("-" * 70)
print(f"隐状态范数 - 最小: {h_norms.min():.4f}, 最大: {h_norms.max():.4f}, 平均: {h_norms.mean():.4f}")
print(f"隐状态相似度 - 最小: {min(h_similarities):.4f}, 最大: {max(h_similarities):.4f}, 平均: {np.mean(h_similarities):.4f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

# 1. 序列预测对比
ax = fig.add_subplot(gs[0, :2])
ax.plot(y, 'k-', linewidth=2, label='True Sequence', alpha=0.7)
ax.plot(y_pred, 'r--', linewidth=1.5, label='RNN Prediction', alpha=0.8)
ax.fill_between(range(SEQUENCE_LENGTH), y, y_pred, alpha=0.2, color='red')
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.set_title('Sequence Prediction')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 预测误差
ax = fig.add_subplot(gs[0, 2])
error = y - y_pred
ax.bar(range(SEQUENCE_LENGTH), error, color='steelblue', alpha=0.7)
ax.axhline(0, color='k', linestyle='-', linewidth=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Error')
ax.set_title('Prediction Error')
ax.grid(True, alpha=0.3, axis='y')

# 3. 隐状态热力图
ax = fig.add_subplot(gs[1, :2])
im = ax.imshow(h_states.T, aspect='auto', cmap='RdBu_r', interpolation='nearest')
ax.set_xlabel('Time Step')
ax.set_ylabel('Hidden Unit')
ax.set_title('Hidden State Evolution (Heatmap)')
plt.colorbar(im, ax=ax, label='Activation')

# 4. 隐状态范数
ax = fig.add_subplot(gs[1, 2])
ax.plot(h_norms, 'b-', linewidth=2, marker='o', markersize=4)
ax.fill_between(range(SEQUENCE_LENGTH), h_norms, alpha=0.3)
ax.set_xlabel('Time Step')
ax.set_ylabel('Hidden State Norm')
ax.set_title('Hidden State Magnitude')
ax.grid(True, alpha=0.3)

# 5. 隐状态相似度
ax = fig.add_subplot(gs[2, 0])
ax.plot(h_similarities, 'g-', linewidth=2, marker='s', markersize=4)
ax.axhline(0, color='k', linestyle='--', linewidth=0.5)
ax.set_xlabel('Time Step')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Adjacent Hidden State Similarity')
ax.grid(True, alpha=0.3)

# 6. 隐状态2D投影
ax = fig.add_subplot(gs[2, 1])
scatter = ax.scatter(h_proj[:, 0], h_proj[:, 1], c=range(SEQUENCE_LENGTH),
                     cmap='viridis', s=100, alpha=0.7, edgecolors='black', linewidth=0.5)
# 添加箭头显示时间流向
for t in range(0, SEQUENCE_LENGTH-1, 5):
    ax.arrow(h_proj[t, 0], h_proj[t, 1],
             h_proj[t+1, 0] - h_proj[t, 0],
             h_proj[t+1, 1] - h_proj[t, 1],
             head_width=0.1, head_length=0.1, fc='gray', ec='gray', alpha=0.5)
ax.set_xlabel('Hidden Unit 1')
ax.set_ylabel('Hidden Unit 2')
ax.set_title('Hidden State Trajectory (2D Projection)')
cbar = plt.colorbar(scatter, ax=ax, label='Time Step')
ax.grid(True, alpha=0.3)

# 7. 隐单元激活分布
ax = fig.add_subplot(gs[2, 2])
h_flat = h_states.flatten()
ax.hist(h_flat, bins=30, color='purple', alpha=0.7, edgecolor='black')
ax.set_xlabel('Activation Value')
ax.set_ylabel('Frequency')
ax.set_title('Hidden Unit Activation Distribution')
ax.grid(True, alpha=0.3, axis='y')

plt.savefig('assets/ch03_rnn_structure.png', dpi=100, bbox_inches='tight')
print("Figure saved to: assets/ch03_rnn_structure.png")
