"""
实验4.2：RNN结构观察
对应章节：第4章 - 卡尔曼滤波与状态空间
目标：用极小型RNN观察更新结构，对比卡尔曼滤波
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
SEQ_LEN = 50
INPUT_DIM = 1
HIDDEN_DIM = 2
OUTPUT_DIM = 1
LEARNING_RATE = 0.1
EPOCHS = 50

# ============ 核心逻辑 ============
# 生成简单的序列数据：y_t = sin(t) + noise
t_array = np.arange(SEQ_LEN)
y_true = np.sin(2 * np.pi * t_array / SEQ_LEN)
y_noisy = y_true + 0.1 * np.random.randn(SEQ_LEN)

# 初始化RNN参数
# h_t = tanh(W_h @ h_{t-1} + W_x @ x_t + b_h)
# y_t = W_y @ h_t + b_y
W_h = np.random.randn(HIDDEN_DIM, HIDDEN_DIM) * 0.1
W_x = np.random.randn(HIDDEN_DIM, INPUT_DIM) * 0.1
W_y = np.random.randn(OUTPUT_DIM, HIDDEN_DIM) * 0.1
b_h = np.zeros((HIDDEN_DIM, 1))
b_y = np.zeros((OUTPUT_DIM, 1))

# 训练RNN
losses = []
h_states_history = []

for epoch in range(EPOCHS):
    h = np.zeros((HIDDEN_DIM, 1))
    loss = 0
    h_states = [h.copy()]

    # 前向传播
    for step in range(SEQ_LEN):
        x_t = np.array([[y_noisy[step]]])

        # RNN隐状态更新
        h = np.tanh(W_h @ h + W_x @ x_t + b_h)
        h_states.append(h.copy())

        # 输出
        y_pred = W_y @ h + b_y

        # 损失
        error = y_pred - np.array([[y_true[step]]])
        loss += np.mean(error ** 2)

    losses.append(loss / SEQ_LEN)
    h_states_history.append(h_states)

    # 简化的反向传播（只更新输出层）
    h = np.zeros((HIDDEN_DIM, 1))
    for step in range(SEQ_LEN):
        x_t = np.array([[y_noisy[step]]])
        h = np.tanh(W_h @ h + W_x @ x_t + b_h)
        y_pred = W_y @ h + b_y
        error = y_pred - np.array([[y_true[step]]])

        # 梯度
        dW_y = error @ h.T
        db_y = error

        # 更新
        W_y -= LEARNING_RATE * dW_y
        b_y -= LEARNING_RATE * db_y

# 最终预测
h = np.zeros((HIDDEN_DIM, 1))
y_pred_rnn = []
h_final_states = []

for step in range(SEQ_LEN):
    x_t = np.array([[y_noisy[step]]])
    h = np.tanh(W_h @ h + W_x @ x_t + b_h)
    h_final_states.append(h.copy())
    y_pred = W_y @ h + b_y
    y_pred_rnn.append(y_pred[0, 0])

y_pred_rnn = np.array(y_pred_rnn)

# ============ 结果输出 ============
print("=" * 70)
print("RNN 结构观察")
print("=" * 70)
print(f"序列长度: {SEQ_LEN}")
print(f"隐层维度: {HIDDEN_DIM}")
print(f"训练轮数: {EPOCHS}")
print()

print("RNN 参数形状:")
print("-" * 70)
print(f"W_h (隐→隐): {W_h.shape}")
print(f"W_x (输入→隐): {W_x.shape}")
print(f"W_y (隐→输出): {W_y.shape}")
print()

print("RNN 参数值:")
print("-" * 70)
print(f"W_h:\n{W_h.round(4)}")
print(f"W_x:\n{W_x.round(4)}")
print(f"W_y:\n{W_y.round(4)}")
print()

# 计算性能
mse_rnn = np.mean((y_pred_rnn - y_true) ** 2)
mse_noisy = np.mean((y_noisy - y_true) ** 2)

print("性能对比:")
print("-" * 70)
print(f"原始观测 MSE: {mse_noisy:.6f}")
print(f"RNN 预测 MSE: {mse_rnn:.6f}")
print(f"改进比例: {(1 - mse_rnn / mse_noisy) * 100:.2f}%")
print()

# 隐状态分析
h_final_array = np.array([h.flatten() for h in h_final_states])
print("隐状态统计:")
print("-" * 70)
print(f"隐状态范围: [{h_final_array.min():.4f}, {h_final_array.max():.4f}]")
print(f"隐状态均值: {h_final_array.mean():.4f}")
print(f"隐状态方差: {h_final_array.var():.4f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 序列预测
ax = axes[0, 0]
ax.plot(t_array, y_true, 'k-', linewidth=2, label='True Signal')
ax.plot(t_array, y_noisy, 'r.', markersize=4, alpha=0.6, label='Noisy Input')
ax.plot(t_array, y_pred_rnn, 'b-', linewidth=1.5, label='RNN Prediction')
ax.set_xlabel('Time Step')
ax.set_ylabel('Value')
ax.set_title('RNN Sequence Prediction')
ax.legend()
ax.grid(True, alpha=0.3)

# 2. 训练损失
ax = axes[0, 1]
ax.plot(losses, 'g-', linewidth=2)
ax.set_xlabel('Epoch')
ax.set_ylabel('MSE Loss')
ax.set_title('Training Loss')
ax.grid(True, alpha=0.3)

# 3. 隐状态轨迹
ax = axes[1, 0]
h_array = np.array([h.flatten() for h in h_final_states])
ax.plot(h_array[:, 0], 'b-', linewidth=1.5, label='h[0]', alpha=0.8)
ax.plot(h_array[:, 1], 'r-', linewidth=1.5, label='h[1]', alpha=0.8)
ax.set_xlabel('Time Step')
ax.set_ylabel('Hidden State Value')
ax.set_title('Hidden State Dynamics')
ax.legend()
ax.grid(True, alpha=0.3)

# 4. 隐状态相空间
ax = axes[1, 1]
ax.plot(h_array[:, 0], h_array[:, 1], 'b-', linewidth=1.5, alpha=0.7)
ax.scatter(h_array[0, 0], h_array[0, 1], color='g', s=100, marker='o', label='Start', zorder=5)
ax.scatter(h_array[-1, 0], h_array[-1, 1], color='r', s=100, marker='s', label='End', zorder=5)
ax.set_xlabel('h[0]')
ax.set_ylabel('h[1]')
ax.set_title('Hidden State Phase Space')
ax.legend()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch04_rnn_structure.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch04_rnn_structure.png")
