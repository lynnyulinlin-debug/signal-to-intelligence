"""
实验3.5：序列模型与一维信号处理
对应章节：第3章 - 深度学习快速通道
目标：对比1D CNN、RNN和Transformer风格模型在时序信号上的建模方式
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)
T = 200

# 合成信号：低频趋势 + 高频振荡 + 局部脉冲
x = np.linspace(0, 4 * np.pi, T)
signal = 0.6 * np.sin(x) + 0.25 * np.sin(4 * x)
signal[80:90] += 0.8
signal[145:150] -= 0.6
signal += 0.05 * np.random.randn(T)

# 1D CNN风格：局部卷积核响应
kernel_local = np.array([-1.0, 0.0, 1.0])
kernel_smooth = np.array([0.25, 0.5, 0.25])
cnn_edge = np.convolve(signal, kernel_local, mode="same")
cnn_smooth = np.convolve(signal, kernel_smooth, mode="same")

# RNN风格：递推状态
alpha = 0.85
rnn_state = np.zeros(T)
for t in range(1, T):
    rnn_state[t] = alpha * rnn_state[t - 1] + (1 - alpha) * signal[t]

# Transformer风格：全局注意力示意（用全局均值+局部信号构造）
window = 25
attention_summary = np.zeros(T)
for t in range(T):
    left = max(0, t - window)
    right = min(T, t + window)
    local = signal[left:right]
    weights = np.exp(-np.abs(np.arange(left, right) - t) / 8)
    weights = weights / weights.sum()
    attention_summary[t] = np.sum(local * weights)

global_context = np.mean(signal)
transformer_like = 0.7 * attention_summary + 0.3 * global_context

print("=" * 70)
print("Sequence Models and 1D Signal Processing")
print("=" * 70)
print(f"信号长度: {T}")
print(f"信号均值: {signal.mean():.4f}")
print(f"信号标准差: {signal.std():.4f}")
print()
print("模型直觉对比:")
print("- 1D CNN: 更关注局部变化和模式")
print("- RNN: 通过递推状态累积历史信息")
print("- Transformer-like: 通过全局注意机制聚合远近信息")
print("=" * 70)

fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

axes[0].plot(signal, color="black", linewidth=1.8)
axes[0].set_title("Input 1D Signal")
axes[0].grid(True, alpha=0.3)

axes[1].plot(cnn_edge, label="Edge-like Kernel", color="tab:red")
axes[1].plot(cnn_smooth, label="Smoothing Kernel", color="tab:blue")
axes[1].set_title("1D CNN-style Local Responses")
axes[1].legend()
axes[1].grid(True, alpha=0.3)

axes[2].plot(rnn_state, color="tab:green")
axes[2].set_title("RNN-style Recursive State")
axes[2].grid(True, alpha=0.3)

axes[3].plot(transformer_like, color="tab:purple")
axes[3].set_title("Transformer-style Global Context Summary")
axes[3].grid(True, alpha=0.3)
axes[3].set_xlabel("Time Step")

plt.tight_layout()
plt.savefig("assets/ch03_sequence_models.png", dpi=120, bbox_inches="tight")
print("Figure saved to: assets/ch03_sequence_models.png")
