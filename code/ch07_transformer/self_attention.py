"""
实验7.1：自注意力
对应章节：第7章 - Transformer与自注意力
目标：用numpy实现简单的自注意力，在小序列上可视化权重
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
SEQ_LEN = 8
D_MODEL = 64
NUM_HEADS = 4
D_HEAD = D_MODEL // NUM_HEADS

# ============ 核心逻辑 ============
def scaled_dot_product_attention(Q, K, V, mask=None):
    """
    缩放点积注意力
    Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V
    """
    d_k = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(d_k)

    if mask is not None:
        scores = scores + mask

    attn_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
    attn_weights = attn_weights / np.sum(attn_weights, axis=-1, keepdims=True)

    output = attn_weights @ V
    return output, attn_weights

def multi_head_attention(X, W_q, W_k, W_v, W_o):
    """
    多头注意力
    """
    batch_size, seq_len, d_model = X.shape

    # 线性投射到Q, K, V
    Q = X @ W_q  # (batch_size, seq_len, d_model)
    K = X @ W_k
    V = X @ W_v

    # 分割成多个头
    Q = Q.reshape(batch_size, seq_len, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    K = K.reshape(batch_size, seq_len, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)
    V = V.reshape(batch_size, seq_len, NUM_HEADS, D_HEAD).transpose(0, 2, 1, 3)

    # 对每个头计算注意力
    attn_outputs = []
    attn_weights_list = []

    for h in range(NUM_HEADS):
        output, attn_weights = scaled_dot_product_attention(Q[0, h], K[0, h], V[0, h])
        attn_outputs.append(output)
        attn_weights_list.append(attn_weights)

    # 拼接多个头的输出
    attn_output = np.concatenate(attn_outputs, axis=-1)  # (seq_len, d_model)

    # 最终线性投射
    output = attn_output @ W_o

    return output, attn_weights_list

# 生成输入序列
X = np.random.randn(1, SEQ_LEN, D_MODEL) * 0.1

# 初始化权重
W_q = np.random.randn(D_MODEL, D_MODEL) * 0.01
W_k = np.random.randn(D_MODEL, D_MODEL) * 0.01
W_v = np.random.randn(D_MODEL, D_MODEL) * 0.01
W_o = np.random.randn(D_MODEL, D_MODEL) * 0.01

# 计算多头注意力
output, attn_weights_list = multi_head_attention(X, W_q, W_k, W_v, W_o)

# 计算平均注意力权重（用于可视化）
avg_attn_weights = np.mean(np.array(attn_weights_list), axis=0)

# ============ 结果输出 ============
print("=" * 70)
print("自注意力（Self-Attention）")
print("=" * 70)
print(f"序列长度: {SEQ_LEN}")
print(f"模型维度: {D_MODEL}")
print(f"注意力头数: {NUM_HEADS}")
print(f"每个头的维度: {D_HEAD}")
print()

print("输入形状:")
print("-" * 70)
print(f"X: {X.shape}")
print()

print("权重矩阵形状:")
print("-" * 70)
print(f"W_q: {W_q.shape}")
print(f"W_k: {W_k.shape}")
print(f"W_v: {W_v.shape}")
print(f"W_o: {W_o.shape}")
print()

print("输出形状:")
print("-" * 70)
print(f"output: {output.shape}")
print()

print("注意力权重统计:")
print("-" * 70)
for h in range(NUM_HEADS):
    weights = attn_weights_list[h]
    print(f"头 {h}: 最小={weights.min():.4f}, 最大={weights.max():.4f}, "
          f"均值={weights.mean():.4f}")
print()

print("平均注意力权重（所有头的平均）:")
print("-" * 70)
print(avg_attn_weights.round(4))
print()

# 分析注意力模式
print("注意力模式分析:")
print("-" * 70)
for i in range(SEQ_LEN):
    top_k = 3
    top_indices = np.argsort(avg_attn_weights[i])[-top_k:][::-1]
    top_weights = avg_attn_weights[i, top_indices]
    print(f"位置 {i} 最关注的位置: {top_indices} (权重: {top_weights.round(4)})")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 平均注意力权重热力图
ax = axes[0, 0]
im = ax.imshow(avg_attn_weights, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Average Attention Weights (All Heads)')
plt.colorbar(im, ax=ax)

# 2. 各个头的注意力权重
for h in range(min(4, NUM_HEADS)):
    ax = axes[0 + h // 2, 1 - h % 2] if h < 2 else axes[1 + (h-2) // 2, 1 - (h-2) % 2]
    if h < 2:
        ax = axes[0, 1] if h == 1 else axes[0, 0]
    else:
        ax = axes[1, 1] if h == 3 else axes[1, 0]

# 重新组织可视化
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 平均注意力权重热力图
ax = axes[0, 0]
im = ax.imshow(avg_attn_weights, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Key Position')
ax.set_ylabel('Query Position')
ax.set_title('Average Attention Weights')
plt.colorbar(im, ax=ax)

# 2-4. 各个头的注意力权重
for h in range(min(3, NUM_HEADS)):
    row = (h + 1) // 2
    col = (h + 1) % 2
    ax = axes[row, col]
    im = ax.imshow(attn_weights_list[h], cmap='YlOrRd', aspect='auto')
    ax.set_xlabel('Key Position')
    ax.set_ylabel('Query Position')
    ax.set_title(f'Head {h} Attention Weights')
    plt.colorbar(im, ax=ax)

# 5. 注意力权重分布
ax = axes[1, 1]
for h in range(NUM_HEADS):
    weights_flat = attn_weights_list[h].flatten()
    ax.hist(weights_flat, bins=30, alpha=0.5, label=f'Head {h}')
ax.set_xlabel('Attention Weight')
ax.set_ylabel('Frequency')
ax.set_title('Attention Weight Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/ch07_self_attention.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch07_self_attention.png")
