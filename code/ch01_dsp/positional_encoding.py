"""
实验1.2：位置编码
对应章节：第1章 - 数字信号处理基础
目标：打印位置编码矩阵前几行，观察其周期性和结构
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
SEQ_LEN = 32  # 序列长度
D_MODEL = 64  # 模型维度

# ============ 核心逻辑 ============
def positional_encoding(seq_len, d_model):
    """
    生成Transformer位置编码矩阵
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
    div_term = np.exp(np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model))  # (d_model/2,)

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数位置：sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数位置：cos

    return pe

# 生成位置编码
pe = positional_encoding(SEQ_LEN, D_MODEL)

# ============ 结果输出 ============
print("=" * 70)
print("Positional Encoding")
print("=" * 70)
print(f"Sequence length: {SEQ_LEN}")
print(f"Model dimension: {D_MODEL}")
print(f"Encoding matrix shape: {pe.shape}")
print()

# 打印前5个位置的编码（前8个维度）
print("First 5 positions (first 8 dimensions):")
print("-" * 70)
for pos in range(5):
    print(f"Position {pos}: {pe[pos, :8].round(4)}")
print()

# 验证周期性：相邻位置的编码应该通过线性变换相关
print("Periodicity check:")
print("-" * 70)
# 计算位置0和位置1的编码之间的相似度
sim_01 = np.dot(pe[0], pe[1]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[1]))
sim_12 = np.dot(pe[1], pe[2]) / (np.linalg.norm(pe[1]) * np.linalg.norm(pe[2]))
print(f"Cosine similarity between positions 0 and 1: {sim_01:.4f}")
print(f"Cosine similarity between positions 1 and 2: {sim_12:.4f}")
print()

# 验证正交性：不同维度的编码应该近似正交
print("Orthogonality check (first 4 dimensions):")
print("-" * 70)
pe_subset = pe[:, :4]
gram_matrix = pe_subset.T @ pe_subset / SEQ_LEN
print("Gram matrix (should be near identity after normalization):")
print(gram_matrix.round(4))
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# 1. 位置编码热力图
ax = axes[0, 0]
im = ax.imshow(pe.T, cmap='RdBu', aspect='auto', vmin=-1, vmax=1)
ax.set_xlabel('Position')
ax.set_ylabel('Dimension')
ax.set_title('Positional Encoding Heatmap')
plt.colorbar(im, ax=ax)

# 2. 不同维度的周期性
ax = axes[0, 1]
for dim in [0, 2, 4, 6]:
    ax.plot(pe[:, dim], label=f'Dim {dim}', linewidth=1.5)
ax.set_xlabel('Position')
ax.set_ylabel('Value')
ax.set_title('Periodicity of Different Dimensions')
ax.legend()
ax.grid(True, alpha=0.3)

# 3. 位置编码的范数
ax = axes[1, 0]
norms = np.linalg.norm(pe, axis=1)
ax.plot(norms, 'b-', linewidth=2)
ax.set_xlabel('Position')
ax.set_ylabel('Norm')
ax.set_title('Norm of Positional Encoding')
ax.grid(True, alpha=0.3)

# 4. 相邻位置的相似度
ax = axes[1, 1]
similarities = []
for i in range(SEQ_LEN - 1):
    sim = np.dot(pe[i], pe[i+1]) / (np.linalg.norm(pe[i]) * np.linalg.norm(pe[i+1]))
    similarities.append(sim)
ax.plot(similarities, 'g-', linewidth=2)
ax.set_xlabel('Position')
ax.set_ylabel('Cosine Similarity')
ax.set_title('Similarity Between Adjacent Positions')
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('assets/ch01_positional_encoding.png', dpi=100, bbox_inches='tight')
print("Figure saved to: assets/ch01_positional_encoding.png")
