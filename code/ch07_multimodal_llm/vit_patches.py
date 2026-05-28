"""
实验8.1：ViT Patches
对应章节：第8章 - 视觉与多模态（ViT）
目标：ViT的patch embedding可视化
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
np.random.seed(42)
IMG_SIZE = 224
PATCH_SIZE = 16
D_MODEL = 768
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2

# ============ 核心逻辑 ============
def create_patches(image, patch_size):
    """
    将图像分割成patches
    image: (H, W, C)
    返回: (num_patches, patch_size*patch_size*C)
    """
    H, W, C = image.shape
    num_patches_h = H // patch_size
    num_patches_w = W // patch_size

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[i*patch_size:(i+1)*patch_size,
                         j*patch_size:(j+1)*patch_size, :]
            patches.append(patch.flatten())

    return np.array(patches)

def patch_embedding(patches, d_model):
    """
    将patches投射到d_model维度
    patches: (num_patches, patch_dim)
    返回: (num_patches, d_model)
    """
    patch_dim = patches.shape[1]
    W = np.random.randn(patch_dim, d_model) * 0.01
    embeddings = patches @ W
    return embeddings

# 生成合成图像（简单的梯度图）
image = np.zeros((IMG_SIZE, IMG_SIZE, 3))
for i in range(IMG_SIZE):
    for j in range(IMG_SIZE):
        image[i, j, 0] = i / IMG_SIZE  # R通道：垂直梯度
        image[i, j, 1] = j / IMG_SIZE  # G通道：水平梯度
        image[i, j, 2] = (i + j) / (2 * IMG_SIZE)  # B通道：对角梯度

# 提取patches
patches = create_patches(image, PATCH_SIZE)
print(f"Patches形状: {patches.shape}")

# 计算patch embeddings
embeddings = patch_embedding(patches, D_MODEL)

# 计算patch之间的相似度
patch_similarity = patches @ patches.T / (np.linalg.norm(patches, axis=1, keepdims=True) *
                                          np.linalg.norm(patches, axis=1, keepdims=True).T + 1e-8)

# ============ 结果输出 ============
print("=" * 70)
print("ViT Patches 可视化")
print("=" * 70)
print(f"图像大小: {IMG_SIZE}x{IMG_SIZE}")
print(f"Patch大小: {PATCH_SIZE}x{PATCH_SIZE}")
print(f"Patch数量: {NUM_PATCHES}")
print(f"Patch维度: {patches.shape[1]}")
print(f"Embedding维度: {D_MODEL}")
print()

print("Patch统计:")
print("-" * 70)
print(f"Patch范围: [{patches.min():.4f}, {patches.max():.4f}]")
print(f"Patch均值: {patches.mean():.4f}")
print(f"Patch方差: {patches.var():.4f}")
print()

print("Embedding统计:")
print("-" * 70)
print(f"Embedding范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
print(f"Embedding均值: {embeddings.mean():.4f}")
print(f"Embedding方差: {embeddings.var():.4f}")
print()

print("Patch相似度统计:")
print("-" * 70)
print(f"相似度范围: [{patch_similarity.min():.4f}, {patch_similarity.max():.4f}]")
print(f"相似度均值: {patch_similarity.mean():.4f}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 原始图像
ax = axes[0, 0]
ax.imshow(image)
ax.set_title('Original Image')
ax.axis('off')

# 2. Patch网格
ax = axes[0, 1]
grid = np.zeros((IMG_SIZE, IMG_SIZE))
for idx, patch in enumerate(patches):
    i = idx // (IMG_SIZE // PATCH_SIZE)
    j = idx % (IMG_SIZE // PATCH_SIZE)
    patch_mean = patch.mean()
    grid[i*PATCH_SIZE:(i+1)*PATCH_SIZE, j*PATCH_SIZE:(j+1)*PATCH_SIZE] = patch_mean
ax.imshow(grid, cmap='gray')
ax.set_title('Patch Mean Values')
ax.axis('off')

# 3. Patch相似度热力图
ax = axes[1, 0]
im = ax.imshow(patch_similarity, cmap='YlOrRd', aspect='auto')
ax.set_xlabel('Patch Index')
ax.set_ylabel('Patch Index')
ax.set_title('Patch Similarity Matrix')
plt.colorbar(im, ax=ax)

# 4. Embedding分布（PCA投影）
ax = axes[1, 1]
# 简单的PCA投影到2D
mean = embeddings.mean(axis=0)
centered = embeddings - mean
U, S, Vt = np.linalg.svd(centered.T @ centered, full_matrices=False)
proj = centered @ U[:, :2]

scatter = ax.scatter(proj[:, 0], proj[:, 1], c=np.arange(NUM_PATCHES),
                     cmap='viridis', s=50, alpha=0.6)
ax.set_xlabel('PC1')
ax.set_ylabel('PC2')
ax.set_title('Embedding Distribution (PCA)')
plt.colorbar(scatter, ax=ax, label='Patch Index')

plt.tight_layout()
plt.savefig('assets/ch08_vit_patches.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch08_vit_patches.png")
