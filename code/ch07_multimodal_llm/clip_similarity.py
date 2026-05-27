"""
实验8.2：CLIP相似度
对应章节：第8章 - 视觉与多模态（ViT）
目标：计算文本与图像embedding的余弦相似度
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)
D_MODEL = 512
NUM_IMAGES = 10
NUM_TEXTS = 10

# ============ 核心逻辑 ============
def text_to_embedding(text, d_model):
    """
    简化的文本编码器：将文本转换为embedding
    实际应用中会使用BERT或其他预训练模型
    """
    # 简单的哈希+投射方式
    text_hash = hash(text) % 10000
    np.random.seed(text_hash)
    embedding = np.random.randn(d_model) * 0.1
    return embedding / (np.linalg.norm(embedding) + 1e-8)

def image_to_embedding(image_id, d_model):
    """
    简化的图像编码器：将图像转换为embedding
    实际应用中会使用ViT或其他预训练模型
    """
    np.random.seed(image_id)
    embedding = np.random.randn(d_model) * 0.1
    return embedding / (np.linalg.norm(embedding) + 1e-8)

# 生成图像和文本描述
image_descriptions = [
    "a dog running in the park",
    "a cat sleeping on a bed",
    "a bird flying in the sky",
    "a car on the road",
    "a tree in the forest",
    "a person reading a book",
    "a flower in the garden",
    "a mountain landscape",
    "a sunset over the ocean",
    "a city street at night"
]

# 生成相关和不相关的文本
related_texts = [
    "a dog playing outside",
    "a cat resting indoors",
    "a bird in flight",
    "a vehicle driving",
    "a plant in nature",
    "someone with a book",
    "a blooming flower",
    "mountains and hills",
    "sun setting on water",
    "urban street scene"
]

unrelated_texts = [
    "a computer on a desk",
    "a pizza on a plate",
    "a swimming pool",
    "a musical instrument",
    "a cup of coffee",
    "a pair of shoes",
    "a mobile phone",
    "a bicycle",
    "a painting on wall",
    "a clock on wall"
]

# 计算embeddings
image_embeddings = np.array([image_to_embedding(i, D_MODEL) for i in range(NUM_IMAGES)])
related_embeddings = np.array([text_to_embedding(text, D_MODEL) for text in related_texts])
unrelated_embeddings = np.array([text_to_embedding(text, D_MODEL) for text in unrelated_texts])

# 计算相似度矩阵
def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# 相关文本的相似度
related_similarity = np.zeros((NUM_IMAGES, NUM_TEXTS))
for i in range(NUM_IMAGES):
    for j in range(NUM_TEXTS):
        related_similarity[i, j] = cosine_similarity(image_embeddings[i], related_embeddings[j])

# 不相关文本的相似度
unrelated_similarity = np.zeros((NUM_IMAGES, NUM_TEXTS))
for i in range(NUM_IMAGES):
    for j in range(NUM_TEXTS):
        unrelated_similarity[i, j] = cosine_similarity(image_embeddings[i], unrelated_embeddings[j])

# ============ 结果输出 ============
print("=" * 70)
print("CLIP 相似度计算")
print("=" * 70)
print(f"Embedding维度: {D_MODEL}")
print(f"图像数量: {NUM_IMAGES}")
print(f"文本数量: {NUM_TEXTS}")
print()

print("相关文本-图像相似度统计:")
print("-" * 70)
print(f"最小值: {related_similarity.min():.4f}")
print(f"最大值: {related_similarity.max():.4f}")
print(f"均值: {related_similarity.mean():.4f}")
print(f"标准差: {related_similarity.std():.4f}")
print()

print("不相关文本-图像相似度统计:")
print("-" * 70)
print(f"最小值: {unrelated_similarity.min():.4f}")
print(f"最大值: {unrelated_similarity.max():.4f}")
print(f"均值: {unrelated_similarity.mean():.4f}")
print(f"标准差: {unrelated_similarity.std():.4f}")
print()

print("相似度对比:")
print("-" * 70)
print(f"相关文本平均相似度: {related_similarity.mean():.4f}")
print(f"不相关文本平均相似度: {unrelated_similarity.mean():.4f}")
print(f"差异: {related_similarity.mean() - unrelated_similarity.mean():.4f}")
print()

# 计算检索准确率（对角线元素应该最大）
print("检索准确率（对角线匹配）:")
print("-" * 70)
correct = 0
for i in range(NUM_IMAGES):
    # 找到与第i个图像最相似的文本
    best_idx = np.argmax(related_similarity[i])
    if best_idx == i:
        correct += 1
accuracy = correct / NUM_IMAGES
print(f"准确率: {accuracy:.2%}")
print()

print("=" * 70)

# ============ 可视化 ============
fig, axes = plt.subplots(2, 2, figsize=(14, 12))

# 1. 相关文本相似度热力图
ax = axes[0, 0]
im = ax.imshow(related_similarity, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel('Text Index')
ax.set_ylabel('Image Index')
ax.set_title('Related Text-Image Similarity')
plt.colorbar(im, ax=ax)

# 2. 不相关文本相似度热力图
ax = axes[0, 1]
im = ax.imshow(unrelated_similarity, cmap='YlOrRd', aspect='auto', vmin=0, vmax=1)
ax.set_xlabel('Text Index')
ax.set_ylabel('Image Index')
ax.set_title('Unrelated Text-Image Similarity')
plt.colorbar(im, ax=ax)

# 3. 相似度分布对比
ax = axes[1, 0]
ax.hist(related_similarity.flatten(), bins=30, alpha=0.6, label='Related', color='green')
ax.hist(unrelated_similarity.flatten(), bins=30, alpha=0.6, label='Unrelated', color='red')
ax.set_xlabel('Cosine Similarity')
ax.set_ylabel('Frequency')
ax.set_title('Similarity Distribution')
ax.legend()
ax.grid(True, alpha=0.3, axis='y')

# 4. 对角线相似度 vs 非对角线相似度
ax = axes[1, 1]
diagonal_related = np.diag(related_similarity)
off_diagonal_related = related_similarity[~np.eye(NUM_IMAGES, dtype=bool)]

positions = [1, 2]
data = [diagonal_related, off_diagonal_related]
bp = ax.boxplot(data, positions=positions, labels=['Diagonal\n(Matched)', 'Off-diagonal\n(Mismatched)'],
                 patch_artist=True)

for patch, color in zip(bp['boxes'], ['lightgreen', 'lightcoral']):
    patch.set_facecolor(color)

ax.set_ylabel('Cosine Similarity')
ax.set_title('Matched vs Mismatched Pairs')
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('assets/ch08_clip_similarity.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch08_clip_similarity.png")
