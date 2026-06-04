"""
实验8.2：CLIP相似度
对应章节：第8章 - 视觉与多模态（ViT）
目标：计算文本与图像embedding的余弦相似度
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
D_MODEL = 512
NUM_IMAGES = 10
NUM_TEXTS = 10
OUTPUT_PATH = Path("assets/ch07_clip_similarity.png")

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
    "a city street at night",
]

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
    "urban street scene",
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
    "a clock on wall",
]


# ============ 核心逻辑 ============
def text_to_embedding(text, d_model):
    """
    简化的文本编码器：将文本转换为embedding
    实际应用中会使用BERT或其他预训练模型
    """
    text_hash = hash(text) % 10000
    rng = np.random.RandomState(text_hash)
    embedding = rng.randn(d_model) * 0.1
    return embedding / (np.linalg.norm(embedding) + 1e-8)


def image_to_embedding(image_id, d_model):
    """
    简化的图像编码器：将图像转换为embedding
    实际应用中会使用ViT或其他预训练模型
    """
    rng = np.random.RandomState(image_id)
    embedding = rng.randn(d_model) * 0.1
    return embedding / (np.linalg.norm(embedding) + 1e-8)


def cosine_similarity(a, b):
    """计算两个向量的余弦相似度"""
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)


def compute_similarity_matrix(image_embeddings, text_embeddings):
    similarity = np.zeros((len(image_embeddings), len(text_embeddings)))
    for i in range(len(image_embeddings)):
        for j in range(len(text_embeddings)):
            similarity[i, j] = cosine_similarity(image_embeddings[i], text_embeddings[j])
    return similarity


def retrieval_accuracy(similarity):
    correct = 0
    for i in range(similarity.shape[0]):
        best_idx = np.argmax(similarity[i])
        if best_idx == i:
            correct += 1
    return correct / similarity.shape[0]


def run_experiment(d_model=D_MODEL):
    image_embeddings = np.array([image_to_embedding(i, d_model) for i in range(NUM_IMAGES)])
    related_embeddings = np.array([text_to_embedding(text, d_model) for text in related_texts])
    unrelated_embeddings = np.array([text_to_embedding(text, d_model) for text in unrelated_texts])

    related_similarity = compute_similarity_matrix(image_embeddings, related_embeddings)
    unrelated_similarity = compute_similarity_matrix(image_embeddings, unrelated_embeddings)
    accuracy = retrieval_accuracy(related_similarity)

    return {
        "image_embeddings": image_embeddings,
        "related_embeddings": related_embeddings,
        "unrelated_embeddings": unrelated_embeddings,
        "related_similarity": related_similarity,
        "unrelated_similarity": unrelated_similarity,
        "accuracy": accuracy,
        "d_model": d_model,
    }


def print_summary(result):
    related_similarity = result["related_similarity"]
    unrelated_similarity = result["unrelated_similarity"]

    print("=" * 70)
    print("CLIP 相似度计算")
    print("=" * 70)
    print(f"Embedding维度: {result['d_model']}")
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

    print("检索准确率（对角线匹配）:")
    print("-" * 70)
    print(f"准确率: {result['accuracy']:.2%}")
    print()
    print("=" * 70)


def plot_clip_similarity(result, output_path=OUTPUT_PATH):
    related_similarity = result["related_similarity"]
    unrelated_similarity = result["unrelated_similarity"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    im = ax.imshow(related_similarity, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Text Index")
    ax.set_ylabel("Image Index")
    ax.set_title("Related Text-Image Similarity")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    im = ax.imshow(unrelated_similarity, cmap="YlOrRd", aspect="auto", vmin=0, vmax=1)
    ax.set_xlabel("Text Index")
    ax.set_ylabel("Image Index")
    ax.set_title("Unrelated Text-Image Similarity")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 0]
    ax.hist(related_similarity.flatten(), bins=30, alpha=0.6, label="Related", color="green")
    ax.hist(unrelated_similarity.flatten(), bins=30, alpha=0.6, label="Unrelated", color="red")
    ax.set_xlabel("Cosine Similarity")
    ax.set_ylabel("Frequency")
    ax.set_title("Similarity Distribution")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")

    ax = axes[1, 1]
    diagonal_related = np.diag(related_similarity)
    off_diagonal_related = related_similarity[~np.eye(NUM_IMAGES, dtype=bool)]

    data = [diagonal_related, off_diagonal_related]
    bp = ax.boxplot(
        data,
        positions=[1, 2],
        labels=["Diagonal\n(Matched)", "Off-diagonal\n(Mismatched)"],
        patch_artist=True,
    )

    for patch, color in zip(bp["boxes"], ["lightgreen", "lightcoral"]):
        patch.set_facecolor(color)

    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Matched vs Mismatched Pairs")
    ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_clip_similarity(result)
    print(f"图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
