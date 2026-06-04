"""
实验8.1：ViT Patches
对应章节：第8章 - 视觉与多模态（ViT）
目标：ViT的patch embedding可视化
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
IMG_SIZE = 224
PATCH_SIZE = 16
D_MODEL = 768
NUM_PATCHES = (IMG_SIZE // PATCH_SIZE) ** 2
OUTPUT_PATH = Path("assets/ch07_vit_patches.png")


# ============ 核心逻辑 ============
def create_patches(image, patch_size):
    """
    将图像分割成patches
    image: (H, W, C)
    返回: (num_patches, patch_size*patch_size*C)
    """
    h, w, _ = image.shape
    num_patches_h = h // patch_size
    num_patches_w = w // patch_size

    patches = []
    for i in range(num_patches_h):
        for j in range(num_patches_w):
            patch = image[
                i * patch_size : (i + 1) * patch_size,
                j * patch_size : (j + 1) * patch_size,
                :,
            ]
            patches.append(patch.flatten())

    return np.array(patches)


def patch_embedding(patches, d_model, seed=42):
    """
    将patches投射到d_model维度
    patches: (num_patches, patch_dim)
    返回: (num_patches, d_model)
    """
    rng = np.random.RandomState(seed)
    patch_dim = patches.shape[1]
    w = rng.randn(patch_dim, d_model) * 0.01
    embeddings = patches @ w
    return embeddings


def create_gradient_image(img_size=IMG_SIZE):
    """生成合成梯度图像。"""
    image = np.zeros((img_size, img_size, 3))
    for i in range(img_size):
        for j in range(img_size):
            image[i, j, 0] = i / img_size
            image[i, j, 1] = j / img_size
            image[i, j, 2] = (i + j) / (2 * img_size)
    return image


def compute_patch_similarity(patches):
    norms = np.linalg.norm(patches, axis=1, keepdims=True)
    return patches @ patches.T / (norms * norms.T + 1e-8)


def run_experiment(img_size=IMG_SIZE, patch_size=PATCH_SIZE, d_model=D_MODEL, seed=42):
    image = create_gradient_image(img_size)
    patches = create_patches(image, patch_size)
    embeddings = patch_embedding(patches, d_model, seed=seed)
    patch_similarity = compute_patch_similarity(patches)
    return {
        "image": image,
        "patches": patches,
        "embeddings": embeddings,
        "patch_similarity": patch_similarity,
        "img_size": img_size,
        "patch_size": patch_size,
        "d_model": d_model,
        "num_patches": (img_size // patch_size) ** 2,
    }


def print_summary(result):
    print("=" * 70)
    print("ViT Patches 可视化")
    print("=" * 70)
    print(f"图像大小: {result['img_size']}x{result['img_size']}")
    print(f"Patch大小: {result['patch_size']}x{result['patch_size']}")
    print(f"Patch数量: {result['num_patches']}")
    print(f"Patch维度: {result['patches'].shape[1]}")
    print(f"Embedding维度: {result['d_model']}")
    print()

    print("Patch统计:")
    print("-" * 70)
    patches = result["patches"]
    print(f"Patch范围: [{patches.min():.4f}, {patches.max():.4f}]")
    print(f"Patch均值: {patches.mean():.4f}")
    print(f"Patch方差: {patches.var():.4f}")
    print()

    print("Embedding统计:")
    print("-" * 70)
    embeddings = result["embeddings"]
    print(f"Embedding范围: [{embeddings.min():.4f}, {embeddings.max():.4f}]")
    print(f"Embedding均值: {embeddings.mean():.4f}")
    print(f"Embedding方差: {embeddings.var():.4f}")
    print()

    print("Patch相似度统计:")
    print("-" * 70)
    patch_similarity = result["patch_similarity"]
    print(f"相似度范围: [{patch_similarity.min():.4f}, {patch_similarity.max():.4f}]")
    print(f"相似度均值: {patch_similarity.mean():.4f}")
    print()

    print("=" * 70)


def plot_vit_patches(result, output_path=OUTPUT_PATH):
    image = result["image"]
    patches = result["patches"]
    embeddings = result["embeddings"]
    patch_similarity = result["patch_similarity"]
    img_size = result["img_size"]
    patch_size = result["patch_size"]
    num_patches = result["num_patches"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 12))

    ax = axes[0, 0]
    ax.imshow(image)
    ax.set_title("Original Image")
    ax.axis("off")

    ax = axes[0, 1]
    grid = np.zeros((img_size, img_size))
    for idx, patch in enumerate(patches):
        i = idx // (img_size // patch_size)
        j = idx % (img_size // patch_size)
        patch_mean = patch.mean()
        grid[
            i * patch_size : (i + 1) * patch_size,
            j * patch_size : (j + 1) * patch_size,
        ] = patch_mean
    ax.imshow(grid, cmap="gray")
    ax.set_title("Patch Mean Values")
    ax.axis("off")

    ax = axes[1, 0]
    im = ax.imshow(patch_similarity, cmap="YlOrRd", aspect="auto")
    ax.set_xlabel("Patch Index")
    ax.set_ylabel("Patch Index")
    ax.set_title("Patch Similarity Matrix")
    plt.colorbar(im, ax=ax)

    ax = axes[1, 1]
    mean = embeddings.mean(axis=0)
    centered = embeddings - mean
    u, _, _ = np.linalg.svd(centered.T @ centered, full_matrices=False)
    proj = centered @ u[:, :2]

    scatter = ax.scatter(
        proj[:, 0], proj[:, 1], c=np.arange(num_patches), cmap="viridis", s=50, alpha=0.6
    )
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title("Embedding Distribution (PCA)")
    plt.colorbar(scatter, ax=ax, label="Patch Index")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print(f"Patches形状: {result['patches'].shape}")
    print_summary(result)
    output_path = plot_vit_patches(result)
    print(f"图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
