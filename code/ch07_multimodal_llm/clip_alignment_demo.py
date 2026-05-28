"""
CLIP对齐演示：展示对比学习如何实现视觉-语言对齐

本脚本演示：
1. 如何使用预训练的CLIP模型提取图像和文本特征
2. 如何计算图像-文本对的相似度
3. 如何理解对比学习的对齐效果

运行方式：
    python clip_alignment_demo.py

依赖：
    pip install openai-clip torch pillow matplotlib seaborn numpy
"""

import torch
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
import seaborn as sns


def load_clip_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    """加载预训练的CLIP模型"""
    try:
        import clip
    except ImportError:
        print("请先安装CLIP: pip install openai-clip")
        exit()

    model, preprocess = clip.load("ViT-B/32", device=device)
    return model, preprocess, device


def create_demo_images():
    """创建演示用的图像"""
    images = [
        Image.new('RGB', (224, 224), color='red'),      # 红色图像
        Image.new('RGB', (224, 224), color='blue'),     # 蓝色图像
        Image.new('RGB', (224, 224), color='green'),    # 绿色图像
    ]

    image_names = ["红色图像", "蓝色图像", "绿色图像"]
    return images, image_names


def create_demo_texts():
    """创建演示用的文本"""
    texts = [
        "a red image",
        "a blue image",
        "a green image",
        "a yellow image",  # 不匹配的文本
    ]
    return texts


def extract_image_features(model, preprocess, images, device):
    """提取图像特征"""
    image_features = []
    for img in images:
        img_tensor = preprocess(img).unsqueeze(0).to(device)
        with torch.no_grad():
            features = model.encode_image(img_tensor)
        image_features.append(features)

    return torch.cat(image_features, dim=0)


def extract_text_features(model, texts, device):
    """提取文本特征"""
    import clip

    text_features = []
    for text in texts:
        text_tokens = clip.tokenize(text).to(device)
        with torch.no_grad():
            features = model.encode_text(text_tokens)
        text_features.append(features)

    return torch.cat(text_features, dim=0)


def compute_similarity_matrix(image_features, text_features):
    """计算相似度矩阵"""
    # 归一化特征
    image_features = image_features / image_features.norm(dim=1, keepdim=True)
    text_features = text_features / text_features.norm(dim=1, keepdim=True)

    # 计算余弦相似度
    similarity_matrix = image_features @ text_features.T

    return similarity_matrix


def analyze_alignment(similarity_matrix, image_names, texts):
    """分析对齐效果"""
    print("=" * 60)
    print("CLIP对齐效果分析")
    print("=" * 60)
    print()

    print("相似度矩阵：")
    print(similarity_matrix.cpu().numpy())
    print()

    print("对齐效果分析：")
    print("-" * 60)

    for i, img_name in enumerate(image_names):
        similarities = similarity_matrix[i].cpu().numpy()
        best_match_idx = np.argmax(similarities)
        best_match_text = texts[best_match_idx]
        best_match_score = similarities[best_match_idx]

        print(f"\n{img_name}:")
        print(f"  最匹配的文本：'{best_match_text}'")
        print(f"  相似度：{best_match_score:.4f}")
        print(f"  所有相似度：{[f'{s:.4f}' for s in similarities]}")

        # 检查对齐是否正确
        if best_match_idx == i:
            print(f"  ✅ 对齐正确！")
        else:
            print(f"  ❌ 对齐错误！应该匹配 '{texts[i]}'")

    print()
    print("=" * 60)
    print("对齐效果总结：")
    print("-" * 60)

    # 计算对齐准确率
    correct = 0
    for i in range(len(image_names)):
        if np.argmax(similarity_matrix[i].cpu().numpy()) == i:
            correct += 1

    accuracy = correct / len(image_names) * 100
    print(f"对齐准确率：{accuracy:.1f}% ({correct}/{len(image_names)})")
    print()


def visualize_alignment(similarity_matrix, image_names, texts, output_file="../../assets/ch07_clip_alignment.png"):
    """可视化对齐效果"""
    fig, ax = plt.subplots(figsize=(10, 6))

    # 相似度矩阵
    sim_matrix = similarity_matrix.cpu().numpy()

    # 绘制热力图
    sns.heatmap(sim_matrix,
                xticklabels=texts,
                yticklabels=image_names,
                annot=True,
                fmt='.3f',
                cmap='RdYlGn',
                vmin=0,
                vmax=1,
                ax=ax,
                cbar_kws={'label': 'Similarity'})

    ax.set_title("CLIP Image-Text Similarity Matrix\n(Brighter diagonal indicates better alignment)", fontsize=14, pad=20)
    ax.set_xlabel("Text", fontsize=12)
    ax.set_ylabel("Image", fontsize=12)

    plt.tight_layout()
    plt.savefig(output_file, dpi=100, bbox_inches='tight')
    print(f"✅ 热力图已保存为 {output_file}")

    # 显示图表
    try:
        plt.show()
    except:
        pass


def main():
    """主函数"""
    print("CLIP对齐演示")
    print("=" * 60)
    print()

    # 加载模型
    print("1. 加载CLIP模型...")
    model, preprocess, device = load_clip_model()
    print(f"   ✅ 模型已加载到 {device}")
    print()

    # 创建演示数据
    print("2. 创建演示数据...")
    images, image_names = create_demo_images()
    texts = create_demo_texts()
    print(f"   ✅ 创建了 {len(images)} 张图像和 {len(texts)} 个文本")
    print()

    # 提取特征
    print("3. 提取图像和文本特征...")
    image_features = extract_image_features(model, preprocess, images, device)
    text_features = extract_text_features(model, texts, device)
    print(f"   ✅ 图像特征形状：{image_features.shape}")
    print(f"   ✅ 文本特征形状：{text_features.shape}")
    print()

    # 计算相似度
    print("4. 计算相似度矩阵...")
    similarity_matrix = compute_similarity_matrix(image_features, text_features)
    print(f"   ✅ 相似度矩阵形状：{similarity_matrix.shape}")
    print()

    # 分析对齐效果
    print("5. 分析对齐效果...")
    analyze_alignment(similarity_matrix, image_names, texts)

    # 可视化
    print("6. 可视化对齐效果...")
    visualize_alignment(similarity_matrix, image_names, texts)
    print()

    print("=" * 60)
    print("演示完成！")
    print("=" * 60)


if __name__ == "__main__":
    main()
