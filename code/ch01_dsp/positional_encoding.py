"""
实验1.2：位置编码
对应章节：第1章 - 数字信号处理基础
目标：打印位置编码矩阵前几行，观察其周期性和结构
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
SEQ_LEN = 32  # 序列长度
D_MODEL = 64  # 模型维度
OUTPUT_PATH = Path("assets/ch01_positional_encoding.png")


# ============ 核心逻辑 ============
def positional_encoding(seq_len, d_model):
    """
    生成Transformer位置编码矩阵
    PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
    PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
    """
    pe = np.zeros((seq_len, d_model))
    position = np.arange(seq_len).reshape(-1, 1)  # (seq_len, 1)
    div_term = np.exp(
        np.arange(0, d_model, 2) * -(np.log(10000.0) / d_model)
    )  # (d_model/2,)

    pe[:, 0::2] = np.sin(position * div_term)  # 偶数位置：sin
    pe[:, 1::2] = np.cos(position * div_term)  # 奇数位置：cos

    return pe


def adjacent_similarities(pe):
    """计算相邻位置编码的余弦相似度。"""
    similarities = []
    for i in range(len(pe) - 1):
        sim = np.dot(pe[i], pe[i + 1]) / (np.linalg.norm(pe[i]) * np.linalg.norm(pe[i + 1]))
        similarities.append(sim)
    return np.array(similarities)


def run_experiment(seq_len=SEQ_LEN, d_model=D_MODEL):
    pe = positional_encoding(seq_len, d_model)
    similarities = adjacent_similarities(pe)
    return {
        "pe": pe,
        "similarities": similarities,
        "seq_len": seq_len,
        "d_model": d_model,
    }


def print_summary(result):
    pe = result["pe"]

    print("=" * 70)
    print("Positional Encoding")
    print("=" * 70)
    print(f"Sequence length: {result['seq_len']}")
    print(f"Model dimension: {result['d_model']}")
    print(f"Encoding matrix shape: {pe.shape}")
    print()

    print("First 5 positions (first 8 dimensions):")
    print("-" * 70)
    for pos in range(5):
        print(f"Position {pos}: {pe[pos, :8].round(4)}")
    print()

    print("Periodicity check:")
    print("-" * 70)
    sim_01 = np.dot(pe[0], pe[1]) / (np.linalg.norm(pe[0]) * np.linalg.norm(pe[1]))
    sim_12 = np.dot(pe[1], pe[2]) / (np.linalg.norm(pe[1]) * np.linalg.norm(pe[2]))
    print(f"Cosine similarity between positions 0 and 1: {sim_01:.4f}")
    print(f"Cosine similarity between positions 1 and 2: {sim_12:.4f}")
    print()

    print("Orthogonality check (first 4 dimensions):")
    print("-" * 70)
    pe_subset = pe[:, :4]
    gram_matrix = pe_subset.T @ pe_subset / result["seq_len"]
    print("Gram matrix (should be near identity after normalization):")
    print(gram_matrix.round(4))
    print()

    print("=" * 70)


def plot_positional_encoding(result, output_path=OUTPUT_PATH):
    pe = result["pe"]
    similarities = result["similarities"]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ax = axes[0, 0]
    im = ax.imshow(pe.T, cmap="RdBu", aspect="auto", vmin=-1, vmax=1)
    ax.set_xlabel("Position")
    ax.set_ylabel("Dimension")
    ax.set_title("Positional Encoding Heatmap")
    plt.colorbar(im, ax=ax)

    ax = axes[0, 1]
    for dim in [0, 2, 4, 6]:
        ax.plot(pe[:, dim], label=f"Dim {dim}", linewidth=1.5)
    ax.set_xlabel("Position")
    ax.set_ylabel("Value")
    ax.set_title("Periodicity of Different Dimensions")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1, 0]
    norms = np.linalg.norm(pe, axis=1)
    ax.plot(norms, "b-", linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Norm")
    ax.set_title("Norm of Positional Encoding")
    ax.grid(True, alpha=0.3)

    ax = axes[1, 1]
    ax.plot(similarities, "g-", linewidth=2)
    ax.set_xlabel("Position")
    ax.set_ylabel("Cosine Similarity")
    ax.set_title("Similarity Between Adjacent Positions")
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_positional_encoding(result)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
