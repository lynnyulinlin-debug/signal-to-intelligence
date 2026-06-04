"""
实验5.1a：BPE 分词算法演示
对应章节：第5章 5.1 预训练
目标：实现 BPE 算法，展示词表从字符逐步合并的过程
"""

from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch05_bpe_tokenization.png")
DEFAULT_CORPUS = (
    ["low"] * 5
    + ["lower"] * 2
    + ["newest"] * 6
    + ["widest"] * 3
    + ["new"] * 4
    + ["wide"] * 3
)
NUM_MERGES = 12


# ============ BPE 核心实现 ============
def get_vocab(corpus):
    """将语料转为字符级词频词典"""
    vocab = Counter()
    for word in corpus:
        chars = list(word) + ["</w>"]
        vocab[" ".join(chars)] += 1
    return vocab


def get_pairs(vocab):
    """统计所有相邻字符对的频率"""
    pairs = Counter()
    for word, freq in vocab.items():
        symbols = word.split()
        for i in range(len(symbols) - 1):
            pairs[(symbols[i], symbols[i + 1])] += freq
    return pairs


def merge_vocab(pair, vocab):
    """合并词典中最高频的字符对"""
    new_vocab = {}
    bigram = " ".join(pair)
    replacement = "".join(pair)
    for word, freq in vocab.items():
        new_word = word.replace(bigram, replacement)
        new_vocab[new_word] = freq
    return new_vocab


def run_bpe(corpus, num_merges=10):
    """运行 BPE，返回每步合并记录"""
    vocab = get_vocab(corpus)
    merges = []
    vocab_sizes = [sum(len(w.split()) for w in vocab)]

    for _ in range(num_merges):
        pairs = get_pairs(vocab)
        if not pairs:
            break
        best_pair = max(pairs, key=pairs.get)
        best_freq = pairs[best_pair]
        vocab = merge_vocab(best_pair, vocab)
        merged = "".join(best_pair)
        merges.append((best_pair, best_freq, merged))
        vocab_sizes.append(sum(len(w.split()) for w in vocab))

    return merges, vocab_sizes


def plot_bpe(merges, vocab_sizes, output_path=OUTPUT_PATH):
    """保存 BPE 合并过程可视化图。"""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle(
        "BPE Tokenization: Building a Subword Vocabulary",
        fontsize=14,
        fontweight="bold",
    )

    ax1 = axes[0]
    steps = list(range(len(merges)))
    freqs = [m[1] for m in merges]
    labels = [f'"{m[0][0]}"+""{m[0][1]}"→"{m[2]}"' for m in merges]

    colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(merges)))
    bars = ax1.barh(steps, freqs, color=colors, edgecolor="white", linewidth=0.5)

    for bar, label in zip(bars, labels):
        ax1.text(
            bar.get_width() + 0.1,
            bar.get_y() + bar.get_height() / 2,
            label,
            va="center",
            fontsize=8.5,
        )

    ax1.set_yticks(steps)
    ax1.set_yticklabels([f"Step {i + 1}" for i in steps], fontsize=9)
    ax1.set_xlabel("Merge Frequency", fontsize=10)
    ax1.set_title("Top Merges by Frequency", fontsize=11)
    ax1.set_xlim(0, max(freqs) * 1.8)
    ax1.invert_yaxis()
    ax1.spines["top"].set_visible(False)
    ax1.spines["right"].set_visible(False)

    ax2 = axes[1]
    x = list(range(len(vocab_sizes)))
    ax2.plot(
        x,
        vocab_sizes,
        "o-",
        color="#2196F3",
        linewidth=2,
        markersize=6,
        label="Vocab size (tokens)",
    )
    ax2.fill_between(x, vocab_sizes, alpha=0.15, color="#2196F3")

    ax2.axhline(
        vocab_sizes[0],
        color="#FF5722",
        linestyle="--",
        linewidth=1.2,
        label=f"Initial (char-level): {vocab_sizes[0]}",
    )
    ax2.axhline(
        vocab_sizes[-1],
        color="#4CAF50",
        linestyle="--",
        linewidth=1.2,
        label=f"After {NUM_MERGES} merges: {vocab_sizes[-1]}",
    )

    ax2.set_xlabel("BPE Merge Steps", fontsize=10)
    ax2.set_ylabel("Total Tokens in Corpus", fontsize=10)
    ax2.set_title("Corpus Token Count Decreases with Merges", fontsize=11)
    ax2.legend(fontsize=9)
    ax2.spines["top"].set_visible(False)
    ax2.spines["right"].set_visible(False)

    ax2.annotate(
        "Char-level\n(many tokens)",
        xy=(0, vocab_sizes[0]),
        xytext=(2, vocab_sizes[0] + 2),
        fontsize=8,
        color="#FF5722",
        arrowprops=dict(arrowstyle="->", color="#FF5722", lw=1),
    )
    ax2.annotate(
        "Subword-level\n(fewer tokens)",
        xy=(NUM_MERGES, vocab_sizes[-1]),
        xytext=(NUM_MERGES - 4, vocab_sizes[-1] + 3),
        fontsize=8,
        color="#4CAF50",
        arrowprops=dict(arrowstyle="->", color="#4CAF50", lw=1),
    )

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    merges, vocab_sizes = run_bpe(DEFAULT_CORPUS, NUM_MERGES)
    output_path = plot_bpe(merges, vocab_sizes)
    print(f"Saved: {output_path}")


if __name__ == "__main__":
    main()
