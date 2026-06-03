"""
生成 ch07_case_studies.png 和 ch07_multimodal_applications.png
两张性能对比图，供 docs 引用。无需模型，纯 matplotlib。

运行：python code/ch07_multimodal_llm/gen_case_studies_chart.py
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False


# ── case_studies ────────────────────────────────────────────────────────────

def plot_case_studies():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Case Studies: LLaVA-1.5 vs Qwen2.5-VL Performance",
                 fontsize=13, fontweight="bold")

    datasets = [
        {
            "title": "Document Understanding",
            "metrics": ["Field Accuracy", "Amount Accuracy", "Table Accuracy"],
            "llava": [65.3, 58.2, 52.1],
            "qwen":  [82.1, 76.5, 71.3],
        },
        {
            "title": "Chart Analysis",
            "metrics": ["Data Point", "Trend Judgment", "Anomaly Detection"],
            "llava": [68.2, 72.1, 61.5],
            "qwen":  [85.7, 88.3, 79.2],
        },
        {
            "title": "Multilingual Application (avg)",
            "metrics": ["Chinese", "English", "Japanese"],
            "llava": [None, None, None],
            "qwen":  [88.9, 87.2, 82.1],
            "note": "Qwen2.5-VL only (Comprehension Accuracy %)"
        },
    ]

    colors = {"llava": "#4C72B0", "qwen": "#DD8452"}

    for ax, d in zip(axes, datasets):
        x = np.arange(len(d["metrics"]))
        w = 0.35
        if d["llava"][0] is not None:
            ax.bar(x - w / 2, d["llava"], w, label="LLaVA-1.5", color=colors["llava"], alpha=0.85)
        ax.bar(x + (w / 2 if d["llava"][0] is not None else 0),
               d["qwen"], w, label="Qwen2.5-VL", color=colors["qwen"], alpha=0.85)
        ax.set_title(d["title"], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(d["metrics"], fontsize=8, rotation=15, ha="right")
        ax.set_ylabel("Accuracy (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/ch07_case_studies.png", dpi=120, bbox_inches="tight")
    print("Saved: assets/ch07_case_studies.png")
    plt.close()


# ── multimodal_applications ──────────────────────────────────────────────────

def plot_multimodal_applications():
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Multimodal Applications: LLaVA-1.5 vs Qwen2.5-VL Performance",
                 fontsize=13, fontweight="bold")

    datasets = [
        {
            "title": "Image Captioning",
            "metrics": ["BLEU-4", "METEOR", "CIDEr"],
            "llava": [35.2, 28.1, 112.3],
            "qwen":  [38.9, 31.5, 128.7],
            "ylabel": "Score",
            "scale": 1.0,
        },
        {
            "title": "Visual Question Answering",
            "metrics": ["VQA v2", "GQA", "TextVQA"],
            "llava": [82.1, 62.0, 58.3],
            "qwen":  [89.3, 70.5, 71.2],
            "ylabel": "Accuracy (%)",
            "scale": 1.0,
        },
        {
            "title": "Image Retrieval",
            "metrics": ["Flickr30K R@1", "Flickr30K R@5", "COCO R@1"],
            "llava": [68.2, 88.5, 58.1],
            "qwen":  [75.8, 93.2, 67.3],
            "ylabel": "Recall (%)",
            "scale": 1.0,
        },
    ]

    colors = {"llava": "#4C72B0", "qwen": "#DD8452"}

    for ax, d in zip(axes, datasets):
        x = np.arange(len(d["metrics"]))
        w = 0.35
        ax.bar(x - w / 2, d["llava"], w, label="LLaVA-1.5", color=colors["llava"], alpha=0.85)
        ax.bar(x + w / 2, d["qwen"], w, label="Qwen2.5-VL", color=colors["qwen"], alpha=0.85)
        ax.set_title(d["title"], fontsize=10, fontweight="bold")
        ax.set_xticks(x)
        ax.set_xticklabels(d["metrics"], fontsize=8, rotation=15, ha="right")
        ax.set_ylabel(d["ylabel"], fontsize=9)
        ymax = max(max(d["llava"]), max(d["qwen"])) * 1.15
        ax.set_ylim(0, ymax)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    plt.savefig("assets/ch07_multimodal_applications.png", dpi=120, bbox_inches="tight")
    print("Saved: assets/ch07_multimodal_applications.png")
    plt.close()


if __name__ == "__main__":
    plot_case_studies()
    plot_multimodal_applications()
