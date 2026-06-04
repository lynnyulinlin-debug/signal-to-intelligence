"""
实验7.3：高分辨率图像处理
对应章节：第7章 - 多模态LLM
目标：对比三种高分辨率图像处理方法（Patch分割、动态分辨率、渐进式处理）
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
IMAGE_SIZE = 1024
PATCH_SIZE = 16
STANDARD_SIZE = 224
OUTPUT_PATH = Path("assets/ch07_high_resolution_processing.png")


# ============ 核心逻辑 ============
def create_demo_image(image_size=IMAGE_SIZE, seed=42):
    rng = np.random.RandomState(seed)
    return rng.randn(image_size, image_size)


def patch_division_method(image, patch_size, standard_size):
    """将图像分割成patch，每个patch独立处理。"""
    h, w = image.shape
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    patches = []
    patch_positions = []

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = image[i * patch_size : (i + 1) * patch_size, j * patch_size : (j + 1) * patch_size]
            # 缩放patch到标准大小。这里保持原演示的简化平均池化逻辑。
            shrink = max(1, patch_size // 4)
            patch_resized = np.mean(
                patch.reshape(patch_size // shrink, shrink, patch_size // shrink, shrink),
                axis=(1, 3),
            )
            patches.append(patch_resized)
            patch_positions.append((i, j))

    return patches, patch_positions, (n_patches_h, n_patches_w)


def dynamic_resolution_method(image, target_tokens=576):
    """动态调整分辨率以保持token数量恒定。"""
    h, w = image.shape
    aspect_ratio = w / h

    patches_h = int(np.sqrt(target_tokens / aspect_ratio))
    patches_w = int(patches_h * aspect_ratio)

    patches_h = max(1, (patches_h // 16) * 16)
    patches_w = max(1, (patches_w // 16) * 16)
    resolution = (patches_h * 16, patches_w * 16)

    scale_h = resolution[0] / h
    scale_w = resolution[1] / w
    step_h = max(1, int(1 / scale_h))
    step_w = max(1, int(1 / scale_w))
    image_resized = image[::step_h, ::step_w]

    return image_resized, resolution, (patches_h, patches_w)


def progressive_processing_method(image, scales=None):
    """在多个尺度上处理图像。"""
    if scales is None:
        scales = [1.0, 0.5, 0.25]

    multi_scale_features = []

    for scale in scales:
        step = max(1, int(1 / scale))
        image_scaled = image[::step, ::step]

        feature = {
            "scale": scale,
            "size": image_scaled.shape,
            "mean": np.mean(image_scaled),
            "std": np.std(image_scaled),
            "patches": (
                image_scaled.shape[0] // PATCH_SIZE,
                image_scaled.shape[1] // PATCH_SIZE,
            ),
        }
        multi_scale_features.append(feature)

    return multi_scale_features


def run_experiment(image_size=IMAGE_SIZE, patch_size=PATCH_SIZE, standard_size=STANDARD_SIZE, seed=42):
    image = create_demo_image(image_size, seed=seed)

    patches_m1, _, grid_m1 = patch_division_method(image, patch_size, standard_size)
    tokens_m1 = len(patches_m1) + 1

    image_m2, resolution_m2, grid_m2 = dynamic_resolution_method(image)
    tokens_m2 = (grid_m2[0] * grid_m2[1]) + 1

    features_m3 = progressive_processing_method(image)
    tokens_m3 = sum([f["patches"][0] * f["patches"][1] for f in features_m3]) + 1

    methods_info = {
        "Patch Division": {
            "tokens": tokens_m1,
            "resolution": (image_size, image_size),
            "grid": grid_m1,
            "memory": tokens_m1 * 768,
            "latency": tokens_m1 * 0.1,
        },
        "Dynamic Resolution": {
            "tokens": tokens_m2,
            "resolution": resolution_m2,
            "grid": grid_m2,
            "memory": tokens_m2 * 768,
            "latency": tokens_m2 * 0.1,
        },
        "Progressive": {
            "tokens": tokens_m3,
            "resolution": image_size,
            "scales": len(features_m3),
            "memory": tokens_m3 * 768,
            "latency": tokens_m3 * 0.15,
        },
    }

    return {
        "image": image,
        "image_m2": image_m2,
        "grid_m1": grid_m1,
        "methods_info": methods_info,
        "image_size": image_size,
        "patch_size": patch_size,
    }


def print_summary(result):
    print("=" * 70)
    print("高分辨率图像处理方法对比")
    print("=" * 70)
    print(f"原始图像大小: {result['image_size']}x{result['image_size']}")
    print(f"Patch大小: {result['patch_size']}x{result['patch_size']}")
    print()

    for method_name, info in result["methods_info"].items():
        print(f"{method_name}:")
        print("-" * 70)
        print(f"  Token数量: {info['tokens']}")
        print(f"  分辨率: {info['resolution']}")
        if "grid" in info:
            print(f"  Grid: {info['grid'][0]}x{info['grid'][1]}")
        if "scales" in info:
            print(f"  处理尺度数: {info['scales']}")
        print(f"  内存占用: {info['memory']:.0f} (相对单位)")
        print(f"  相对延迟: {info['latency']:.1f} (相对单位)")
        print()

    print("=" * 70)


def plot_high_resolution_processing(result, output_path=OUTPUT_PATH):
    image = result["image"]
    image_m2 = result["image_m2"]
    methods_info = result["methods_info"]
    image_size = result["image_size"]
    patch_size = result["patch_size"]
    grid_m1 = result["grid_m1"]

    fig = plt.figure(figsize=(16, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    im = ax.imshow(image, cmap="gray")
    ax.set_title("Original Image\n(1024x1024)")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")
    plt.colorbar(im, ax=ax)

    ax = fig.add_subplot(gs[0, 1])
    for i in range(grid_m1[0]):
        for j in range(grid_m1[1]):
            rect = plt.Rectangle(
                (j * patch_size, i * patch_size),
                patch_size,
                patch_size,
                fill=False,
                edgecolor="red",
                linewidth=0.5,
                alpha=0.5,
            )
            ax.add_patch(rect)
    ax.imshow(image, cmap="gray", alpha=0.3)
    ax.set_title(f"Patch Division\n({grid_m1[0]}x{grid_m1[1]} patches)")
    ax.set_xlim(0, image_size)
    ax.set_ylim(image_size, 0)

    ax = fig.add_subplot(gs[0, 2])
    ax.imshow(image_m2, cmap="gray")
    ax.set_title(f"Dynamic Resolution\n({image_m2.shape[0]}x{image_m2.shape[1]})")
    ax.set_xlabel("Width")
    ax.set_ylabel("Height")

    methods = list(methods_info.keys())
    tokens = [methods_info[m]["tokens"] for m in methods]
    memory = [methods_info[m]["memory"] for m in methods]
    latency = [methods_info[m]["latency"] for m in methods]
    colors = ["steelblue", "coral", "lightgreen"]

    for subplot, values, ylabel, title in [
        (gs[1, 0], tokens, "Token Count", "Token Count Comparison"),
        (gs[1, 1], memory, "Memory (relative units)", "Memory Consumption"),
        (gs[1, 2], latency, "Latency (relative units)", "Processing Latency"),
    ]:
        ax = fig.add_subplot(subplot)
        bars = ax.bar(methods, values, color=colors, alpha=0.7, edgecolor="black")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.grid(True, alpha=0.3, axis="y")
        for bar, value in zip(bars, values):
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                bar.get_height(),
                f"{value:.1f}" if isinstance(value, float) else f"{int(value)}",
                ha="center",
                va="bottom",
            )

    ax = fig.add_subplot(gs[2, :], projection="polar")
    max_tokens = max(tokens)
    max_memory = max(memory)
    max_latency = max(latency)
    angles = np.linspace(0, 2 * np.pi, 3, endpoint=False).tolist()
    angles += angles[:1]

    for idx, method in enumerate(methods):
        values = [
            tokens[idx] / max_tokens,
            memory[idx] / max_memory,
            latency[idx] / max_latency,
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=method, color=colors[idx])
        ax.fill(angles, values, alpha=0.15, color=colors[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(["Tokens", "Memory", "Latency"])
    ax.set_ylim(0, 1)
    ax.set_title("Performance Characteristics (Normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_high_resolution_processing(result)
    print(f"图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
