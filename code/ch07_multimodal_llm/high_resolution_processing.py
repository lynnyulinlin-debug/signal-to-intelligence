"""
实验7.3：高分辨率图像处理
对应章节：第7章 - 多模态LLM
目标：对比三种高分辨率图像处理方法（Patch分割、动态分辨率、渐进式处理）
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
np.random.seed(42)
IMAGE_SIZE = 1024  # 高分辨率图像大小
PATCH_SIZE = 16    # 基础patch大小
STANDARD_SIZE = 224  # 标准分辨率

# ============ 核心逻辑 ============
# 生成模拟高分辨率图像（简化为2D）
image = np.random.randn(IMAGE_SIZE, IMAGE_SIZE)

# 方法1：Patch分割（LLaVA方式）
# 将高分辨率图像分割成多个patch，每个patch缩放到标准大小
def patch_division_method(image, patch_size, standard_size):
    """
    将图像分割成patch，每个patch独立处理
    """
    h, w = image.shape
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size

    patches = []
    patch_positions = []

    for i in range(n_patches_h):
        for j in range(n_patches_w):
            patch = image[i*patch_size:(i+1)*patch_size,
                         j*patch_size:(j+1)*patch_size]
            # 缩放patch到标准大小
            patch_resized = np.mean(patch.reshape(patch_size//4, 4, patch_size//4, 4),
                                   axis=(1, 3))
            patches.append(patch_resized)
            patch_positions.append((i, j))

    return patches, patch_positions, (n_patches_h, n_patches_w)

# 方法2：动态分辨率（Qwen VL方式）
# 根据内容自适应调整分辨率
def dynamic_resolution_method(image, target_tokens=576):
    """
    动态调整分辨率以保持token数量恒定
    """
    h, w = image.shape
    aspect_ratio = w / h

    # 计算最优分辨率
    # 假设每个patch是16x16，目标token数是576
    total_patches = target_tokens
    patches_h = int(np.sqrt(total_patches / aspect_ratio))
    patches_w = int(patches_h * aspect_ratio)

    # 调整到16的倍数
    patches_h = (patches_h // 16) * 16
    patches_w = (patches_w // 16) * 16

    resolution = (patches_h * 16, patches_w * 16)

    # 缩放图像到动态分辨率
    scale_h = resolution[0] / h
    scale_w = resolution[1] / w

    # 简化的缩放（实际使用插值）
    image_resized = image[::int(1/scale_h), ::int(1/scale_w)]

    return image_resized, resolution, (patches_h, patches_w)

# 方法3：渐进式处理（多尺度）
# 在多个分辨率上处理图像，然后融合
def progressive_processing_method(image, scales=[1.0, 0.5, 0.25]):
    """
    在多个尺度上处理图像
    """
    multi_scale_features = []

    for scale in scales:
        # 缩放图像
        new_size = int(IMAGE_SIZE * scale)
        # 简化的缩放
        step = int(1 / scale)
        image_scaled = image[::step, ::step]

        # 计算特征（这里用简单的统计特征）
        feature = {
            'scale': scale,
            'size': image_scaled.shape,
            'mean': np.mean(image_scaled),
            'std': np.std(image_scaled),
            'patches': (image_scaled.shape[0] // PATCH_SIZE,
                       image_scaled.shape[1] // PATCH_SIZE)
        }
        multi_scale_features.append(feature)

    return multi_scale_features

# ============ 执行三种方法 ============
# 方法1：Patch分割
patches_m1, positions_m1, grid_m1 = patch_division_method(image, PATCH_SIZE, STANDARD_SIZE)
n_patches_m1 = len(patches_m1)
tokens_m1 = n_patches_m1 + 1  # +1 for class token

# 方法2：动态分辨率
image_m2, resolution_m2, grid_m2 = dynamic_resolution_method(image)
tokens_m2 = (grid_m2[0] * grid_m2[1]) + 1

# 方法3：渐进式处理
features_m3 = progressive_processing_method(image)
tokens_m3 = sum([f['patches'][0] * f['patches'][1] for f in features_m3]) + 1

# ============ 性能分析 ============
# 计算各方法的特性
methods_info = {
    'Patch Division': {
        'tokens': tokens_m1,
        'resolution': (IMAGE_SIZE, IMAGE_SIZE),
        'grid': grid_m1,
        'memory': tokens_m1 * 768,  # 假设每个token 768维
        'latency': tokens_m1 * 0.1,  # 相对延迟
    },
    'Dynamic Resolution': {
        'tokens': tokens_m2,
        'resolution': resolution_m2,
        'grid': grid_m2,
        'memory': tokens_m2 * 768,
        'latency': tokens_m2 * 0.1,
    },
    'Progressive': {
        'tokens': tokens_m3,
        'resolution': IMAGE_SIZE,
        'scales': len(features_m3),
        'memory': tokens_m3 * 768,
        'latency': tokens_m3 * 0.15,  # 多尺度处理更慢
    }
}

# ============ 结果输出 ============
print("=" * 70)
print("高分辨率图像处理方法对比")
print("=" * 70)
print(f"原始图像大小: {IMAGE_SIZE}x{IMAGE_SIZE}")
print(f"Patch大小: {PATCH_SIZE}x{PATCH_SIZE}")
print()

for method_name, info in methods_info.items():
    print(f"{method_name}:")
    print("-" * 70)
    print(f"  Token数量: {info['tokens']}")
    print(f"  分辨率: {info['resolution']}")
    if 'grid' in info:
        print(f"  Grid: {info['grid'][0]}x{info['grid'][1]}")
    if 'scales' in info:
        print(f"  处理尺度数: {info['scales']}")
    print(f"  内存占用: {info['memory']:.0f} (相对单位)")
    print(f"  相对延迟: {info['latency']:.1f} (相对单位)")
    print()

print("=" * 70)

# ============ 可视化 ============
fig = plt.figure(figsize=(16, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. 原始图像
ax = fig.add_subplot(gs[0, 0])
im = ax.imshow(image, cmap='gray')
ax.set_title('Original Image\n(1024x1024)')
ax.set_xlabel('Width')
ax.set_ylabel('Height')
plt.colorbar(im, ax=ax)

# 2. 方法1：Patch分割可视化
ax = fig.add_subplot(gs[0, 1])
# 绘制patch网格
for i in range(grid_m1[0]):
    for j in range(grid_m1[1]):
        rect = plt.Rectangle((j*PATCH_SIZE, i*PATCH_SIZE), PATCH_SIZE, PATCH_SIZE,
                            fill=False, edgecolor='red', linewidth=0.5, alpha=0.5)
        ax.add_patch(rect)
ax.imshow(image, cmap='gray', alpha=0.3)
ax.set_title(f'Patch Division\n({grid_m1[0]}x{grid_m1[1]} patches)')
ax.set_xlim(0, IMAGE_SIZE)
ax.set_ylim(IMAGE_SIZE, 0)

# 3. 方法2：动态分辨率
ax = fig.add_subplot(gs[0, 2])
ax.imshow(image_m2, cmap='gray')
ax.set_title(f'Dynamic Resolution\n({resolution_m2[0]}x{resolution_m2[1]})')
ax.set_xlabel('Width')
ax.set_ylabel('Height')

# 4. Token数量对比
ax = fig.add_subplot(gs[1, 0])
methods = list(methods_info.keys())
tokens = [methods_info[m]['tokens'] for m in methods]
colors = ['steelblue', 'coral', 'lightgreen']
bars = ax.bar(methods, tokens, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Token Count')
ax.set_title('Token Count Comparison')
ax.grid(True, alpha=0.3, axis='y')
for bar, token in zip(bars, tokens):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(token)}', ha='center', va='bottom')

# 5. 内存占用对比
ax = fig.add_subplot(gs[1, 1])
memory = [methods_info[m]['memory'] for m in methods]
bars = ax.bar(methods, memory, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Memory (relative units)')
ax.set_title('Memory Consumption')
ax.grid(True, alpha=0.3, axis='y')
for bar, mem in zip(bars, memory):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(mem)}', ha='center', va='bottom')

# 6. 延迟对比
ax = fig.add_subplot(gs[1, 2])
latency = [methods_info[m]['latency'] for m in methods]
bars = ax.bar(methods, latency, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Latency (relative units)')
ax.set_title('Processing Latency')
ax.grid(True, alpha=0.3, axis='y')
for bar, lat in zip(bars, latency):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{lat:.1f}', ha='center', va='bottom')

# 7. 方法特性雷达图数据
ax = fig.add_subplot(gs[2, :], projection='polar')

# 归一化指标
max_tokens = max(tokens)
max_memory = max(memory)
max_latency = max(latency)

angles = np.linspace(0, 2*np.pi, 3, endpoint=False).tolist()
angles += angles[:1]

for idx, method in enumerate(methods):
    values = [
        tokens[idx] / max_tokens,
        memory[idx] / max_memory,
        latency[idx] / max_latency,
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=method, color=colors[idx])
    ax.fill(angles, values, alpha=0.15, color=colors[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(['Tokens', 'Memory', 'Latency'])
ax.set_ylim(0, 1)
ax.set_title('Performance Characteristics (Normalized)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.savefig('assets/ch07_high_resolution_processing.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch07_high_resolution_processing.png")
