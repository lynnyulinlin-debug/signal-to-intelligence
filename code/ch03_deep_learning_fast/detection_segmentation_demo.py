"""
实验3.4：目标检测与图像分割演示
对应章节：第3章 - 深度学习快速通道
目标：对比目标检测与图像分割的输出形式和空间理解粒度
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False
from matplotlib.patches import Rectangle

np.random.seed(42)

# 创建一个简单的合成图像
H, W = 256, 256
image = np.ones((H, W, 3), dtype=float)
image[:] = [0.95, 0.95, 0.95]

# 目标1：红色矩形
obj1 = (40, 60, 90, 70)  # x, y, w, h
image[obj1[1]:obj1[1]+obj1[3], obj1[0]:obj1[0]+obj1[2], :] = [0.9, 0.3, 0.3]

# 目标2：蓝色圆形近似
cy, cx, r = 170, 170, 35
Y, X = np.ogrid[:H, :W]
mask_circle = (X - cx) ** 2 + (Y - cy) ** 2 <= r ** 2
image[mask_circle] = [0.3, 0.4, 0.9]

# 检测框
boxes = [
    {"label": "rectangle", "bbox": obj1, "color": "red", "score": 0.93},
    {"label": "circle", "bbox": (cx-r, cy-r, 2*r, 2*r), "color": "blue", "score": 0.89},
]

# 分割mask
seg_mask = np.zeros((H, W), dtype=int)
seg_mask[obj1[1]:obj1[1]+obj1[3], obj1[0]:obj1[0]+obj1[2]] = 1
seg_mask[mask_circle] = 2

# 指标演示
num_objects = len(boxes)
covered_pixels = np.sum(seg_mask > 0)
coverage_ratio = covered_pixels / (H * W)

print("=" * 70)
print("Object Detection and Segmentation Demo")
print("=" * 70)
print(f"Image size: {H} x {W}")
print(f"Number of detected objects: {num_objects}")
print(f"Foreground pixel ratio: {coverage_ratio:.2%}")
print()
print("Detection output:")
for item in boxes:
    x, y, w, h = item["bbox"]
    print(f"- {item['label']}: bbox=({x}, {y}, {w}, {h}), score={item['score']:.2f}")
print()
print("Segmentation output:")
print("- 0: background")
print("- 1: rectangle")
print("- 2: circle")
print("=" * 70)

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# 1. 原图
ax = axes[0]
ax.imshow(image)
ax.set_title("Synthetic Scene")
ax.axis("off")

# 2. 检测结果
ax = axes[1]
ax.imshow(image)
for item in boxes:
    x, y, w, h = item["bbox"]
    rect = Rectangle((x, y), w, h, linewidth=2, edgecolor=item["color"], facecolor="none")
    ax.add_patch(rect)
    ax.text(x, y - 5, f"{item['label']} {item['score']:.2f}", color=item["color"], fontsize=9,
            bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"))
ax.set_title("Object Detection")
ax.axis("off")

# 3. 分割结果
ax = axes[2]
cmap = plt.get_cmap("Set1", 3)
ax.imshow(seg_mask, cmap=cmap, vmin=0, vmax=2)
ax.set_title("Image Segmentation")
ax.axis("off")

plt.tight_layout()
plt.savefig("assets/ch03_yolo_vs_segmentation.png", dpi=120, bbox_inches="tight")
print("Figure saved to: assets/ch03_yolo_vs_segmentation.png")

fig, axes = plt.subplots(1, 3, figsize=(15, 5))

ax = axes[0]
ax.imshow(image)
ax.set_title("Input Image")
ax.axis("off")

ax = axes[1]
ax.imshow(seg_mask, cmap=cmap, vmin=0, vmax=2)
ax.set_title("Semantic Mask")
ax.axis("off")

ax = axes[2]
ax.imshow(image)
seg_colors = np.zeros((H, W, 4))
seg_colors[seg_mask == 1] = [0.95, 0.2, 0.2, 0.45]
seg_colors[seg_mask == 2] = [0.2, 0.35, 0.95, 0.45]
ax.imshow(seg_colors)
ax.contour(seg_mask == 1, levels=[0.5], colors=["darkred"], linewidths=1.5)
ax.contour(seg_mask == 2, levels=[0.5], colors=["navy"], linewidths=1.5)
ax.set_title("Overlay with Boundaries")
ax.axis("off")

plt.tight_layout()
plt.savefig("assets/ch03_segmentation_mask_overlay.png", dpi=120, bbox_inches="tight")
print("Figure saved to: assets/ch03_segmentation_mask_overlay.png")
