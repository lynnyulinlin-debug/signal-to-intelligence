"""
7.2/7.3 架构说明图生成脚本

生成：
  assets/ch07_architecture_comparison.png   LLaVA vs Qwen2.5-VL 架构对比
  assets/ch07_fusion_strategies.png         三种融合策略示意图
  assets/ch07_dynamic_resolution.png        动态分辨率处理示意图
  assets/ch07_image_tiling.png              图像分块+重叠示意图

运行方式：
    python code/ch07_multimodal_llm/explainer_diagrams.py
"""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import numpy as np

# ── helpers ───────────────────────────────────────────────────────────────────

def box(ax, x, y, w, h, label, color, fontsize=9, label2=None):
    rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.04',
                                    facecolor=color, edgecolor='#555555', linewidth=1.2)
    ax.add_patch(rect)
    cy = y + h / 2 + (0.05 if label2 else 0)
    ax.text(x + w / 2, cy, label, ha='center', va='center',
            fontsize=fontsize, fontweight='bold', color='white')
    if label2:
        ax.text(x + w / 2, y + h / 2 - 0.1, label2, ha='center', va='center',
                fontsize=7, color='white')

def arrow(ax, x1, y1, x2, y2, color='#444444'):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle='->', color=color, lw=1.4))

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 1: LLaVA vs Qwen2.5-VL Architecture Comparison
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 7))
fig.suptitle('LLaVA vs Qwen2.5-VL Architecture', fontsize=14, fontweight='bold', y=0.97)

COLORS = {
    'input':  '#5c8ab8',
    'vit':    '#6aaa64',
    'proj':   '#e08b3a',
    'llm':    '#9b5fa5',
    'output': '#c0392b',
    'extra':  '#2980b9',
}

# ── Left: LLaVA ───────────────────────────────────────────────────────────────
ax = axes[0]
ax.set_xlim(0, 4); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('LLaVA', fontsize=12, fontweight='bold', color='#2c3e50', pad=8)

items_l = [
    (1.2, 7.2, 1.6, 0.55, 'Image Input\n(any size)',  COLORS['input'],  8),
    (1.2, 6.2, 1.6, 0.55, 'Resize → 336×336',         '#607d8b',        8),
    (1.2, 5.0, 1.6, 0.65, 'ViT Encoder\n(CLIP ViT-L)', COLORS['vit'],    9),
    (1.2, 3.9, 1.6, 0.55, 'Single-layer MLP',          COLORS['proj'],   9),
    (1.2, 2.6, 1.6, 0.65, 'LLM\n(LLaMA 7B)',           COLORS['llm'],    9),
    (1.2, 1.5, 1.6, 0.55, 'Text Output',                COLORS['output'], 9),
]
for x, y, w, h, lbl, c, fs in items_l:
    box(ax, x, y, w, h, lbl, c, fs)

for i in range(len(items_l) - 1):
    _, y1, _, h1, _, _, _ = items_l[i]
    _, y2, _, _, _, _, _ = items_l[i + 1]
    arrow(ax, 2.0, y1, 2.0, y2 + items_l[i+1][3])

# Text input branch
box(ax, 0.05, 5.0, 1.0, 0.55, 'Text\nInput', COLORS['input'], 8)
arrow(ax, 0.55, 5.0, 0.55, 3.2)
ax.annotate('', xy=(1.2, 2.925), xytext=(0.55, 2.925),
            arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))

ax.text(0.1, 0.3, 'Fixed 336×336 resolution\nSingle image only\nLate fusion',
        fontsize=7.5, color='#c0392b',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#fdecea', edgecolor='#e74c3c', alpha=0.9))

# ── Right: Qwen2.5-VL ─────────────────────────────────────────────────────────
ax = axes[1]
ax.set_xlim(0, 5); ax.set_ylim(0, 8.5); ax.axis('off')
ax.set_title('Qwen2.5-VL', fontsize=12, fontweight='bold', color='#2c3e50', pad=8)

# Multiple image inputs
for xi, lbl in [(0.2, 'Img 1'), (1.05, 'Img 2'), (1.9, 'Img N')]:
    box(ax, xi, 7.3, 0.75, 0.45, lbl, COLORS['input'], 8)
    arrow(ax, xi + 0.375, 7.3, xi + 0.375, 6.85)

ax.text(1.325, 7.15, '···', fontsize=14, ha='center', va='center', color='#555')

# Dynamic resolution
box(ax, 0.1, 6.15, 2.4, 0.6, 'Dynamic Resolution', '#607d8b', 9,
    '(448–1024px, keep aspect ratio)')

# ViT with HR note
box(ax, 0.1, 5.05, 2.4, 0.65, 'ViT Encoder (ViT-L)', COLORS['vit'], 9,
    '+ tiling for large images')

# Multi-layer projection
box(ax, 0.1, 3.95, 2.4, 0.65, '2-layer MLP Projection', COLORS['proj'], 9,
    'bottleneck: 768 → 2048 → 3584')

# Text input
box(ax, 2.65, 5.05, 1.5, 0.65, 'Text Input\n+ System Prompt', COLORS['input'], 8)

# LLM
box(ax, 0.1, 2.65, 4.05, 0.75, 'Qwen2 LLM (7B / 32B)', COLORS['llm'], 10,
    'context: up to 32768 tokens  |  hybrid fusion')

# Output
box(ax, 0.1, 1.5, 4.05, 0.6, 'Text Output', COLORS['output'], 9)

# Arrows in right panel
arrow(ax, 1.3, 6.15, 1.3, 5.7)
arrow(ax, 1.3, 5.05, 1.3, 4.6)
arrow(ax, 1.3, 3.95, 1.3, 3.4)
arrow(ax, 3.4, 5.05, 3.4, 3.4)
ax.annotate('', xy=(4.15, 3.4), xytext=(3.4, 3.4),
            arrowprops=dict(arrowstyle='->', color='#444', lw=1.2))
arrow(ax, 2.125, 2.65, 2.125, 2.1)

ax.text(0.05, 0.3, '✓ 1024×1024 resolution\n✓ Multi-image support\n✓ Hybrid fusion',
        fontsize=7.5, color='#1a7a1a',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='#eafaea', edgecolor='#2ecc71', alpha=0.9))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assets/ch07_architecture_comparison.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved assets/ch07_architecture_comparison.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 2: Three Fusion Strategies
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 3, figsize=(14, 5.5))
fig.suptitle('Multimodal Fusion Strategies', fontsize=13, fontweight='bold', y=0.98)

FCOLORS = {
    'img': '#5c8ab8', 'txt': '#e08b3a', 'fused': '#9b5fa5',
    'out': '#c0392b', 'enc': '#6aaa64', 'cross': '#2980b9',
}

panels = [
    ('Early Fusion', '#fdecea', '#e74c3c',
     'High compute cost\nFull cross-modal interaction early',
     '#c0392b'),
    ('Late Fusion', '#eafaea', '#2ecc71',
     'Efficient\nLimited cross-modal interaction',
     '#1a7a1a'),
    ('Hybrid Fusion  (Qwen2.5-VL)', '#eaf2fb', '#3498db',
     'Balanced: multi-level interaction\nBest accuracy / cost trade-off',
     '#154360'),
]

for ax, (title, bg, bord, note, nc) in zip(axes, panels):
    ax.set_xlim(0, 3); ax.set_ylim(0, 8); ax.axis('off')
    ax.set_facecolor(bg)
    ax.set_title(title, fontsize=10.5, fontweight='bold', pad=6)
    ax.add_patch(mpatches.FancyBboxPatch((0, 0), 3, 8, boxstyle='square,pad=0',
                 facecolor=bg, edgecolor=bord, linewidth=2))

# ── Early Fusion ──────────────────────────────────────────────────────────────
ax = axes[0]
box(ax, 0.2, 6.8, 0.9, 0.6, 'Image', FCOLORS['img'], 9)
box(ax, 1.9, 6.8, 0.9, 0.6, 'Text', FCOLORS['txt'], 9)
arrow(ax, 0.65, 6.8, 1.0, 6.05)
arrow(ax, 2.35, 6.8, 2.0, 6.05)
box(ax, 0.7, 5.35, 1.6, 0.6, 'Concat + Joint\nSelf-Attention', FCOLORS['fused'], 8)
arrow(ax, 1.5, 5.35, 1.5, 4.75)
box(ax, 0.7, 4.1, 1.6, 0.6, 'Joint Encoding\n(high dim)', FCOLORS['fused'], 8)
arrow(ax, 1.5, 4.1, 1.5, 3.5)
box(ax, 0.7, 2.85, 1.6, 0.6, 'LLM Decoder', FCOLORS['out'], 9)
arrow(ax, 1.5, 2.85, 1.5, 2.25)
box(ax, 0.7, 1.6, 1.6, 0.55, 'Output', FCOLORS['out'], 9)
ax.text(0.1, 0.2, note, fontsize=7.5, color=nc,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
ax.text(1.5, 4.95, 'O((N_img+N_txt)²·d)', ha='center', fontsize=7, color='#c0392b', style='italic')

# ── Late Fusion ───────────────────────────────────────────────────────────────
ax = axes[1]
box(ax, 0.1, 6.8, 1.1, 0.6, 'Image', FCOLORS['img'], 9)
box(ax, 1.8, 6.8, 1.1, 0.6, 'Text', FCOLORS['txt'], 9)
arrow(ax, 0.65, 6.8, 0.65, 6.2)
arrow(ax, 2.35, 6.8, 2.35, 6.2)
box(ax, 0.1, 5.5, 1.1, 0.6, 'Image\nEncoder', FCOLORS['enc'], 8)
box(ax, 1.8, 5.5, 1.1, 0.6, 'Text\nEncoder', FCOLORS['enc'], 8)
arrow(ax, 0.65, 5.5, 0.65, 4.9)
arrow(ax, 2.35, 5.5, 2.35, 4.9)
box(ax, 0.1, 4.2, 1.1, 0.6, 'Img Feat', FCOLORS['img'], 9)
box(ax, 1.8, 4.2, 1.1, 0.6, 'Txt Feat', FCOLORS['txt'], 9)
arrow(ax, 0.65, 4.2, 1.2, 3.6)
arrow(ax, 2.35, 4.2, 1.8, 3.6)
box(ax, 0.7, 2.9, 1.6, 0.6, 'Late Concat\n/ Projection', FCOLORS['fused'], 8)
arrow(ax, 1.5, 2.9, 1.5, 2.3)
box(ax, 0.7, 1.6, 1.6, 0.55, 'Output', FCOLORS['out'], 9)
ax.text(0.1, 0.2, note, fontsize=7.5, color=nc,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))
ax.text(1.5, 3.55, 'O(N_img²) + O(N_txt²)', ha='center', fontsize=7, color='#1a7a1a', style='italic')

# ── Hybrid Fusion ─────────────────────────────────────────────────────────────
ax = axes[2]
box(ax, 0.1, 7.1, 0.95, 0.55, 'Image', FCOLORS['img'], 9)
box(ax, 1.95, 7.1, 0.95, 0.55, 'Text', FCOLORS['txt'], 9)
arrow(ax, 0.575, 7.1, 0.575, 6.55)
arrow(ax, 2.425, 7.1, 2.425, 6.55)
box(ax, 0.1, 5.9, 0.95, 0.55, 'ViT\nEncoder', FCOLORS['enc'], 8)
box(ax, 1.95, 5.9, 0.95, 0.55, 'Text\nEncoder', FCOLORS['enc'], 8)
# Cross-attention level 1
arrow(ax, 0.575, 5.9, 0.575, 5.35)
box(ax, 0.1, 4.75, 2.8, 0.5, 'Cross-Attention Layer 1  (visual ↔ text)', FCOLORS['cross'], 8)
arrow(ax, 1.5, 4.75, 1.5, 4.2)
box(ax, 0.1, 3.55, 2.8, 0.5, 'Cross-Attention Layer 2  (deeper fusion)', FCOLORS['fused'], 8)
arrow(ax, 1.5, 3.55, 1.5, 3.0)
box(ax, 0.1, 2.3, 2.8, 0.6, 'LLM Layers (final fusion)', FCOLORS['out'], 9)
arrow(ax, 1.5, 2.3, 1.5, 1.75)
box(ax, 0.1, 1.05, 2.8, 0.55, 'Output', FCOLORS['out'], 9)
ax.text(2.425, 5.35, '→', ha='center', va='center', fontsize=14, color=FCOLORS['cross'])
ax.text(0.1, 0.1, note, fontsize=7.5, color=nc,
        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.8))

plt.tight_layout(rect=[0, 0, 1, 0.95])
plt.savefig('assets/ch07_fusion_strategies.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved assets/ch07_fusion_strategies.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 3: Dynamic Resolution Processing
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(2, 3, figsize=(13, 6))
fig.suptitle('Dynamic Resolution Processing', fontsize=13, fontweight='bold', y=0.99)

cases = [
    ('Wide image\n2048×1024', 2048, 1024, '#5c8ab8', 'Resize to 1024×512\n(keep aspect ratio)'),
    ('Square image\n1024×1024', 1024, 1024, '#6aaa64', 'Keep 1024×1024\n(no resize needed)'),
    ('Tall image\n512×2048', 512, 2048, '#e08b3a', 'Resize to 512×1024\n→ short edge ≤ 1024'),
]

max_dim = 2048
for col, (title, w, h, color, note) in enumerate(cases):
    # Top row: input image
    ax_in = axes[0][col]
    ax_in.set_xlim(0, max_dim); ax_in.set_ylim(0, max_dim)
    ax_in.set_aspect('equal')
    ax_in.axis('off')
    ax_in.set_title(f'Input: {title}', fontsize=9, fontweight='bold', color='#333')

    # Draw original image rectangle centered
    ox = (max_dim - w) / 2; oy = (max_dim - h) / 2
    rect = mpatches.FancyBboxPatch((ox, oy), w, h, boxstyle='round,pad=30',
                                    facecolor=color, edgecolor='#333', linewidth=2, alpha=0.85)
    ax_in.add_patch(rect)
    ax_in.text(max_dim/2, max_dim/2, f'{w}×{h}', ha='center', va='center',
               fontsize=11, fontweight='bold', color='white')
    # Dimension labels
    ax_in.annotate('', xy=(ox+w, oy-80), xytext=(ox, oy-80),
                   arrowprops=dict(arrowstyle='<->', color='#333', lw=1.2))
    ax_in.text(max_dim/2, oy-180, f'w={w}', ha='center', fontsize=8, color='#333')

    # Bottom row: output
    ax_out = axes[1][col]
    ax_out.set_xlim(0, max_dim); ax_out.set_ylim(0, max_dim)
    ax_out.set_aspect('equal')
    ax_out.axis('off')
    ax_out.set_title(f'Output: {note}', fontsize=8.5, color='#1a5276')

    # Compute output dims
    if w > h:
        ow, oh = 1024, int(1024 * h / w)
    elif h > w:
        oh, ow = 1024, int(1024 * w / h)
    else:
        ow = oh = 1024

    ox2 = (max_dim - ow) / 2; oy2 = (max_dim - oh) / 2
    rect2 = mpatches.FancyBboxPatch((ox2, oy2), ow, oh, boxstyle='round,pad=30',
                                     facecolor=color, edgecolor='#1a5276', linewidth=2.5, alpha=0.95)
    ax_out.add_patch(rect2)
    ax_out.text(max_dim/2, max_dim/2, f'{ow}×{oh}', ha='center', va='center',
                fontsize=11, fontweight='bold', color='white')

plt.subplots_adjust(hspace=0.35)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('assets/ch07_dynamic_resolution.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved assets/ch07_dynamic_resolution.png")

# ═══════════════════════════════════════════════════════════════════════════════
# Figure 4: Image Tiling with Overlap
# ═══════════════════════════════════════════════════════════════════════════════

fig, axes = plt.subplots(1, 2, figsize=(13, 6))
fig.suptitle('Image Tiling Strategy (Tile Size=512, Stride=256, Overlap=50%)',
             fontsize=12, fontweight='bold')

IMG = 2048
TILE = 512
STRIDE = 256
n = (IMG - TILE) // STRIDE + 1   # = 7

cmap_tile = plt.cm.tab20

# Left: tiling grid
ax = axes[0]
ax.set_xlim(-50, IMG+100); ax.set_ylim(-150, IMG+50)
ax.set_aspect('equal'); ax.axis('off')
ax.set_title(f'2048×2048 Image → {n}×{n}={n*n} Tiles\n(tile 512×512, stride 256, overlap 50%)',
             fontsize=9, pad=8)

# Draw image boundary
img_rect = mpatches.Rectangle((0, 0), IMG, IMG, linewidth=2, edgecolor='#333',
                                facecolor='#f5f5f5')
ax.add_patch(img_rect)

# Draw all tiles (show first few for clarity)
for ri in range(n):
    for ci in range(n):
        x0 = ci * STRIDE
        y0 = ri * STRIDE
        color_idx = (ri * n + ci) % 20
        alpha = 0.25
        tile_rect = mpatches.Rectangle((x0, y0), TILE, TILE,
                                        linewidth=0.8,
                                        edgecolor=cmap_tile(color_idx),
                                        facecolor=cmap_tile(color_idx),
                                        alpha=alpha)
        ax.add_patch(tile_rect)

# Highlight first 2 tiles to show overlap
tile0 = mpatches.Rectangle((0, 0), TILE, TILE, linewidth=2.5,
                             edgecolor='#e74c3c', facecolor='#e74c3c', alpha=0.15)
tile1 = mpatches.Rectangle((STRIDE, 0), TILE, TILE, linewidth=2.5,
                             edgecolor='#2980b9', facecolor='#2980b9', alpha=0.15)
ax.add_patch(tile0); ax.add_patch(tile1)
ax.text(TILE/2, -80, 'Tile 1', ha='center', fontsize=8.5, color='#c0392b', fontweight='bold')
ax.text(STRIDE + TILE/2, -80, 'Tile 2', ha='center', fontsize=8.5, color='#1a5276', fontweight='bold')

# Draw overlap region
overlap_rect = mpatches.Rectangle((STRIDE, 0), TILE-STRIDE, TILE,
                                    linewidth=2, edgecolor='#8e44ad',
                                    facecolor='#8e44ad', alpha=0.35)
ax.add_patch(overlap_rect)
ax.text(STRIDE + (TILE-STRIDE)/2, TILE/2, 'Overlap\n(50%)', ha='center', va='center',
        fontsize=7.5, color='white', fontweight='bold')

# Dimension annotations
ax.annotate('', xy=(TILE, -40), xytext=(0, -40),
            arrowprops=dict(arrowstyle='<->', color='#c0392b', lw=1.5))
ax.text(TILE/2, -100, '512px (tile)', ha='center', fontsize=8, color='#c0392b')
ax.annotate('', xy=(STRIDE*2, -120), xytext=(0, -120),
            arrowprops=dict(arrowstyle='<->', color='#555', lw=1.2))
ax.text(STRIDE, -145, f'stride={STRIDE}', ha='center', fontsize=8, color='#555')

# Full grid lines
for i in range(n+1):
    ax.plot([i*STRIDE, i*STRIDE], [0, IMG], color='gray', linewidth=0.4, alpha=0.5)
    ax.plot([0, IMG], [i*STRIDE, i*STRIDE], color='gray', linewidth=0.4, alpha=0.5)

ax.text(IMG/2, IMG+30, f'Total tiles: {n}×{n} = {n*n}', ha='center',
        fontsize=9, color='#1a5276', fontweight='bold')

# Right: tile count formula
ax2 = axes[1]
ax2.set_xlim(0, 5); ax2.set_ylim(0, 8); ax2.axis('off')
ax2.set_title('Tile Count Formula', fontsize=10, fontweight='bold', pad=8)

lines_txt = [
    (4.2, 'Formula:', 11, 'bold', '#2c3e50'),
    (3.6, 'n = ⌈(W − tile) / stride⌉ + 1', 10, 'normal', '#2980b9'),
    (3.0, f'= ⌈(2048 − 512) / 256⌉ + 1', 10, 'normal', '#555'),
    (2.4, f'= ⌈6⌉ + 1 = 7  per side', 10, 'normal', '#555'),
    (1.7, f'Total = 7 × 7 = 49 tiles', 11, 'bold', '#c0392b'),
    (1.0, '[X]  2048/256 = 64  (wrong -- ignores\n    tile size vs stride difference)',
     8.5, 'normal', '#c0392b'),
]
for y, txt, fs, fw, color in lines_txt:
    ax2.text(0.3, y, txt, fontsize=fs, fontweight=fw, color=color, va='center')

# Info boxes
info = [
    (5.5, '#eafaea', '#2ecc71', 'Overlap benefits',
     '• Boundary info captured by 2 tiles\n• No hard edge artifacts\n• Smooth stitching'),
    (3.8, '#fdecea', '#e74c3c', 'Compute cost',
     f'• {n*n} tiles × ViT forward pass\n• ~{n*n}× slower than single image\n• Trade-off: detail vs speed'),
]
for y, fc, ec, title, body in info:
    ax2.add_patch(mpatches.FancyBboxPatch((0.1, y-0.8), 4.8, 1.5,
                  boxstyle='round,pad=0.1', facecolor=fc, edgecolor=ec, linewidth=1.5))
    ax2.text(0.3, y+0.45, title, fontsize=9, fontweight='bold', color=ec)
    ax2.text(0.3, y-0.45, body, fontsize=8, color='#333', va='center')

plt.tight_layout(rect=[0, 0, 1, 0.94])
plt.savefig('assets/ch07_image_tiling.png', dpi=100, bbox_inches='tight')
plt.close()
print("Saved assets/ch07_image_tiling.png")

print("\nAll 4 diagrams generated successfully.")
