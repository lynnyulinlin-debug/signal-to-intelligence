"""
Feature Extraction Comparison
Chapter 1: Digital Signal Processing

Compares traditional handcrafted features (PCA components)
with learned features (simulated CNN filters), illustrating
the shift from manual design to automatic learning.
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ============ Generate synthetic image patches ============
def make_patch(freq_x, freq_y, angle=0, size=16):
    x = np.linspace(0, 2 * np.pi, size)
    y = np.linspace(0, 2 * np.pi, size)
    X, Y = np.meshgrid(x, y)
    Xr = X * np.cos(angle) - Y * np.sin(angle)
    return np.sin(freq_x * Xr + freq_y * Y)

N_PATCHES = 200
PATCH_SIZE = 16
patches = []
for _ in range(N_PATCHES):
    fx = np.random.uniform(0.5, 3)
    fy = np.random.uniform(0.5, 3)
    angle = np.random.uniform(0, np.pi)
    p = make_patch(fx, fy, angle, PATCH_SIZE)
    p += 0.2 * np.random.randn(PATCH_SIZE, PATCH_SIZE)
    patches.append(p.flatten())

patches = np.array(patches)

# ============ PCA: traditional feature extraction ============
patches_centered = patches - patches.mean(axis=0)
cov = patches_centered.T @ patches_centered / N_PATCHES
eigenvalues, eigenvectors = np.linalg.eigh(cov)
idx = np.argsort(eigenvalues)[::-1]
eigenvalues = eigenvalues[idx]
eigenvectors = eigenvectors[:, idx]
pca_components = eigenvectors[:, :8].T.reshape(8, PATCH_SIZE, PATCH_SIZE)

# ============ Simulated CNN filters ============
def gabor_filter(size, freq, angle, sigma=2.5):
    x = np.linspace(-size // 2, size // 2, size)
    y = np.linspace(-size // 2, size // 2, size)
    X, Y = np.meshgrid(x, y)
    Xr = X * np.cos(angle) + Y * np.sin(angle)
    Yr = -X * np.sin(angle) + Y * np.cos(angle)
    gaussian = np.exp(-(Xr**2 + Yr**2) / (2 * sigma**2))
    sinusoid = np.cos(2 * np.pi * freq * Xr)
    return gaussian * sinusoid

cnn_filters = []
angles = np.linspace(0, np.pi, 8, endpoint=False)
for angle in angles:
    f = gabor_filter(PATCH_SIZE, freq=0.25, angle=angle)
    cnn_filters.append(f)

# ============ Plot ============
fig, axes = plt.subplots(3, 8, figsize=(16, 7))

# Row 0: PCA components
for i in range(8):
    ax = axes[0, i]
    comp = pca_components[i]
    vmax = np.abs(comp).max()
    ax.imshow(comp, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axis('off')
    if i == 0:
        ax.set_title('PCA\ncomponent', fontsize=8, loc='left', pad=2)
    var_pct = eigenvalues[i] / eigenvalues.sum() * 100
    ax.set_xlabel(f'{var_pct:.1f}%', fontsize=7)

# Row 1: CNN filters
for i in range(8):
    ax = axes[1, i]
    f = cnn_filters[i]
    vmax = np.abs(f).max()
    ax.imshow(f, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    ax.axis('off')
    if i == 0:
        ax.set_title('CNN filter\n(Gabor-like)', fontsize=8, loc='left', pad=2)
    ax.set_xlabel(f'{int(np.degrees(angles[i]))}°', fontsize=7)

# Row 2: Variance explained by PCA
ax = axes[2, 0]
ax.remove()
ax = fig.add_subplot(3, 1, 3)
cumvar = np.cumsum(eigenvalues) / eigenvalues.sum() * 100
ax.bar(range(1, 17), eigenvalues[:16] / eigenvalues.sum() * 100,
       color='steelblue', alpha=0.7, label='Individual variance')
ax.plot(range(1, 17), cumvar[:16], 'ro-', markersize=5, label='Cumulative variance')
ax.axhline(y=80, color='gray', linestyle='--', alpha=0.5, label='80% threshold')
ax.set_xlabel('Principal Component Index')
ax.set_ylabel('Variance Explained (%)')
ax.set_title('PCA: Variance Explained by Each Component', fontweight='bold')
ax.legend(fontsize=8)
ax.grid(True, alpha=0.3)

# Remove unused axes in row 2
for i in range(1, 8):
    axes[2, i].remove()

# Row labels
fig.text(0.01, 0.82, 'Handcrafted\n(PCA)', va='center', ha='left',
         fontsize=9, fontweight='bold', color='steelblue',
         bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.4))
fig.text(0.01, 0.60, 'Auto-learned\n(CNN)', va='center', ha='left',
         fontsize=9, fontweight='bold', color='tomato',
         bbox=dict(boxstyle='round', facecolor='mistyrose', alpha=0.4))

plt.suptitle(
    'Traditional Feature Extraction vs Learned Features\n'
    'PCA finds principal directions; CNN learns task-specific filters automatically',
    fontsize=12, fontweight='bold'
)
plt.tight_layout(rect=[0.06, 0, 1, 0.95])
plt.savefig('assets/ch01_feature_extraction.png', dpi=120, bbox_inches='tight')
print("Saved: assets/ch01_feature_extraction.png")
