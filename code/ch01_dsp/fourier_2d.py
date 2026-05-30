"""
2D Fourier Transform Visualization
Chapter 1: Digital Signal Processing

Demonstrates the extension from 1D to 2D Fourier analysis:
1. 1D signal and its frequency spectrum
2. 2D image and its spatial frequency spectrum
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ============ 1D Signal and Spectrum ============
fs = 512
t = np.linspace(0, 1, fs, endpoint=False)
signal_1d = (
    np.sin(2 * np.pi * 5 * t)
    + 0.6 * np.sin(2 * np.pi * 20 * t)
    + 0.3 * np.sin(2 * np.pi * 50 * t)
)
spectrum_1d = np.abs(np.fft.rfft(signal_1d))
freqs_1d = np.fft.rfftfreq(fs, 1 / fs)

# ============ 2D Image and Spatial Spectrum ============
N = 128
x = np.linspace(0, 1, N, endpoint=False)
y = np.linspace(0, 1, N, endpoint=False)
X, Y = np.meshgrid(x, y)

# Structured image: horizontal + diagonal stripes
image_2d = (
    np.sin(2 * np.pi * 4 * X)          # horizontal spatial freq
    + 0.7 * np.sin(2 * np.pi * 8 * Y)  # vertical spatial freq
    + 0.4 * np.sin(2 * np.pi * 4 * (X + Y))  # diagonal
    + 0.1 * np.random.randn(N, N)
)

fft2d = np.fft.fft2(image_2d)
spectrum_2d = np.abs(np.fft.fftshift(fft2d))
spectrum_2d_log = np.log1p(spectrum_2d)

# ============ Plot ============
fig, axes = plt.subplots(2, 2, figsize=(13, 9))

# 1D signal
ax = axes[0, 0]
ax.plot(t, signal_1d, color='steelblue', linewidth=1.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('1D Signal  x[n]', fontweight='bold')
ax.grid(True, alpha=0.3)

# 1D spectrum
ax = axes[0, 1]
ax.plot(freqs_1d[:80], spectrum_1d[:80], color='tomato', linewidth=1.5)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('1D Frequency Spectrum  |X[k]|', fontweight='bold')
ax.grid(True, alpha=0.3)
for f, label in [(5, '5 Hz'), (20, '20 Hz'), (50, '50 Hz')]:
    idx = np.argmin(np.abs(freqs_1d - f))
    ax.annotate(label, xy=(freqs_1d[idx], spectrum_1d[idx]),
                xytext=(freqs_1d[idx] + 2, spectrum_1d[idx] * 0.85),
                fontsize=8, color='darkred')

# 2D image
ax = axes[1, 0]
im = ax.imshow(image_2d, cmap='gray', aspect='auto')
ax.set_xlabel('x (pixels)')
ax.set_ylabel('y (pixels)')
ax.set_title('2D Image  I[x, y]', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

# 2D spatial frequency spectrum
ax = axes[1, 1]
im2 = ax.imshow(spectrum_2d_log, cmap='inferno', aspect='auto')
ax.set_xlabel('Spatial Frequency fx')
ax.set_ylabel('Spatial Frequency fy')
ax.set_title('2D Spatial Frequency Spectrum  |F[fx, fy]|  (log)', fontweight='bold')
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
cx, cy = N // 2, N // 2
ax.plot(cx, cy, 'w+', markersize=12, markeredgewidth=2)
ax.text(cx + 3, cy - 8, 'DC (0,0)', color='white', fontsize=8)

# Annotations
axes[0, 1].text(0.98, 0.95,
    '1D: frequency along\none axis (time)',
    transform=axes[0, 1].transAxes, ha='right', va='top',
    fontsize=8, color='gray',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))

axes[1, 1].text(0.98, 0.95,
    '2D: spatial frequency\nalong two axes (x, y)',
    transform=axes[1, 1].transAxes, ha='right', va='top',
    fontsize=8, color='white',
    bbox=dict(boxstyle='round,pad=0.3', facecolor='black', alpha=0.5))

plt.suptitle(
    'From 1D Fourier to 2D Fourier: Same Idea, More Dimensions',
    fontsize=13, fontweight='bold'
)
plt.tight_layout()
plt.savefig('assets/ch01_fourier_2d.png', dpi=120, bbox_inches='tight')
print("Saved: assets/ch01_fourier_2d.png")
