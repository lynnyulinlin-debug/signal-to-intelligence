"""
Signal Dimensions Visualization
Chapter 1: Digital Signal Processing

Demonstrates the relationship between:
1. 1D temporal signal (audio waveform)
2. 2D spatial signal (image)
3. Derived 2D representation (spectrogram from 1D signal)
"""

import numpy as np
import matplotlib.pyplot as plt

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ============ 1D Signal ============
fs = 1000
t = np.linspace(0, 1, fs)
signal_1d = (
    np.sin(2 * np.pi * 5 * t)
    + 0.5 * np.sin(2 * np.pi * 20 * t)
    + 0.2 * np.random.randn(len(t))
)

# ============ 2D Image ============
H, W = 64, 64
x_img = np.linspace(0, 4 * np.pi, W)
y_img = np.linspace(0, 4 * np.pi, H)
X, Y = np.meshgrid(x_img, y_img)
image_2d = np.sin(X) * np.cos(Y) + 0.1 * np.random.randn(H, W)

# ============ Spectrogram (derived 2D from 1D) ============
window_size = 64
hop = 16
n_frames = (len(signal_1d) - window_size) // hop + 1
spectrogram = np.zeros((window_size // 2, n_frames))
window = np.hanning(window_size)
for i in range(n_frames):
    frame = signal_1d[i * hop: i * hop + window_size] * window
    spectrum = np.abs(np.fft.rfft(frame))[: window_size // 2]
    spectrogram[:, i] = spectrum

# ============ Plot ============
fig, axes = plt.subplots(1, 3, figsize=(15, 4))

# 1D waveform
ax = axes[0]
ax.plot(t[:300], signal_1d[:300], color='steelblue', linewidth=1.2)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('1D Signal\n(Audio / Sensor / Time Series)', fontweight='bold')
ax.grid(True, alpha=0.3)
ax.text(0.05, 0.92, 'x[n]  shape: (N,)', transform=ax.transAxes,
        fontsize=9, color='gray', va='top')

# 2D image
ax = axes[1]
im = ax.imshow(image_2d, cmap='viridis', aspect='auto')
ax.set_xlabel('Width (pixels)')
ax.set_ylabel('Height (pixels)')
ax.set_title('2D Signal\n(Image / Spatial Data)', fontweight='bold')
plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
ax.text(0.05, 0.92, 'I[h,w]  shape: (H, W)', transform=ax.transAxes,
        fontsize=9, color='white', va='top')

# Spectrogram
ax = axes[2]
im2 = ax.imshow(spectrogram, origin='lower', aspect='auto',
                cmap='magma',
                extent=[0, 1, 0, fs // 2])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Derived 2D Representation\n(Spectrogram from 1D Signal)', fontweight='bold')
plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04, label='Magnitude')
ax.text(0.05, 0.92, 'S[f,t]  shape: (F, T)', transform=ax.transAxes,
        fontsize=9, color='white', va='top')

plt.suptitle(
    'Signal Dimensionality: 1D → 2D → Derived 2D',
    fontsize=13, fontweight='bold', y=1.02
)
plt.tight_layout()
plt.savefig('assets/ch01_signal_dimensions.png', dpi=120, bbox_inches='tight')
print("Saved: assets/ch01_signal_dimensions.png")
