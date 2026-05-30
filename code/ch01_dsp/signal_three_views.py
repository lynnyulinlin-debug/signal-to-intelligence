"""
Signal Three Views Visualization
Chapter 1: Digital Signal Processing - Section 1.1

Shows the same signal from three perspectives:
1. Time domain (waveform)
2. Frequency domain (spectrum)
3. Time-frequency domain (spectrogram)

Uses a non-stationary signal so the three views show clearly different information.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

# ============ Signal: two tones that switch halfway ============
FS = 2000
DURATION = 2.0
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)

half = len(t) // 2
sig = np.zeros_like(t)
# First half: 100 Hz + 300 Hz
sig[:half] = (
    np.sin(2 * np.pi * 100 * t[:half])
    + 0.6 * np.sin(2 * np.pi * 300 * t[:half])
)
# Second half: 200 Hz + 500 Hz
sig[half:] = (
    0.8 * np.sin(2 * np.pi * 200 * t[half:])
    + 0.5 * np.sin(2 * np.pi * 500 * t[half:])
)
# Smooth transition
fade = 40
sig[half - fade:half] *= np.linspace(1, 0, fade)
sig[half:half + fade] *= np.linspace(0, 1, fade)
sig += 0.05 * np.random.randn(len(t))

# ============ Frequency spectrum ============
freqs = np.fft.rfftfreq(len(sig), 1 / FS)
spectrum = np.abs(np.fft.rfft(sig))

# ============ Spectrogram ============
f_spec, t_spec, Sxx = scipy_signal.spectrogram(
    sig, fs=FS, window='hann', nperseg=256, noverlap=240, scaling='spectrum')
Sxx_db = 10 * np.log10(Sxx + 1e-10)
freq_mask = f_spec <= 700

# ============ Plot ============
fig, axes = plt.subplots(3, 1, figsize=(12, 9))

# --- Row 1: Time domain ---
ax = axes[0]
ax.plot(t, sig, color='steelblue', linewidth=0.9)
ax.axvline(x=DURATION / 2, color='gray', linestyle='--', linewidth=1.2, alpha=0.7)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Time Domain  x[n]', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
ax.text(0.25, 0.88, '100 Hz + 300 Hz', transform=ax.transAxes,
        ha='center', fontsize=9, color='steelblue',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(0.75, 0.88, '200 Hz + 500 Hz', transform=ax.transAxes,
        ha='center', fontsize=9, color='tomato',
        bbox=dict(boxstyle='round', facecolor='white', alpha=0.7))
ax.text(0.98, 0.05,
        'Shows WHEN the signal changes,\nbut not WHAT frequencies are present',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray')

# --- Row 2: Frequency domain ---
ax = axes[1]
ax.plot(freqs[:400], spectrum[:400], color='tomato', linewidth=1.2)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Frequency Domain  |X[k]|', fontweight='bold', fontsize=12)
ax.grid(True, alpha=0.3)
for f, label in [(100, '100'), (200, '200'), (300, '300'), (500, '500')]:
    idx = np.argmin(np.abs(freqs - f))
    ax.annotate(f'{label} Hz', xy=(freqs[idx], spectrum[idx]),
                xytext=(freqs[idx] + 8, spectrum[idx] * 0.82),
                fontsize=8, color='darkred',
                arrowprops=dict(arrowstyle='->', color='darkred', lw=0.8))
ax.text(0.98, 0.05,
        'Shows WHAT frequencies exist,\nbut not WHEN they appear',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='gray')

# --- Row 3: Time-frequency domain ---
ax = axes[2]
im = ax.pcolormesh(t_spec, f_spec[freq_mask], Sxx_db[freq_mask],
                   shading='gouraud', cmap='magma',
                   vmin=Sxx_db[freq_mask].max() - 45)
ax.axvline(x=DURATION / 2, color='white', linestyle='--', linewidth=1.2, alpha=0.6)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Time-Frequency Domain  S(t, f)  — Spectrogram', fontweight='bold', fontsize=12)
plt.colorbar(im, ax=ax, label='Power (dB)')
ax.text(0.98, 0.05,
        'Shows BOTH when and what:\nfrequency content over time',
        transform=ax.transAxes, ha='right', va='bottom', fontsize=8, color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.4))

plt.suptitle(
    'Same Signal — Three Perspectives\n'
    'Time domain loses frequency info; Frequency domain loses time info; '
    'Spectrogram keeps both',
    fontsize=11, fontweight='bold'
)
plt.tight_layout()
plt.savefig('assets/ch01_three_views.png', dpi=120, bbox_inches='tight')
print("Saved: assets/ch01_three_views.png")
