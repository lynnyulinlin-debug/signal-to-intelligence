"""
Time-Frequency Analysis Visualization
Chapter 1: Digital Signal Processing - Section 1.4

Demonstrates:
1. Music-like signal: time domain, frequency spectrum, spectrogram
2. STFT with different window sizes (time-frequency resolution tradeoff)
3. Wavelet-like multi-scale analysis
4. Multi-signal comparison: stationary vs non-stationary
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy import signal as scipy_signal

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

np.random.seed(42)

FS = 4000  # sample rate Hz
DURATION = 2.0
t = np.linspace(0, DURATION, int(FS * DURATION), endpoint=False)

# ============================================================
# Figure 1: Music-like signal — time, spectrum, spectrogram
# ============================================================

def make_music_signal(t, fs):
    """Simulate a music-like signal with melody notes and harmonics."""
    # Note sequence: C4(262Hz) D4(294Hz) E4(330Hz) G4(392Hz)
    note_freqs = [262, 294, 330, 392, 330, 294, 262]
    note_dur = DURATION / len(note_freqs)
    sig = np.zeros_like(t)
    for i, freq in enumerate(note_freqs):
        start = int(i * note_dur * fs)
        end = int((i + 1) * note_dur * fs)
        seg = t[start:end] - t[start]
        # Fundamental + harmonics
        sig[start:end] = (
            np.sin(2 * np.pi * freq * seg)
            + 0.5 * np.sin(2 * np.pi * 2 * freq * seg)
            + 0.25 * np.sin(2 * np.pi * 3 * freq * seg)
        )
        # Amplitude envelope (ADSR-like)
        env = np.ones(end - start)
        attack = int(0.05 * (end - start))
        release = int(0.15 * (end - start))
        env[:attack] = np.linspace(0, 1, attack)
        env[-release:] = np.linspace(1, 0, release)
        sig[start:end] *= env
    sig += 0.03 * np.random.randn(len(t))
    return sig

music = make_music_signal(t, FS)

fig1, axes = plt.subplots(3, 1, figsize=(14, 10))

# Time domain
ax = axes[0]
ax.plot(t, music, color='steelblue', linewidth=0.8)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude')
ax.set_title('Music Signal — Time Domain', fontweight='bold')
ax.grid(True, alpha=0.3)
note_labels = ['C4', 'D4', 'E4', 'G4', 'E4', 'D4', 'C4']
note_dur = DURATION / len(note_labels)
for i, label in enumerate(note_labels):
    ax.axvline(x=i * note_dur, color='gray', linestyle='--', alpha=0.4, linewidth=0.8)
    ax.text(i * note_dur + note_dur / 2, music.max() * 0.85, label,
            ha='center', fontsize=8, color='darkred')

# Frequency spectrum
ax = axes[1]
freqs = np.fft.rfftfreq(len(music), 1 / FS)
spectrum = np.abs(np.fft.rfft(music))
ax.plot(freqs[:800], spectrum[:800], color='tomato', linewidth=1.0)
ax.set_xlabel('Frequency (Hz)')
ax.set_ylabel('Magnitude')
ax.set_title('Music Signal — Frequency Spectrum (FFT)', fontweight='bold')
ax.grid(True, alpha=0.3)
for freq, label in [(262, 'C4'), (294, 'D4'), (330, 'E4'), (392, 'G4')]:
    ax.axvline(x=freq, color='steelblue', linestyle=':', alpha=0.6, linewidth=1)
    ax.text(freq + 4, spectrum.max() * 0.7, label, fontsize=7, color='steelblue')
ax.text(0.98, 0.92, 'FFT loses time info:\ncannot tell when each note plays',
        transform=ax.transAxes, ha='right', va='top', fontsize=8, color='gray',
        bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))

# Spectrogram
ax = axes[2]
f_spec, t_spec, Sxx = scipy_signal.spectrogram(
    music, fs=FS, window='hann', nperseg=512, noverlap=480, scaling='spectrum')
Sxx_db = 10 * np.log10(Sxx + 1e-10)
im = ax.pcolormesh(t_spec, f_spec[:100], Sxx_db[:100], shading='gouraud',
                   cmap='magma', vmin=Sxx_db[:100].max() - 50)
ax.set_xlabel('Time (s)')
ax.set_ylabel('Frequency (Hz)')
ax.set_title('Music Signal — Spectrogram (STFT)', fontweight='bold')
plt.colorbar(im, ax=ax, label='Power (dB)')
for i, (freq, label) in enumerate([(262, 'C4'), (294, 'D4'), (330, 'E4'), (392, 'G4')]):
    ax.axhline(y=freq, color='white', linestyle='--', alpha=0.4, linewidth=0.8)
ax.text(0.98, 0.92, 'Spectrogram shows WHEN each note plays',
        transform=ax.transAxes, ha='right', va='top', fontsize=8, color='white',
        bbox=dict(boxstyle='round', facecolor='black', alpha=0.5))

plt.suptitle('Same Signal, Three Views: Time / Frequency / Time-Frequency',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/ch01_time_freq_music.png', dpi=120, bbox_inches='tight')
print("Saved: assets/ch01_time_freq_music.png")
plt.close()

# ============================================================
# Figure 2: STFT window size tradeoff + wavelet + comparison
# ============================================================

# Non-stationary test signal: chirp + impulse + tone
def make_test_signal(t, fs):
    chirp = scipy_signal.chirp(t, f0=50, f1=400, t1=DURATION, method='linear')
    tone = 0.6 * np.sin(2 * np.pi * 200 * t)
    impulse = np.zeros_like(t)
    impulse[int(1.0 * fs)] = 3.0
    impulse[int(1.5 * fs)] = -2.5
    return chirp + tone + impulse + 0.05 * np.random.randn(len(t))

test_sig = make_test_signal(t, FS)

fig2, axes = plt.subplots(3, 2, figsize=(16, 12))

# Row 0: STFT small window (high time res, low freq res)
for col, (nperseg, title_note) in enumerate([
    (64,  'Short window (high time res, low freq res)'),
    (1024, 'Long window (low time res, high freq res)'),
]):
    ax = axes[0, col]
    f_s, t_s, Sxx = scipy_signal.spectrogram(
        test_sig, fs=FS, window='hann', nperseg=nperseg,
        noverlap=int(nperseg * 0.75), scaling='spectrum')
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    mask = f_s <= 600
    im = ax.pcolormesh(t_s, f_s[mask], Sxx_db[mask], shading='gouraud',
                       cmap='inferno', vmin=Sxx_db[mask].max() - 50)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(f'STFT — {title_note}', fontweight='bold')
    plt.colorbar(im, ax=ax, label='dB')

# Row 1: Continuous Wavelet Transform (CWT)
ax = axes[1, 0]
widths = np.geomspace(2, 200, 80)
cwt_matrix = scipy_signal.cwt(test_sig[::4], scipy_signal.ricker, widths)
t_cwt = t[::4]
pseudo_freq = FS / (4 * widths * 4)
im = ax.pcolormesh(t_cwt, pseudo_freq, np.abs(cwt_matrix),
                   shading='gouraud', cmap='plasma')
ax.set_xlabel('Time (s)')
ax.set_ylabel('Pseudo-frequency (Hz)')
ax.set_title('Wavelet Transform (CWT)\nAdaptive resolution: wide at low freq, narrow at high freq',
             fontweight='bold')
ax.set_ylim(0, 600)
plt.colorbar(im, ax=ax, label='Magnitude')

# Row 1 right: Resolution comparison diagram
ax = axes[1, 1]
ax.set_xlim(0, 10)
ax.set_ylim(0, 10)
ax.set_aspect('equal')
ax.set_title('Time-Frequency Resolution Tradeoff', fontweight='bold')
ax.set_xlabel('Time resolution →')
ax.set_ylabel('Frequency resolution →')

# STFT tiles (fixed size)
for i in range(5):
    for j in range(5):
        rect = plt.Rectangle((i * 2, j * 2), 1.8, 1.8,
                              linewidth=1, edgecolor='steelblue',
                              facecolor='lightblue', alpha=0.5)
        ax.add_patch(rect)
ax.text(5, 9.5, 'STFT: fixed tiles', color='steelblue', fontsize=9, ha='center')

# Wavelet tiles (variable size)
for j, (h, w) in enumerate([(0.4, 3.5), (0.8, 2.5), (1.5, 1.5), (2.5, 0.8), (3.5, 0.4)]):
    rect = plt.Rectangle((6, j * 2), w, h * 2,
                          linewidth=1, edgecolor='tomato',
                          facecolor='mistyrose', alpha=0.6)
    ax.add_patch(rect)
ax.text(8, 9.5, 'Wavelet: adaptive tiles', color='tomato', fontsize=9, ha='center')
ax.grid(False)
ax.set_xticks([])
ax.set_yticks([])

# Row 2: Stationary vs non-stationary signal comparison
stationary = np.sin(2 * np.pi * 150 * t) + 0.3 * np.random.randn(len(t))
nonstationary = np.concatenate([
    np.sin(2 * np.pi * 80 * t[:len(t)//3]),
    np.sin(2 * np.pi * 200 * t[len(t)//3:2*len(t)//3]),
    np.sin(2 * np.pi * 350 * t[2*len(t)//3:]),
]) + 0.1 * np.random.randn(len(t))

for col, (sig, label) in enumerate([
    (stationary, 'Stationary Signal (constant 150 Hz)'),
    (nonstationary, 'Non-stationary Signal (80→200→350 Hz)'),
]):
    ax = axes[2, col]
    f_s, t_s, Sxx = scipy_signal.spectrogram(
        sig, fs=FS, window='hann', nperseg=256, noverlap=240, scaling='spectrum')
    Sxx_db = 10 * np.log10(Sxx + 1e-10)
    mask = f_s <= 500
    im = ax.pcolormesh(t_s, f_s[mask], Sxx_db[mask], shading='gouraud',
                       cmap='viridis', vmin=Sxx_db[mask].max() - 40)
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    ax.set_title(label, fontweight='bold')
    plt.colorbar(im, ax=ax, label='dB')

plt.suptitle('Time-Frequency Analysis: Window Size, Wavelet, and Signal Types',
             fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('assets/ch01_time_freq_comparison.png', dpi=120, bbox_inches='tight')
print("Saved: assets/ch01_time_freq_comparison.png")
plt.close()

print("\nAll figures generated successfully.")
