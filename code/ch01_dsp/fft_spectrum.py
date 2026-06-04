"""
实验1.1：FFT频谱分析
对应章节：第1章 - 数字信号处理基础
目标：生成正弦波+噪声，用FFT观察频谱
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

# ============ 配置 ============
SIGNAL_LENGTH = 1000
SAMPLING_RATE = 100  # Hz
SIGNAL_FREQ = [5, 10]  # Hz
NOISE_LEVEL = 0.1
OUTPUT_PATH = Path("assets/ch01_fft_spectrum.png")


# ============ 核心逻辑 ============
def generate_signal(
    signal_length=SIGNAL_LENGTH,
    sampling_rate=SAMPLING_RATE,
    signal_freq=SIGNAL_FREQ,
    noise_level=NOISE_LEVEL,
    seed=42,
):
    """生成多频率正弦信号和加噪版本。"""
    rng = np.random.RandomState(seed)
    t = np.arange(signal_length) / sampling_rate

    signal = np.zeros(signal_length)
    for freq in signal_freq:
        signal += np.sin(2 * np.pi * freq * t)

    signal_noisy = signal + noise_level * rng.randn(signal_length)
    return t, signal, signal_noisy


def compute_fft_spectrum(signal, sampling_rate=SAMPLING_RATE):
    """计算 FFT 并返回正频率部分。"""
    signal_length = len(signal)
    fft_result = np.fft.fft(signal)
    frequencies = np.fft.fftfreq(signal_length, 1 / sampling_rate)
    magnitude = np.abs(fft_result)

    positive_freq_idx = frequencies > 0
    return frequencies[positive_freq_idx], magnitude[positive_freq_idx], fft_result


def detect_top_frequencies(frequencies_positive, magnitude_positive, top_k=2):
    """找出幅度最大的频率分量。"""
    top_indices = np.argsort(magnitude_positive)[-top_k:][::-1]
    detected_freqs = frequencies_positive[top_indices]
    return detected_freqs, top_indices


def run_experiment(
    signal_length=SIGNAL_LENGTH,
    sampling_rate=SAMPLING_RATE,
    signal_freq=SIGNAL_FREQ,
    noise_level=NOISE_LEVEL,
    seed=42,
):
    """运行 FFT 频谱分析实验并返回中间结果。"""
    t, signal, signal_noisy = generate_signal(
        signal_length=signal_length,
        sampling_rate=sampling_rate,
        signal_freq=signal_freq,
        noise_level=noise_level,
        seed=seed,
    )
    frequencies_positive, magnitude_positive, fft_result = compute_fft_spectrum(
        signal_noisy, sampling_rate
    )
    detected_freqs, top_indices = detect_top_frequencies(
        frequencies_positive, magnitude_positive, top_k=len(signal_freq)
    )

    return {
        "t": t,
        "signal": signal,
        "signal_noisy": signal_noisy,
        "fft_result": fft_result,
        "frequencies_positive": frequencies_positive,
        "magnitude_positive": magnitude_positive,
        "detected_freqs": detected_freqs,
        "top_indices": top_indices,
        "signal_length": signal_length,
        "sampling_rate": sampling_rate,
        "signal_freq": signal_freq,
        "noise_level": noise_level,
    }


def print_summary(result):
    print("=" * 50)
    print("FFT Spectrum Analysis")
    print("=" * 50)
    print(f"Signal length: {result['signal_length']}")
    print(f"Sampling rate: {result['sampling_rate']} Hz")
    print(f"True frequencies: {result['signal_freq']}")
    print(f"Detected frequencies: {result['detected_freqs'].round(1)}")
    print(f"Noise level: {result['noise_level']}")
    print("=" * 50)


def plot_spectrum(result, output_path=OUTPUT_PATH):
    """保存时域和频域可视化图。"""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8))

    t = result["t"]
    signal = result["signal"]
    signal_noisy = result["signal_noisy"]
    frequencies_positive = result["frequencies_positive"]
    magnitude_positive = result["magnitude_positive"]
    detected_freqs = result["detected_freqs"]
    top_indices = result["top_indices"]

    axes[0].plot(t[:200], signal_noisy[:200], "b-", linewidth=0.8, label="Noisy Signal")
    axes[0].plot(t[:200], signal[:200], "r--", linewidth=1, label="Clean Signal")
    axes[0].set_xlabel("Time (s)")
    axes[0].set_ylabel("Amplitude")
    axes[0].set_title("Time Domain Signal (first 2 seconds)")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(frequencies_positive[:100], magnitude_positive[:100], "b-", linewidth=1)
    axes[1].scatter(
        detected_freqs,
        magnitude_positive[top_indices],
        color="r",
        s=100,
        label=f"Detected: {detected_freqs.round(1)} Hz",
        zorder=5,
    )
    axes[1].set_xlabel("Frequency (Hz)")
    axes[1].set_ylabel("Magnitude")
    axes[1].set_title("Frequency Domain (FFT)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_spectrum(result)
    print(f"\n图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
