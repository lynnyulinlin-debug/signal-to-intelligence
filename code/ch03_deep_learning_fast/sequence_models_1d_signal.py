"""
实验3.5：序列模型与一维信号处理
对应章节：第3章 - 深度学习快速通道
目标：对比1D CNN、RNN和Transformer风格模型在时序信号上的建模方式
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

TIME_STEPS = 200
OUTPUT_PATH = Path("assets/ch03_sequence_models.png")


def generate_signal(timesteps=TIME_STEPS, seed=42):
    rng = np.random.RandomState(seed)
    x = np.linspace(0, 4 * np.pi, timesteps)
    signal = 0.6 * np.sin(x) + 0.25 * np.sin(4 * x)
    signal = signal.copy()
    signal[80:90] += 0.8
    signal[145:150] -= 0.6
    signal += 0.05 * rng.randn(timesteps)
    return signal


def cnn_style_responses(signal):
    kernel_local = np.array([-1.0, 0.0, 1.0])
    kernel_smooth = np.array([0.25, 0.5, 0.25])
    edge_response = np.convolve(signal, kernel_local, mode="same")
    smooth_response = np.convolve(signal, kernel_smooth, mode="same")
    return edge_response, smooth_response


def rnn_style_state(signal, alpha=0.85):
    state = np.zeros(len(signal))
    for t in range(1, len(signal)):
        state[t] = alpha * state[t - 1] + (1 - alpha) * signal[t]
    return state


def transformer_like_summary(signal, window=25, decay=8, global_mix=0.3):
    summary = np.zeros(len(signal))
    for t in range(len(signal)):
        left = max(0, t - window)
        right = min(len(signal), t + window)
        local = signal[left:right]
        weights = np.exp(-np.abs(np.arange(left, right) - t) / decay)
        weights = weights / weights.sum()
        summary[t] = np.sum(local * weights)

    global_context = np.mean(signal)
    combined = (1 - global_mix) * summary + global_mix * global_context
    return summary, combined, global_context


def run_experiment(seed=42):
    signal = generate_signal(seed=seed)
    cnn_edge, cnn_smooth = cnn_style_responses(signal)
    rnn_state = rnn_style_state(signal)
    attention_summary, transformer_like, global_context = transformer_like_summary(signal)

    return {
        "signal": signal,
        "cnn_edge": cnn_edge,
        "cnn_smooth": cnn_smooth,
        "rnn_state": rnn_state,
        "attention_summary": attention_summary,
        "transformer_like": transformer_like,
        "global_context": global_context,
        "timesteps": len(signal),
    }


def print_summary(result):
    signal = result["signal"]
    print("=" * 70)
    print("Sequence Models and 1D Signal Processing")
    print("=" * 70)
    print(f"信号长度: {result['timesteps']}")
    print(f"信号均值: {signal.mean():.4f}")
    print(f"信号标准差: {signal.std():.4f}")
    print()
    print("模型直觉对比:")
    print("- 1D CNN: 更关注局部变化和模式")
    print("- RNN: 通过递推状态累积历史信息")
    print("- Transformer-like: 通过全局注意机制聚合远近信息")
    print("=" * 70)


def plot_results(result, output_path=OUTPUT_PATH):
    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)

    axes[0].plot(result["signal"], color="black", linewidth=1.8)
    axes[0].set_title("Input 1D Signal")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(result["cnn_edge"], label="Edge-like Kernel", color="tab:red")
    axes[1].plot(result["cnn_smooth"], label="Smoothing Kernel", color="tab:blue")
    axes[1].set_title("1D CNN-style Local Responses")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(result["rnn_state"], color="tab:green")
    axes[2].set_title("RNN-style Recursive State")
    axes[2].grid(True, alpha=0.3)

    axes[3].plot(result["transformer_like"], color="tab:purple")
    axes[3].set_title("Transformer-style Global Context Summary")
    axes[3].grid(True, alpha=0.3)
    axes[3].set_xlabel("Time Step")

    plt.tight_layout()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(output_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return output_path


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"Figure saved to: {output_path}")


if __name__ == "__main__":
    main()
