"""
实验7.4：Qwen2.5-VL模型分析
对应章节：第7章 - 多模态LLM
目标：分析Qwen2.5-VL的架构和性能，对比不同多模态模型
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

plt.rcParams["font.family"] = "sans-serif"
plt.rcParams["axes.unicode_minus"] = False

OUTPUT_PATH = Path("assets/ch07_qwen_vl_analysis.png")


def run_experiment():
    models_data = {
        "CLIP": {
            "release_year": 2021,
            "resolution": 224,
            "multi_image": False,
            "chinese_optimized": False,
            "architecture": "Dual Encoder",
            "parameters": 400,
            "inference_speed": 100,
            "accuracy": 76.2,
            "cost": 1.0,
        },
        "LLaVA-1.5": {
            "release_year": 2023,
            "resolution": 224,
            "multi_image": False,
            "chinese_optimized": False,
            "architecture": "Single Encoder + LLM",
            "parameters": 7000,
            "inference_speed": 80,
            "accuracy": 78.5,
            "cost": 1.5,
        },
        "LLaVA-NeXT": {
            "release_year": 2024,
            "resolution": 1024,
            "multi_image": True,
            "chinese_optimized": False,
            "architecture": "Single Encoder + LLM",
            "parameters": 34000,
            "inference_speed": 60,
            "accuracy": 82.3,
            "cost": 2.5,
        },
        "Qwen2.5-VL": {
            "release_year": 2024,
            "resolution": 1024,
            "multi_image": True,
            "chinese_optimized": True,
            "architecture": "Unified Transformer",
            "parameters": 32000,
            "inference_speed": 70,
            "accuracy": 84.1,
            "cost": 2.0,
        },
        "GPT-4V": {
            "release_year": 2023,
            "resolution": 2048,
            "multi_image": True,
            "chinese_optimized": False,
            "architecture": "Proprietary",
            "parameters": 1000000,
            "inference_speed": 40,
            "accuracy": 88.5,
            "cost": 10.0,
        },
    }

    model_names = list(models_data.keys())
    years = sorted(set(model["release_year"] for model in models_data.values()))
    models_by_year = {year: [] for year in years}
    for model, data in models_data.items():
        models_by_year[data["release_year"]].append(model)

    resolutions = np.array([models_data[m]["resolution"] for m in model_names], dtype=float)
    accuracies = np.array([models_data[m]["accuracy"] for m in model_names], dtype=float)
    speeds = np.array([models_data[m]["inference_speed"] for m in model_names], dtype=float)
    costs = np.array([models_data[m]["cost"] for m in model_names], dtype=float)
    parameters = np.array([models_data[m]["parameters"] for m in model_names], dtype=float)
    multi_image_support = np.array([models_data[m]["multi_image"] for m in model_names], dtype=float)
    chinese_support = np.array([models_data[m]["chinese_optimized"] for m in model_names], dtype=float)
    efficiency = accuracies / costs
    speed_accuracy_ratio = accuracies / (101 - speeds)

    return {
        "models_data": models_data,
        "model_names": model_names,
        "years": years,
        "models_by_year": models_by_year,
        "resolutions": resolutions,
        "accuracies": accuracies,
        "speeds": speeds,
        "costs": costs,
        "parameters": parameters,
        "multi_image_support": multi_image_support,
        "chinese_support": chinese_support,
        "efficiency": efficiency,
        "speed_accuracy_ratio": speed_accuracy_ratio,
    }


def print_summary(result):
    print("=" * 80)
    print("多模态LLM模型对比分析")
    print("=" * 80)
    print()
    print("模型时间演进:")
    print("-" * 80)
    for year in sorted(result["years"]):
        print(f"{year}: {', '.join(result['models_by_year'][year])}")
    print()
    print("详细对比:")
    print("-" * 80)
    for model in result["model_names"]:
        data = result["models_data"][model]
        idx = result["model_names"].index(model)
        print(f"\n{model}:")
        print(f"  发布年份: {data['release_year']}")
        print(f"  分辨率: {data['resolution']}x{data['resolution']}")
        print(f"  多图像支持: {'✓' if data['multi_image'] else '✗'}")
        print(f"  中文优化: {'✓' if data['chinese_optimized'] else '✗'}")
        print(f"  架构: {data['architecture']}")
        print(f"  参数量: {data['parameters']}M")
        print(f"  推理速度: {data['inference_speed']} (相对)")
        print(f"  准确率: {data['accuracy']:.1f}%")
        print(f"  相对成本: {data['cost']:.1f}x")
        print(f"  性价比: {result['efficiency'][idx]:.2f}")
    print()
    print("=" * 80)


def plot_results(result, output_path=OUTPUT_PATH):
    fig = plt.figure(figsize=(18, 12))
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

    ax = fig.add_subplot(gs[0, 0])
    years_list = [result["models_data"][m]["release_year"] for m in result["model_names"]]
    ax.scatter(years_list, result["accuracies"], s=200, alpha=0.6, c=range(len(result["model_names"])), cmap="viridis")
    for i, model in enumerate(result["model_names"]):
        ax.annotate(model, (years_list[i], result["accuracies"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.plot(years_list, result["accuracies"], "k--", alpha=0.3)
    ax.set_xlabel("Release Year")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Accuracy Evolution")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 1])
    scatter = ax.scatter(result["resolutions"], result["accuracies"], s=200, alpha=0.6, c=result["costs"], cmap="RdYlGn_r")
    for i, model in enumerate(result["model_names"]):
        ax.annotate(model, (result["resolutions"][i], result["accuracies"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Resolution")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Resolution vs Accuracy")
    plt.colorbar(scatter, ax=ax, label="Cost")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[0, 2])
    colors = ["steelblue" if m != "Qwen2.5-VL" else "coral" for m in result["model_names"]]
    bars = ax.barh(result["model_names"], result["efficiency"], color=colors, alpha=0.7, edgecolor="black")
    ax.set_xlabel("Efficiency (Accuracy/Cost)")
    ax.set_title("Cost-Effectiveness")
    ax.grid(True, alpha=0.3, axis="x")
    for i, (bar, eff) in enumerate(zip(bars, result["efficiency"])):
        ax.text(eff, i, f" {eff:.2f}", va="center", fontsize=9)

    ax = fig.add_subplot(gs[1, 0])
    scatter = ax.scatter(result["parameters"], result["speeds"], s=200, alpha=0.6, c=result["accuracies"], cmap="viridis")
    for i, model in enumerate(result["model_names"]):
        ax.annotate(model, (result["parameters"][i], result["speeds"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Parameters (Millions)")
    ax.set_ylabel("Inference Speed (relative)")
    ax.set_title("Model Size vs Speed")
    plt.colorbar(scatter, ax=ax, label="Accuracy")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[1, 1])
    x_pos = np.arange(len(result["model_names"]))
    width = 0.35
    ax.bar(x_pos - width / 2, result["multi_image_support"], width, label="Multi-Image", alpha=0.7)
    ax.bar(x_pos + width / 2, result["chinese_support"], width, label="Chinese Optimized", alpha=0.7)
    ax.set_ylabel("Support (1=Yes, 0=No)")
    ax.set_title("Feature Support")
    ax.set_xticks(x_pos)
    ax.set_xticklabels(result["model_names"], rotation=45, ha="right")
    ax.legend()
    ax.set_ylim(0, 1.2)
    ax.grid(True, alpha=0.3, axis="y")

    ax = fig.add_subplot(gs[1, 2])
    scatter = ax.scatter(result["costs"], result["accuracies"], s=300, alpha=0.6, c=result["speeds"], cmap="coolwarm")
    for i, model in enumerate(result["model_names"]):
        ax.annotate(model, (result["costs"][i], result["accuracies"][i]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    ax.set_xlabel("Relative Cost")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Cost vs Accuracy")
    plt.colorbar(scatter, ax=ax, label="Speed")
    ax.grid(True, alpha=0.3)

    ax = fig.add_subplot(gs[2, :], projection="polar")
    compare_models = ["CLIP", "LLaVA-1.5", "Qwen2.5-VL", "GPT-4V"]
    categories = ["Accuracy", "Resolution", "Speed", "Cost-Effectiveness", "Multi-Image"]
    n = len(categories)
    angles = np.linspace(0, 2 * np.pi, n, endpoint=False).tolist()
    angles += angles[:1]
    max_accuracy = max(result["accuracies"])
    max_resolution = max(result["resolutions"])
    max_speed = max(result["speeds"])
    max_efficiency = max(result["efficiency"])
    colors_radar = ["steelblue", "coral", "lightgreen", "gold"]

    for idx, model in enumerate(compare_models):
        model_idx = result["model_names"].index(model)
        values = [
            result["accuracies"][model_idx] / max_accuracy,
            result["resolutions"][model_idx] / max_resolution,
            result["speeds"][model_idx] / max_speed,
            result["efficiency"][model_idx] / max_efficiency,
            result["multi_image_support"][model_idx],
        ]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=model, color=colors_radar[idx])
        ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(categories)
    ax.set_ylim(0, 1)
    ax.set_title("Model Comparison (Normalized)", pad=20)
    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1))
    ax.grid(True)

    plt.savefig(output_path, dpi=100, bbox_inches="tight")
    plt.close(fig)
    return Path(output_path)


def main():
    result = run_experiment()
    print_summary(result)
    output_path = plot_results(result)
    print(f"图表已保存到: {output_path}")


if __name__ == "__main__":
    main()
