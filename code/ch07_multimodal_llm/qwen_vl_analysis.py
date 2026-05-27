"""
实验7.4：Qwen2.5-VL模型分析
对应章节：第7章 - 多模态LLM
目标：分析Qwen2.5-VL的架构和性能，对比不同多模态模型
"""

import numpy as np
import matplotlib.pyplot as plt

# ============ 配置 ============
np.random.seed(42)

# ============ 核心逻辑 ============
# 多模态模型对比数据（基于公开信息）
models_data = {
    'CLIP': {
        'release_year': 2021,
        'resolution': 224,
        'multi_image': False,
        'chinese_optimized': False,
        'architecture': 'Dual Encoder',
        'parameters': 400,  # 百万
        'inference_speed': 100,  # 相对速度
        'accuracy': 76.2,  # ImageNet准确率
        'cost': 1.0,  # 相对成本
    },
    'LLaVA-1.5': {
        'release_year': 2023,
        'resolution': 224,
        'multi_image': False,
        'chinese_optimized': False,
        'architecture': 'Single Encoder + LLM',
        'parameters': 7000,
        'inference_speed': 80,
        'accuracy': 78.5,
        'cost': 1.5,
    },
    'LLaVA-NeXT': {
        'release_year': 2024,
        'resolution': 1024,
        'multi_image': True,
        'chinese_optimized': False,
        'architecture': 'Single Encoder + LLM',
        'parameters': 34000,
        'inference_speed': 60,
        'accuracy': 82.3,
        'cost': 2.5,
    },
    'Qwen2.5-VL': {
        'release_year': 2024,
        'resolution': 1024,
        'multi_image': True,
        'chinese_optimized': True,
        'architecture': 'Unified Transformer',
        'parameters': 32000,
        'inference_speed': 70,
        'accuracy': 84.1,
        'cost': 2.0,
    },
    'GPT-4V': {
        'release_year': 2023,
        'resolution': 2048,
        'multi_image': True,
        'chinese_optimized': False,
        'architecture': 'Proprietary',
        'parameters': 1000000,  # 估计
        'inference_speed': 40,
        'accuracy': 88.5,
        'cost': 10.0,
    },
}

# ============ 分析 ============
# 1. 时间演进分析
years = sorted(set(m['release_year'] for m in models_data.values()))
models_by_year = {year: [] for year in years}
for model, data in models_data.items():
    models_by_year[data['release_year']].append(model)

# 2. 性能指标分析
model_names = list(models_data.keys())
resolutions = [models_data[m]['resolution'] for m in model_names]
accuracies = [models_data[m]['accuracy'] for m in model_names]
speeds = [models_data[m]['inference_speed'] for m in model_names]
costs = [models_data[m]['cost'] for m in model_names]
parameters = [models_data[m]['parameters'] for m in model_names]

# 3. 特性对比
multi_image_support = [models_data[m]['multi_image'] for m in model_names]
chinese_support = [models_data[m]['chinese_optimized'] for m in model_names]

# 4. 计算性价比指标
efficiency = [accuracies[i] / costs[i] for i in range(len(model_names))]
speed_accuracy_ratio = [accuracies[i] / (101 - speeds[i]) for i in range(len(model_names))]

# ============ 结果输出 ============
print("=" * 80)
print("多模态LLM模型对比分析")
print("=" * 80)
print()

print("模型时间演进:")
print("-" * 80)
for year in sorted(years):
    print(f"{year}: {', '.join(models_by_year[year])}")
print()

print("详细对比:")
print("-" * 80)
for model in model_names:
    data = models_data[model]
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
    print(f"  性价比: {efficiency[model_names.index(model)]:.2f}")

print()
print("=" * 80)

# ============ 可视化 ============
fig = plt.figure(figsize=(18, 12))
gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

# 1. 准确率演进
ax = fig.add_subplot(gs[0, 0])
years_list = [models_data[m]['release_year'] for m in model_names]
ax.scatter(years_list, accuracies, s=200, alpha=0.6, c=range(len(model_names)), cmap='viridis')
for i, model in enumerate(model_names):
    ax.annotate(model, (years_list[i], accuracies[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.plot(years_list, accuracies, 'k--', alpha=0.3)
ax.set_xlabel('Release Year')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Accuracy Evolution')
ax.grid(True, alpha=0.3)

# 2. 分辨率 vs 准确率
ax = fig.add_subplot(gs[0, 1])
scatter = ax.scatter(resolutions, accuracies, s=200, alpha=0.6, c=costs, cmap='RdYlGn_r')
for i, model in enumerate(model_names):
    ax.annotate(model, (resolutions[i], accuracies[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Resolution')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Resolution vs Accuracy')
cbar = plt.colorbar(scatter, ax=ax, label='Cost')
ax.grid(True, alpha=0.3)

# 3. 性价比对比
ax = fig.add_subplot(gs[0, 2])
colors = ['steelblue' if m != 'Qwen2.5-VL' else 'coral' for m in model_names]
bars = ax.barh(model_names, efficiency, color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Efficiency (Accuracy/Cost)')
ax.set_title('Cost-Effectiveness')
ax.grid(True, alpha=0.3, axis='x')
for i, (bar, eff) in enumerate(zip(bars, efficiency)):
    ax.text(eff, i, f' {eff:.2f}', va='center', fontsize=9)

# 4. 参数量 vs 推理速度
ax = fig.add_subplot(gs[1, 0])
scatter = ax.scatter(parameters, speeds, s=200, alpha=0.6, c=accuracies, cmap='viridis')
for i, model in enumerate(model_names):
    ax.annotate(model, (parameters[i], speeds[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Parameters (Millions)')
ax.set_ylabel('Inference Speed (relative)')
ax.set_title('Model Size vs Speed')
cbar = plt.colorbar(scatter, ax=ax, label='Accuracy')
ax.grid(True, alpha=0.3)

# 5. 特性对比（多图像和中文支持）
ax = fig.add_subplot(gs[1, 1])
x_pos = np.arange(len(model_names))
width = 0.35
bars1 = ax.bar(x_pos - width/2, multi_image_support, width, label='Multi-Image', alpha=0.7)
bars2 = ax.bar(x_pos + width/2, chinese_support, width, label='Chinese Optimized', alpha=0.7)
ax.set_ylabel('Support (1=Yes, 0=No)')
ax.set_title('Feature Support')
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names, rotation=45, ha='right')
ax.legend()
ax.set_ylim(0, 1.2)
ax.grid(True, alpha=0.3, axis='y')

# 6. 成本 vs 准确率
ax = fig.add_subplot(gs[1, 2])
scatter = ax.scatter(costs, accuracies, s=300, alpha=0.6, c=speeds, cmap='coolwarm')
for i, model in enumerate(model_names):
    ax.annotate(model, (costs[i], accuracies[i]),
               xytext=(5, 5), textcoords='offset points', fontsize=8)
ax.set_xlabel('Relative Cost')
ax.set_ylabel('Accuracy (%)')
ax.set_title('Cost vs Accuracy')
cbar = plt.colorbar(scatter, ax=ax, label='Speed')
ax.grid(True, alpha=0.3)

# 7. 雷达图 - Qwen2.5-VL vs 其他
ax = fig.add_subplot(gs[2, :], projection='polar')

# 选择几个关键模型进行对比
compare_models = ['CLIP', 'LLaVA-1.5', 'Qwen2.5-VL', 'GPT-4V']
categories = ['Accuracy', 'Resolution', 'Speed', 'Cost-Effectiveness', 'Multi-Image']
N = len(categories)

angles = np.linspace(0, 2*np.pi, N, endpoint=False).tolist()
angles += angles[:1]

# 归一化数据
max_accuracy = max(accuracies)
max_resolution = max(resolutions)
max_speed = max(speeds)
max_efficiency = max(efficiency)

colors_radar = ['steelblue', 'coral', 'lightgreen', 'gold']

for idx, model in enumerate(compare_models):
    model_idx = model_names.index(model)
    values = [
        accuracies[model_idx] / max_accuracy,
        resolutions[model_idx] / max_resolution,
        speeds[model_idx] / max_speed,
        efficiency[model_idx] / max_efficiency,
        (1 if multi_image_support[model_idx] else 0),
    ]
    values += values[:1]
    ax.plot(angles, values, 'o-', linewidth=2, label=model, color=colors_radar[idx])
    ax.fill(angles, values, alpha=0.15, color=colors_radar[idx])

ax.set_xticks(angles[:-1])
ax.set_xticklabels(categories)
ax.set_ylim(0, 1)
ax.set_title('Model Comparison (Normalized)', pad=20)
ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.1))
ax.grid(True)

plt.savefig('assets/ch07_qwen_vl_analysis.png', dpi=100, bbox_inches='tight')
print("图表已保存到: assets/ch07_qwen_vl_analysis.png")
