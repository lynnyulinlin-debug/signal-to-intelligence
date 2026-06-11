<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->

# 第8章：LLM 工程实践与部署

## 章节概览

本章回答一个核心问题：**如何把 LLM 应用稳定、可控、可负担地上线？**

本章主线是生产闭环：**量化 / 剪枝 / 蒸馏 → 推理优化 → 成本控制 → 评估监控 → 部署与生产系统设计**。第 5 章解释模型怎么训练，第 6 章解释应用怎么设计，第 8 章负责把这些能力变成可运行、可观测、可回退的工程系统。

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 8.1 模型压缩：量化、剪枝与蒸馏 | [01_quantization_distillation.md](01_quantization_distillation.md) | ⭐⭐⭐ | 20分钟 |
| 8.2 推理优化 | [02_inference_optimization.md](02_inference_optimization.md) | ⭐⭐⭐ | 15分钟 |
| 8.3 成本优化 | [03_cost_optimization.md](03_cost_optimization.md) | ⭐⭐ | 10分钟 |
| 8.4 评估与基准 | [04_evaluation_benchmark.md](04_evaluation_benchmark.md) | ⭐⭐⭐ | 15分钟 |
| 8.5 生产系统设计 | [05_production_system.md](05_production_system.md) | ⭐⭐⭐ | 15分钟 |

## 小节目录

### 压缩与推理（8.1-8.2）

**8.1 模型压缩：量化、剪枝与蒸馏** — [📖 阅读](01_quantization_distillation.md)
- PTQ / QAT / 剪枝 / 稀疏化 / 知识蒸馏的基本思路
- 精度、延迟、显存和部署复杂度的权衡
- 压缩失败模式：精度回退、长尾任务退化、硬件不匹配、稀疏收益无法兑现

**8.2 推理优化** — [📖 阅读](02_inference_optimization.md)
- KV cache、批处理、Flash Attention、vLLM 等关键机制
- 吞吐、延迟、显存占用之间的取舍
- 在线服务中如何处理并发、长上下文和流式输出

### 成本与评估（8.3-8.4）

**8.3 成本优化** — [📖 阅读](03_cost_optimization.md)
- Token 成本、推理成本、存储成本
- 模型大小、上下文长度、缓存命中率对成本的影响
- 成本控制策略：路由、缓存、截断、降级和批处理

**8.4 评估与基准** — [📖 阅读](04_evaluation_benchmark.md)
- 通用 Benchmark 与业务自定义评估
- 线上监控：质量、延迟、成本、安全和用户反馈
- 评估闭环：上线前测试、灰度、回归集和持续监控

### 生产系统（8.5）

**8.5 生产系统设计** — [📖 阅读](05_production_system.md)
- 单机部署、分布式部署和高可用架构
- 限流、重试、回退、审计和权限控制
- 从原型到生产的上线风险和运维边界

## 核心问题

完成本章后，你应该能回答：

1. 量化、剪枝和蒸馏分别解决什么问题？代价是什么？
2. KV cache、批处理和推理框架如何影响吞吐与延迟？
3. LLM 服务的成本主要由哪些因素决定？
4. 为什么离线 Benchmark 不等于线上质量？
5. 一个生产级 LLM 系统需要哪些监控、回退和风险控制机制？

## 代码实验

| 小节 | 脚本 | 内容 |
|------|------|------|
| 8.1 | [`quantization_demo.py`](../../code/ch08_llm_engineering/quantization_demo.py) | 模型量化演示 |
| 8.2 | [`inference_benchmark.py`](../../code/ch08_llm_engineering/inference_benchmark.py) | 推理性能测试 |
| 8.3 | [`cost_calculator.py`](../../code/ch08_llm_engineering/cost_calculator.py) | 成本计算工具 |

## 扩展阅读

- [LLM Serving as Operations Research](extensions/operations_research_serving.md)：用运筹学视角理解批处理、排队、路由、缓存、容量规划和 SLO 成本优化。
- [高级量化技巧](extensions/advanced_quantization.md)：更细的量化方案与部署风险。
- [分布式推理系统](extensions/distributed_inference.md)：多卡、多机和高并发推理架构。
- [推理加速与硬件适配](extensions/hardware_acceleration_and_conversion.md)：指令集优化、GPU / NPU 并行计算和模型转换。

## 前置知识

- 第 5 章：LLM 原理，理解模型、训练和评估目标
- 第 6 章：LLM 应用，理解 Prompt / 微调 / RAG / Agent 的应用边界
- 第 7 章：多模态 LLM，理解高分辨率和多模态输入带来的额外工程成本

## 关键连接点

### 应用方案 → 工程系统

```text
第6章：Prompt / 微调 / RAG / Agent
    ↓
模型与推理方案选择
    ↓
量化、缓存、批处理、路由
    ↓
评估、监控、回退、成本控制
    ↓
生产级 LLM 系统
```
