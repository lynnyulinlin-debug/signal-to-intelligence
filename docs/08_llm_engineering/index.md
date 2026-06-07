<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->

# 第 8 章：LLM 工程实践与部署

## 章节概述

从科研到生产 —— 如何将 LLM 模型部署到实际应用中？

本章介绍 LLM 工程实践的关键环节：模型优化、推理加速、成本控制、性能评估，最终构建可靠的生产系统。

## 核心主题

| 小节 | 主题 | 关键词 |
|------|------|--------|
| 8.1 | 模型量化与蒸馏 | PTQ、QAT、知识蒸馏 |
| 8.2 | 推理优化 | vLLM、TensorRT、Flash Attention |
| 8.3 | 成本优化 | Token 成本、推理成本、存储成本 |
| 8.4 | 评估与基准 | MMLU、MT-Bench、自定义评估 |
| 8.5 | 生产系统设计 | 单机部署、分布式、高可用 |

## 代码实验

- [quantization_demo.py](../../code/ch08_llm_engineering/quantization_demo.py) - 模型量化演示
- [inference_benchmark.py](../../code/ch08_llm_engineering/inference_benchmark.py) - 推理性能测试
- [cost_calculator.py](../../code/ch08_llm_engineering/cost_calculator.py) - 成本计算工具

## 前置知识

- ✅ 第 5 章：LLM 原理（理解模型架构）
- ✅ 第 6 章：LLM 应用（理解应用场景）
