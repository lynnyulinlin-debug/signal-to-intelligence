# 第 8 章：LLM 工程实践与部署

## 章节概述

从科研到生产 —— 如何将 LLM 模型部署到实际应用中？

本章介绍 LLM 工程实践的关键环节：模型优化、推理加速、成本控制、性能评估，最终构建可靠的生产系统。

## 在线 Notebook

本章提供交互式运行版本，适合边看边试验成本分析、部署配置和评估流程。

- Google Colab: [打开本章 Notebook](https://colab.research.google.com/github/lynnyulinlin-debug/signal-to-intelligence/blob/main/notebooks/ch08_engineering_interactive.ipynb)
- 使用说明: [Notebook 使用方式](/signal-to-intelligence/00_introduction/05_how_to_use_this_tutorial.html)

## 核心主题

| 小节 | 主题 | 关键词 |
|------|------|--------|
| 8.1 | 模型量化与蒸馏 | PTQ、QAT、知识蒸馏 |
| 8.2 | 推理优化 | vLLM、TensorRT、Flash Attention |
| 8.3 | 成本优化 | Token 成本、推理成本、存储成本 |
| 8.4 | 评估与基准 | MMLU、MT-Bench、自定义评估 |
| 8.5 | 生产系统设计 | 单机部署、分布式、高可用 |

## 学习路径

**快速通道**（4-6h）：8.1 → 8.2 → 8.5  
**完整路径**（8-12h）：8.1 → 8.2 → 8.3 → 8.4 → 8.5  
**深度探索**（15-20h）：所有小节 + extensions

## 代码实验

- quantization_demo.py - 模型量化演示
- inference_benchmark.py - 推理性能测试
- cost_calculator.py - 成本计算工具
