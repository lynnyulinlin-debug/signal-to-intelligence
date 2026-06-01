# 第5章：LLM基础

**版本：** v3.0  
**最后更新：** 2026-05-30

## 章节概览

本章回答一个核心问题：**LLM 是怎么训练出来的？**

从 Tokenization 到预训练，从训练数据到模型家族，从微调到强化学习对齐，再到评估——这条训练流程闭环是理解现代 LLM 的主线。

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 5.1 预训练 | [01_pretraining.md](01_pretraining.md) | ⭐⭐ | 15分钟 |
| 5.2 训练数据 | [02_training_data.md](02_training_data.md) | ⭐⭐ | 10分钟 |
| 5.3 主流模型家族 | [03_model_families.md](03_model_families.md) | ⭐⭐ | 10分钟 |
| 5.4 微调 | [04_finetuning.md](04_finetuning.md) | ⭐⭐⭐ | 15分钟 |
| 5.5 强化学习对齐 | [05_rl_alignment.md](05_rl_alignment.md) | ⭐⭐⭐ | 15分钟 |
| 5.6 模型评估 | [06_evaluation.md](06_evaluation.md) | ⭐⭐⭐ | 10分钟 |

## 小节目录

### 训练流程（5.1-5.2）

**5.1 预训练** — [📖 阅读](01_pretraining.md)
- Tokenization 与 BPE：文字如何变成数字
- Token Embedding：从离散符号到连续向量
- Decoder-only 架构为什么赢了
- 预训练目标：下一词预测
- Scaling Laws 与涌现能力
- GPT 系列演化

**5.2 训练数据** — [📖 阅读](02_training_data.md)
- 数据来源与流行数据集（Common Crawl、The Pile、Alpaca、HH-RLHF）
- 数据清洗：去重、质量过滤、有害内容过滤
- 数据配比对模型能力的影响
- 数据质量 > 数据数量（Phi 系列的启示）

### 模型与适配（5.3-5.4）

**5.3 主流模型家族** — [📖 阅读](03_model_families.md)
- GPT 系列演化（GPT-1 → GPT-4 → o1）
- LLaMA 系列（开源生态的基石）
- Qwen 系列（中文场景首选）
- Mistral、DeepSeek、Claude、Gemini
- 开源 vs 闭源的权衡

**5.4 微调** — [📖 阅读](04_finetuning.md)
- SFT：指令微调的原理
- LoRA：低秩分解降低微调成本
- 如何选择基础模型（选型框架）
- 对齐 SFT vs 任务微调的区别

### 对齐与评估（5.5-5.6）

**5.5 强化学习对齐** — [📖 阅读](05_rl_alignment.md)
- 为什么预训练模型不能直接用
- 奖励模型：量化人类偏好
- RLHF：PPO 优化奖励
- DPO：更简单的对齐方法

**5.6 模型评估** — [📖 阅读](06_evaluation.md)
- 为什么评估 LLM 很难
- 主流 Benchmark（MMLU、HumanEval、GSM8K、MT-Bench）
- 人工评估与 Chatbot Arena
- Goodhart's Law：Benchmark 刷分问题

## 学习时间

- **快速版**（仅阅读正文）：40分钟
- **标准版**（包含代码实验）：75分钟
- **完整版**（包含扩展内容）：120分钟

## 核心问题

完成本章后，你应该能回答：

1. Tokenization 解决了什么问题？BPE 的核心思想是什么？
2. 预训练如何让模型从无标注文本中学到知识？
3. Scaling Laws 告诉我们什么？Chinchilla 定律的实践意义？
4. 训练数据的质量和配比如何影响模型能力？
5. GPT 系列和 LLaMA 系列各自的演化主线是什么？
6. SFT 学到的是知识还是行为模式？
7. LoRA 为什么能大幅降低微调成本？
8. RLHF 和 DPO 有什么区别？
9. 如何评估一个 LLM？Benchmark 的局限是什么？

## 代码实验

本章共有 **9 个代码脚本**，覆盖 BPE 分词、Scaling Laws、自回归生成、训练数据分析、模型家族演化、LoRA 可视化、RLHF 流程、评估基准对比，以及 LLM API 调用。

| 小节 | 脚本 | 内容 |
|------|------|------|
| 5.1 | [`bpe_tokenization.py`](../../code/ch05_llm_basics/bpe_tokenization.py) | BPE 算法实现，展示词表合并过程 |
| 5.1 | [`scaling_laws.py`](../../code/ch05_llm_basics/scaling_laws.py) | 幂律拟合（真实数据），涌现能力可视化 |
| 5.1 | [`autoregressive_generation.py`](../../code/ch05_llm_basics/autoregressive_generation.py) | 5种采样策略对比（temperature/top-k/top-p） |
| 5.2 | [`training_data_composition.py`](../../code/ch05_llm_basics/training_data_composition.py) | 数据构成饼图、各阶段数据量、Phi 效率对比 |
| 5.3 | [`model_families_evolution.py`](../../code/ch05_llm_basics/model_families_evolution.py) | 模型家族时间线气泡图、开源 vs 闭源对比 |
| 5.4 | [`lora_visualization.py`](../../code/ch05_llm_basics/lora_visualization.py) | LoRA 矩阵分解原理、rank vs 误差、参数量对比 |
| 5.5 | [`rlhf_pipeline.py`](../../code/ch05_llm_basics/rlhf_pipeline.py) | 对齐前后奖励分布、PPO 训练曲线、RLHF vs DPO |
| 5.6 | [`benchmark_comparison.py`](../../code/ch05_llm_basics/benchmark_comparison.py) | 主流模型基准对比、Chatbot Arena ELO 排行 |
| 5.1-5.5 | [`llm_api_demo.py`](../../code/ch05_llm_basics/llm_api_demo.py) | API 调用、ICL 演示、Prompt 对比（需 API Key） |

## 推荐学习路径

### 路径1：快速入门（40分钟）
- 阅读 5.1-5.3 的正文
- 理解预训练 → 数据 → 模型家族的主线
- 重点：Scaling Laws、LLaMA vs GPT 的区别

### 路径2：标准学习（75分钟）
- 阅读所有正文（5.1-5.6）
- 运行 BPE、Scaling Laws、LoRA 实验
- 回答"核心问题"中的 9 个问题

### 路径3：深度学习（120分钟）
- 阅读所有正文和扩展内容
- 深入理解 RLHF/DPO 的数学细节
- 阅读 GPT-3、LLaMA、Chinchilla 原始论文

## 关键概念速查

| 概念 | 核心思想 | 章节 |
|------|---------|------|
| BPE | 字节对编码，在字符和单词之间取平衡 | 5.1 |
| Decoder-only | 因果掩码 + 自回归生成，训练目标统一 | 5.1 |
| Scaling Laws | 性能 ∝ 模型大小/数据量/计算量的幂律关系 | 5.1 |
| Chinchilla 定律 | 最优数据量 ≈ 20 × 参数量 | 5.1 |
| 涌现能力 | 规模超过阈值后突然出现的新能力 | 5.1 |
| 数据配比 | 不同来源数据的比例影响模型能力 | 5.2 |
| SFT | 用指令-回答对微调，调整行为模式 | 5.4 |
| LoRA | 低秩适配器，只训练 0.1-1% 参数 | 5.4 |
| RLHF | 奖励模型 + PPO，用人类偏好优化模型 | 5.5 |
| DPO | 直接从偏好数据训练，不需要单独奖励模型 | 5.5 |

## 常见问题

**Q: Tokenization 为什么不直接用字符或单词？**  
A: 字符级序列太长（注意力计算代价 O(n²)）；单词级词表太大且无法处理新词。BPE 在两者之间取平衡。

**Q: 预训练模型和微调模型有什么区别？**  
A: 预训练模型学到通用语言能力，微调模型在此基础上适配特定任务。对齐（SFT/RLHF）是特殊的微调，目的是改变行为模式而非适配任务。

**Q: 开源模型和闭源模型哪个更好？**  
A: 取决于需求。闭源模型（GPT-4、Claude）能力更强；开源模型（LLaMA、Qwen）可控性更高、数据不出本地、可以微调。

**Q: RLHF 和 DPO 哪个更好？**  
A: DPO 更简单（不需要单独训练奖励模型），但 RLHF 在某些场景下效果更好。实践中两者都在使用，DPO 因为简单而越来越流行。

**Q: Benchmark 分数高就代表模型好吗？**  
A: 不一定。Benchmark 可能被刷分（数据污染、针对性优化）。需要结合多个 Benchmark 和人工评估（Chatbot Arena）综合判断。

## 扩展内容

### LLM 训练深度细节 — [📖 阅读](extensions/llm_training_details.md)
- 预训练的具体过程（数据准备、分词、训练配置）
- Scaling Laws 的数学基础
- 涌现能力的机制与 LLM 的局限

### 对齐训练深度细节 — [📖 阅读](extensions/alignment_details.md)
- PPO 在 RLHF 中的 4 步循环与 clip 操作
- KL 散度约束的数学含义与 β 系数
- DPO 完整数学推导（Bradley-Terry 模型）
- GRPO 与 o1 风格推理强化学习

### 微调方法调研 — [📖 阅读](extensions/finetuning_survey.md)
- LoRA 变体（QLoRA、AdaLoRA、DoRA）
- Prefix Tuning、Adapter、Prompt Tuning
- 各方法对比表与选择指南

### 评估方法调研 — [📖 阅读](extensions/evaluation_survey.md)
- 主流 Benchmark 详解
- 人工评估流程与 Chatbot Arena 机制
- 中文评估（C-Eval、CMMLU）

### 推理机制优化 — [📖 阅读](extensions/inference_optimization.md)
- KV Cache：为什么自回归生成需要缓存
- Flash Attention：IO 复杂度分析
- 量化原理（INT8/INT4）

### 信息论基础 — [📖 阅读](extensions/information_theory_basics.md)
- 熵、交叉熵、KL 散度的数学基础
- 与 LLM 训练目标的联系

## 关键连接点

### Transformer → LLM 训练流程

```
第4章：Transformer 架构
    ↓ decoder-only + 因果掩码
预训练（5.1）：在海量文本上预测下一词
    ↓ 大规模数据（5.2）+ 大规模计算
基础模型（5.3）：GPT/LLaMA/Qwen 等
    ↓ SFT + LoRA（5.4）
对齐后的模型
    ↓ RLHF/DPO（5.5）
可用的 LLM 助手（ChatGPT/Claude）
    ↓ 评估（5.6）
验证效果，指导下一轮训练
```

---

**下一章：** [第6章：LLM应用](../06_llm_applications/README.md)
