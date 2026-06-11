# 第6章：LLM应用

**版本：** v3.0  
**最后更新：** 2026-05-30

## 章节概览

本章介绍如何使用和定制 LLM，回答一个核心问题：**有了预训练好的 LLM，怎么用最低风险和成本解决实际问题？**

本章主线是应用方案选型：**Prompt → 微调 → RAG → Agent → 系统组合**。每条路线都要回答同一组问题：适合什么任务、解决什么瓶颈、引入什么成本、失败时如何排查。

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 6.1 Prompt 工程 | [01_prompt_engineering.md](01_prompt_engineering.md) | ⭐⭐ | 10分钟 |
| 6.2 微调：让 LLM 适配你的任务 | [02_finetuning.md](02_finetuning.md) | ⭐⭐⭐ | 15分钟 |
| 6.3 RAG（检索增强生成） | [03_rag.md](03_rag.md) | ⭐⭐⭐ | 15分钟 |
| 6.4 Agent 框架 | [04_agent.md](04_agent.md) | ⭐⭐⭐ | 15分钟 |
| 6.5 LLM 系统设计 | [05_system_design.md](05_system_design.md) | ⭐⭐⭐ | 15分钟 |

## 小节目录

### 基础使用（6.1）

**6.1 Prompt 工程** — [📖 阅读](01_prompt_engineering.md)
- Zero-shot / Few-shot / Chain-of-Thought
- 角色设定与结构化输出
- 低成本验证、失败模式和迭代优化

### 适配方法（6.2-6.3）

**6.2 微调：让 LLM 适配你的任务** — [📖 阅读](02_finetuning.md)
- 任务微调 vs 对齐 SFT 的区别
- LoRA：低秩分解降低微调成本
- 何时用微调，何时不要微调

**6.3 RAG（检索增强生成）** — [📖 阅读](03_rag.md)
- 为什么 LLM 需要外部知识
- 检索 + 生成的工作流程
- Embedding 与向量检索
- 知识新鲜度、可解释性和检索失败模式

### 扩展能力（6.4）

**6.4 Agent 框架** — [📖 阅读](04_agent.md)
- LLM 作为规划者
- 工具调用与 ReAct 模式
- 多步任务的执行流程
- 工具可靠性、错误累积和权限边界

### 系统设计（6.5）

**6.5 LLM 系统设计** — [📖 阅读](05_system_design.md)
- 技术路线选择决策树（Prompt vs 微调 vs RAG vs Agent）
- 三种架构模式（简单问答 / RAG 问答 / Agent 工作流）
- 实战案例：客服机器人、代码助手、数据分析
- 模型选择、部署选项、成本和风险控制

## 学习时间

- **快速版**（仅阅读正文）：30分钟
- **标准版**（包含代码实验）：60分钟
- **完整版**（包含扩展内容）：90分钟

## 核心问题

完成本章后，你应该能回答：

1. 如何通过 Prompt 设计充分发挥 LLM 的能力？
2. 任务微调和对齐 SFT 有什么区别？
3. LoRA 为什么能大幅降低微调成本？
4. RAG 如何让 LLM 使用外部知识？
5. Agent 框架的核心是什么？ReAct 模式如何工作？
6. 面对一个实际任务，如何选择 Prompt / 微调 / RAG / Agent？
7. 如何根据成本、延迟、知识新鲜度、可维护性和风险选择 LLM 应用架构？

## 代码实验

本章共有 **5 个代码脚本**，覆盖 Prompt 技术对比、微调参数分析、RAG 向量检索、Agent ReAct 循环和系统设计选型框架。

| 小节 | 脚本 | 生成图表 | 内容 |
|------|------|---------|------|
| 6.1 Prompt 工程 | [`prompt_demo.py`](../../code/ch06_llm_applications/prompt_demo.py) | `ch06_prompt_techniques.png` | Zero-shot/Few-shot/CoT 对比、成本-效果权衡、迭代优化流程 |
| 6.2 微调 | [`finetuning_demo.py`](../../code/ch06_llm_applications/finetuning_demo.py) | `ch06_lora_parameters.png` | 全量微调 vs LoRA 参数量对比 |
| 6.3 RAG | [`rag_demo.py`](../../code/ch06_llm_applications/rag_demo.py) | `ch06_rag_vector_search.png` | 向量检索语义空间可视化 |
| 6.4 Agent | [`agent_demo.py`](../../code/ch06_llm_applications/agent_demo.py) | `ch06_agent_error_accumulation.png` | ReAct 循环 + 错误累积曲线 |
| 6.5 系统设计 | [`system_design_demo.py`](../../code/ch06_llm_applications/system_design_demo.py) | `ch06_system_design.png` | 四方案雷达图、决策树、成本对比 |

**运行方式：**
```bash
python code/ch06_llm_applications/prompt_demo.py
python code/ch06_llm_applications/finetuning_demo.py
python code/ch06_llm_applications/rag_demo.py
python code/ch06_llm_applications/agent_demo.py
python code/ch06_llm_applications/system_design_demo.py
```

## 推荐学习路径

### 路径1：快速入门（30分钟）
- 阅读 6.1 Prompt 工程（必读，零成本起点）
- 阅读 6.3 RAG 和 6.4 Agent 的正文
- 重点：三条路线的核心思想

### 路径2：标准学习（60分钟）
- 阅读所有正文（6.1-6.5）
- 运行 RAG 实验
- 回答"核心问题"中的 7 个问题

### 路径3：深度学习（90分钟）
- 阅读所有正文和扩展内容
- 深入理解 PEFT 高级技巧和推理部署优化
- 阅读 LoRA、RAG 原始论文

## 关键概念速查

| 技术 | 核心思想 | 适用场景 |
|------|---------|---------|
| Prompt 工程 | 设计输入指令，不改模型 | 快速验证、通用任务、低风险试错 |
| 全量微调 | 更新所有参数 | 资源充足、任务差异大、需要深度适配 |
| LoRA | 低秩适配器，只训练 0.1-1% 参数 | 资源有限、格式/风格/领域适配 |
| RAG | 检索相关文档 + 生成回答 | 知识需要更新、可追溯性要求高 |
| Agent | LLM 规划 + 工具调用 | 多步骤任务、外部系统操作、流程自动化 |

## 常见问题

**Q: 应该先学哪一节？**  
A: 先读 6.1 Prompt 工程——它是零成本的起点，也是理解其他技术"为什么需要"的基础。

**Q: LoRA 为什么有效？**  
A: 微调时模型需要学习的"变化"通常是低秩的——任务适配不需要改变所有方向，只需在少数几个方向上调整。

**Q: RAG 和微调能同时用吗？**  
A: 可以。微调让模型适配输出格式和领域风格，RAG 提供最新的外部知识，两者互补。

**Q: Agent 框架为什么容易出错？**  
A: 多步推理中每一步都可能出错，错误会累积。工具调用的可靠性依赖 LLM 的指令遵循能力。

**Q: 如何选择微调 vs RAG vs Prompt？**  
A: 先用 Prompt 验证可行性；知识新鲜度和可追溯性不足时加 RAG；输出格式、领域风格或固定行为要求严格时再考虑微调；需要外部操作时才引入 Agent。详见 [6.5 LLM 系统设计](05_system_design.md)。

## 扩展内容

### 微调工程实践 — [📖 阅读](extensions/finetuning_advanced.md)
- 数据格式（Alpaca / ShareGPT）与质量检查
- 工具选型（PEFT、LLaMA-Factory、Unsloth）
- 训练配置参考与监控指标
- LoRA 权重合并与部署

### RAG 系统深度优化 — [📖 阅读](extensions/rag_advanced.md)
- 混合检索（BM25 + 向量）与重排序
- 分块策略与父子分块
- 上下文管理与 Lost in the Middle 问题
- RAGAS 评估框架与工具选型

### Agent 框架深度 — [📖 阅读](extensions/agent_advanced.md)
- 工具定义规范与可靠性
- ReAct 模式详解与多步推理控制
- 记忆系统与主流框架选型（LangChain / AutoGen / CrewAI）

### 推理部署优化 — [📖 阅读](extensions/inference_deployment.md)
- 量化方案选型（INT8/INT4/GPTQ/AWQ/GGUF）
- 推理框架对比（vLLM、llama.cpp、Ollama、TGI）
- 连续批处理与推测解码

## 关键连接点

### 预训练模型 → 任务适配

```
通用 LLM（第5章）
    ├─ Prompt 工程 → 零成本适配
    ├─ 微调（LoRA）→ 领域专用模型
    ├─ RAG → 知识增强模型
    └─ Agent → 工具使用模型
```

### Embedding → RAG 检索

```
第5章：token embedding（语义向量）
    ↓ 同样的思想
RAG：文档 embedding → 向量检索 → 找相关文档
```

---

**下一步：** 阅读 [6.1 Prompt 工程](01_prompt_engineering.md)
