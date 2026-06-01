# 扩展：Agent 框架深度

**所属章节：** [第6章：LLM应用](../README.md)  
**前置阅读：** [6.4 Agent 框架](../04_agent.md)

---

## 概览

6.4 节介绍了 Agent 的基本概念和 ReAct 模式。本节聚焦**工程可靠性**：如何让工具调用更稳定、多步推理更可控、复杂任务更可靠。

---

## 工具定义规范

### 标准格式

工具定义的质量直接决定 LLM 能否正确调用。关键是 `description` 要精确：

```python
tools = [
    {
        "name": "search_web",
        "description": "搜索互联网获取最新信息。适用于：需要实时数据、近期事件、具体数字。不适用于：通用知识、历史事实（模型已知）。",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "搜索关键词，建议使用英文以获得更好结果"
                },
                "num_results": {
                    "type": "integer",
                    "description": "返回结果数量，默认 5，最大 10",
                    "default": 5
                }
            },
            "required": ["query"]
        }
    }
]
```

**工具描述的三要素：**
1. **做什么**：工具的功能
2. **什么时候用**：适用场景
3. **什么时候不用**：避免误调用

### 工具调用可靠性

LLM 可能调用错误的工具或传入无效参数。防御策略：

```
LLM 生成工具调用
    │
    ▼
[参数验证]（JSON Schema 校验）
    │
    ├─ 校验失败 → 返回错误信息给 LLM，要求重试
    │
    ▼
[工具执行]
    │
    ├─ 执行失败 → 返回错误信息给 LLM，最多重试 N 次
    │
    ▼
工具结果 → 返回给 LLM
```

---

## ReAct 模式详解

ReAct（Reasoning + Acting）是目前最主流的 Agent 范式：

```
用户：2024年诺贝尔物理学奖得主是谁？他们的主要贡献是什么？

Thought: 我需要查找2024年诺贝尔物理学奖的信息
Action: search_web(query="2024 Nobel Prize Physics winner")
Observation: 搜索结果显示获奖者是 John Hopfield 和 Geoffrey Hinton...

Thought: 我已经知道获奖者，现在需要了解他们的具体贡献
Action: search_web(query="Hopfield Hinton Nobel Prize contributions neural networks")
Observation: 他们因在人工神经网络领域的基础性发现而获奖...

Thought: 我现在有足够的信息来回答问题了
Final Answer: 2024年诺贝尔物理学奖授予 John Hopfield 和 Geoffrey Hinton，
表彰他们在人工神经网络领域的基础性发现...
```

**关键设计：** Thought 步骤让 LLM 显式推理"为什么"调用工具，而不是盲目调用，显著提升可靠性。

---

## 多步推理控制

### 循环终止条件

Agent 循环必须有明确的终止条件，否则可能无限循环：

```python
MAX_STEPS = 10   # 最大步数限制

for step in range(MAX_STEPS):
    response = llm.call(messages)
    
    if response.is_final_answer:
        return response.content
    
    tool_result = execute_tool(response.tool_call)
    messages.append(tool_result)

# 超过最大步数，强制返回
return "任务超过最大步数限制，当前进度：..."
```

### 错误恢复

工具调用失败时，让 LLM 自主决策如何恢复：

```
工具调用失败
    │
    ▼
将错误信息返回给 LLM：
"工具 search_web 调用失败：网络超时。
请选择：1) 重试 2) 使用其他工具 3) 基于已有信息回答"
    │
    ▼
LLM 决策下一步
```

### 并行工具调用

当多个工具调用相互独立时，并行执行可大幅降低延迟：

```python
# 串行：总时间 = t1 + t2 + t3
result1 = tool_a(...)
result2 = tool_b(...)
result3 = tool_c(...)

# 并行：总时间 = max(t1, t2, t3)
import asyncio
results = await asyncio.gather(
    tool_a(...),
    tool_b(...),
    tool_c(...)
)
```

现代 LLM API（GPT-4、Claude）支持在单次响应中返回多个并行工具调用。

---

## 记忆系统

Agent 的记忆分为四类：

| 类型 | 存储位置 | 生命周期 | 用途 |
|------|---------|---------|------|
| 工作记忆 | 上下文窗口 | 单次对话 | 当前任务的中间结果 |
| 对话历史 | 数据库 | 跨对话 | 记住用户偏好和历史 |
| 知识库 | 向量数据库 | 持久 | 领域知识（RAG） |
| 程序记忆 | 代码/工具 | 持久 | 固定的操作流程 |

**上下文窗口管理：** 长对话需要压缩历史，常见策略：
- 保留最近 N 轮
- 用 LLM 总结早期对话
- 只保留关键事实（实体、决策）

---

## 主流 Agent 框架

| 框架 | 定位 | 特点 |
|------|------|------|
| LangChain Agents | 通用 | 工具生态丰富，抽象层多 |
| LlamaIndex Agents | RAG + Agent | 与知识库集成好 |
| AutoGen | 多 Agent | 多个 Agent 协作对话 |
| CrewAI | 多 Agent | 角色分工，适合复杂任务 |
| Semantic Kernel | 企业级 | 微软出品，.NET/Python |

**选择建议：**
- 单 Agent + 工具调用 → LangChain 或直接用 API 的 function calling
- 需要多个 Agent 协作 → AutoGen 或 CrewAI
- 与知识库深度集成 → LlamaIndex

---

## 成本与延迟控制

Agent 的主要成本来源是多轮 LLM 调用：

```
单次 Agent 任务成本 ≈ 平均步数 × 每步 token 数 × token 单价

示例（GPT-4o，平均 5 步，每步 2000 tokens）：
  5 × 2000 × $0.005/1K = $0.05/次
  日均 1000 次 → $50/天
```

**成本优化策略：**
- 用小模型处理简单步骤（工具选择），大模型处理复杂推理
- 缓存工具调用结果（相同参数的重复调用）
- 设置合理的最大步数限制

---

## 推荐论文

- **Yao et al. (2022)** — "ReAct: Synergizing Reasoning and Acting in Language Models"（ReAct 框架）
- **Schick et al. (2023)** — "Toolformer: Language Models Can Teach Themselves to Use Tools"
- **Wang et al. (2023)** — "Voyager: An Open-Ended Embodied Agent with Large Language Models"
- **Wu et al. (2023)** — "AutoGen: Enabling Next-Gen LLM Applications via Multi-Agent Conversation"

---

**返回：** [第6章：LLM应用](../README.md)
