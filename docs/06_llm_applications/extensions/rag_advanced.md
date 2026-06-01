# 扩展：RAG 系统深度优化

**所属章节：** [第6章：LLM应用](../README.md)  
**前置阅读：** [6.3 RAG（检索增强生成）](../03_rag.md)

---

## 概览

6.3 节介绍了 RAG 的基本工作流程。本节聚焦**工程质量**：如何让检索更准、上下文更精、系统更可靠。

---

## 检索质量优化

### 问题：语义检索的局限

纯向量检索（dense retrieval）擅长语义相似，但对精确关键词匹配效果差。例如搜索"GPT-4 发布日期"，向量检索可能返回语义相关但日期不准确的文档。

### 混合检索（Hybrid Retrieval）

结合关键词检索（BM25）和语义检索（向量），取两者的并集后重排序：

```
用户查询
    │
    ├──→ BM25 关键词检索 → Top-K 文档
    │
    ├──→ 向量语义检索   → Top-K 文档
    │
    ▼
[融合排序]（RRF 或加权融合）
    │
    ▼
最终 Top-K 文档
```

**RRF（Reciprocal Rank Fusion）融合公式：**

```
score(d) = Σ 1 / (k + rank_i(d))
```

其中 k=60 是平滑参数，rank_i(d) 是文档 d 在第 i 个检索结果中的排名。

### 重排序（Reranking）

初步检索召回候选文档后，用更强的模型精排：

```
初步检索（快，召回 Top-50）
    │
    ▼
Cross-Encoder 重排序（慢，精排 Top-5）
    │
    ▼
最终送入 LLM 的文档
```

**Bi-Encoder vs Cross-Encoder：**

| 方式 | 速度 | 精度 | 用途 |
|------|------|------|------|
| Bi-Encoder（向量检索） | 快（毫秒级） | 中 | 初步召回 |
| Cross-Encoder（重排序） | 慢（百毫秒级） | 高 | 精排 Top-K |

Cross-Encoder 同时看查询和文档，能捕捉细粒度的相关性，但不能预计算向量，只能用于精排。

---

## 文档处理优化

### 分块策略（Chunking）

分块大小直接影响检索质量：

| 策略 | 块大小 | 适用场景 |
|------|--------|---------|
| 固定大小 | 256-512 tokens | 通用场景，简单 |
| 句子级 | 1-3 句 | 问答，精确匹配 |
| 段落级 | 1-2 段 | 长文档，保留上下文 |
| 语义分块 | 动态 | 结构化文档（论文、合同） |

**父子分块（Parent-Child Chunking）：** 用小块检索（精确），返回大块给 LLM（上下文完整）：

```
存储：大块（512 tokens）→ 拆分为小块（128 tokens）→ 小块建索引
检索：查询匹配小块 → 返回对应的大块给 LLM
```

### 元数据过滤

在向量检索前用元数据缩小范围，大幅提升效率和精度：

```python
# 示例：只检索 2024 年后的文档，且来源为官方文档
results = vector_db.query(
    query_vector=embed(query),
    filter={"date": {"$gte": "2024-01-01"}, "source": "official_docs"},
    top_k=10
)
```

---

## 上下文管理

### 上下文窗口限制

检索到的文档片段会填充 LLM 的上下文窗口。典型分配：

```
总上下文（128K tokens）
    ├─ System Prompt：~500 tokens
    ├─ 对话历史：~2000 tokens
    ├─ 检索文档：~4000-8000 tokens  ← 主要消耗
    └─ 用户输入 + 输出预留：~1000 tokens
```

### 文档压缩

检索到文档后，先用小模型提取关键信息，再送给大模型：

```
检索文档（长）
    │
    ▼
[压缩模型]（提取与查询相关的句子）
    │
    ▼
压缩后的文档（短）→ 送入 LLM
```

**效果：** 可将上下文长度减少 60-80%，同时保留关键信息。

### Lost in the Middle 问题

研究发现 LLM 对上下文中间位置的信息注意力最弱。缓解方法：
- 将最相关的文档放在上下文的**开头或结尾**
- 控制检索文档数量（3-5 篇通常优于 10+ 篇）

---

## RAG 评估

### 三个核心指标

| 指标 | 含义 | 评估方式 |
|------|------|---------|
| 检索召回率 | 相关文档是否被检索到 | 人工标注 + 自动计算 |
| 答案忠实度 | 答案是否基于检索内容 | LLM-as-Judge |
| 答案相关性 | 答案是否回答了问题 | LLM-as-Judge |

### RAGAS 框架

开源 RAG 评估框架，自动计算上述指标：

```python
from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy, context_recall

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_recall]
)
```

---

## 主流工具选型

| 工具 | 定位 | 特点 |
|------|------|------|
| LangChain | 通用 LLM 框架 | 生态丰富，抽象层多 |
| LlamaIndex | 专注 RAG | 数据连接器完善，适合文档问答 |
| Haystack | 企业级 RAG | 生产就绪，支持混合检索 |
| Chroma | 向量数据库 | 轻量，适合本地开发 |
| Qdrant | 向量数据库 | 高性能，支持元数据过滤 |
| Weaviate | 向量数据库 | 支持混合检索，GraphQL API |

---

## 推荐论文

- **Lewis et al. (2020)** — "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks"（RAG 原始论文）
- **Karpukhin et al. (2020)** — "Dense Passage Retrieval for Open-Domain Question Answering"（DPR）
- **Ma et al. (2023)** — "Query Rewriting for Retrieval-Augmented Large Language Models"（查询改写）
- **Liu et al. (2023)** — "Lost in the Middle: How Language Models Use Long Contexts"（上下文位置效应）

---

**返回：** [第6章：LLM应用](../README.md)
