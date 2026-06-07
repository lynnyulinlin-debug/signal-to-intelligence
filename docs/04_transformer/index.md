<!-- AUTO-GENERATED from README.md. Do not edit index.md directly. -->

# 第4章：Transformer详解

**版本：** v3.1
**最后更新：** 2026-05-30

## 章节概览

本章介绍 Transformer 的核心机制，重点回答一个问题：**为什么注意力、位置编码和层级堆叠的组合，能成为现代LLM的基础架构？**

这一章既承接前面关于矩阵分解、深度学习和序列建模的内容，也是进入后续 LLM 与多模态章节前最关键的一步。

本章重点关注：
- 自注意力如何动态聚合序列中的信息
- 多头注意力为什么比单头更强
- 为什么位置必须显式注入，以及它与傅里叶思想的联系
- Transformer 的各个组件为什么这样组合
- 为什么 decoder-only 结构最终成为现代 LLM 的主干

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 4.1 自注意力机制 | [01_attention.md](01_attention.md) | ⭐⭐⭐ | 15分钟 |
| 4.2 多头注意力 | [02_multihead.md](02_multihead.md) | ⭐⭐⭐ | 15分钟 |
| 4.3 位置编码 | [03_positional_encoding.md](03_positional_encoding.md) | ⭐⭐⭐ | 15分钟 |
| 4.4 完整架构 | [04_architecture.md](04_architecture.md) | ⭐⭐⭐⭐ | 20分钟 |

## 小节目录

**4.1 自注意力机制** — [📖 阅读](01_attention.md)
- 注意力如何计算 token 之间的相关性
- 为什么需要 Q / K / V 投影
- 为什么要除以 `sqrt(d_k)`
- mask 如何改变信息流方向

**4.2 多头注意力** — [📖 阅读](02_multihead.md)
- 为什么单一注意力模式不够
- 多头如何在不同子空间学习不同关系
- 多头的表达能力与计算代价权衡

**4.3 位置编码** — [📖 阅读](03_positional_encoding.md)
- 为什么注意力本身不知道顺序
- 正弦位置编码与傅里叶思想的联系
- 绝对位置、相对位置与现代变体的直觉

**4.4 完整架构** — [📖 阅读](04_architecture.md)
- 残差连接、LayerNorm、FFN 的设计动机
- encoder-only / encoder-decoder / decoder-only 的差异
- 为什么 decoder-only 架构更适合生成式 LLM

### 预备与补充材料

**图论基础（可选）** — [📖 阅读](extensions/graph_theory_basics.md)
- 图、邻接关系与信息传播视角
- 帮助理解 attention 中的全连接依赖结构

**向量空间基础（可选）** — [📖 阅读](extensions/vector_space_basics.md)
- 向量、内积、投影与线性变换
- 帮助理解 Q / K / V、相似度计算与表示空间

## 章节逻辑导图

```text
序列建模问题
    ↓
RNN：递推处理，难并行，长依赖困难
    ↓
Self-Attention：直接建模序列位置之间的关系
    ↓
Multi-Head：同时建模多种关系
    ↓
Position Encoding：补上顺序信息
    ↓
Transformer Block：Attention + FFN + Residual + Norm
    ↓
Decoder-only Transformer
    ↓
LLM：大规模预训练的生成式序列模型
```

## 学习时间

- **快速版**（仅阅读正文）：25分钟
- **标准版**（包含代码实验）：60分钟
- **深度版**（包含所有原理与扩展理解）：90分钟

## 核心问题

完成本章后，你应该能回答：

### 原理理解部分
1. 自注意力机制为什么能比RNN更好地处理长距离依赖？
2. 为什么自注意力里需要 Q / K / V 三组投影，而不是直接做一次相似度计算？
3. 为什么注意力分数要除以 `sqrt(d_k)`？
4. 多头注意力为什么不是简单重复，而是带来更强表达能力？
5. 为什么位置编码是 Transformer 必不可少的一部分？
6. 残差连接、LayerNorm、FFN 在架构里分别起什么作用？
7. encoder-only、encoder-decoder、decoder-only 有什么本质差异？
8. 为什么现代 LLM 大多采用 decoder-only Transformer？

### 应用实践部分
9. 如何用代码实验直观理解注意力权重、mask 和位置编码？
10. 第4章的内容如何直接过渡到后续的 LLM 与多模态章节？

## 代码实验

本章共有 **5 个代码脚本**，生成 **6 张图片**，覆盖注意力机制到完整架构的核心思想。

| 小节 | 脚本 | 生成图片 | 文档位置 |
|------|------|---------|---------|
| 4.1 自注意力 | [`self_attention.py`](../../code/ch04_transformer/self_attention.py) | `ch04_self_attention.png` | [4.1](01_attention.md) / [README](README.md) |
| 4.1 缩放注意力 | [`scaled_attention_demo.py`](../../code/ch04_transformer/scaled_attention_demo.py) | `ch04_scaled_attention.png` | [4.1](01_attention.md) / [README](README.md) |
| 4.3 位置编码 | [`positional_encoding.py`](../../code/ch04_transformer/positional_encoding.py) | `ch04_positional_encoding.png` | [4.3](03_positional_encoding.md) / [README](README.md) |
| 4.4 因果掩码 | [`causal_mask_demo.py`](../../code/ch04_transformer/causal_mask_demo.py) | `ch04_causal_mask.png` | [4.4](04_architecture.md) / [README](README.md) |
| 扩展：图论 | [`graph_theory_demo.py`](../../code/ch04_transformer/graph_theory_demo.py) | `ch04_graph_theory.png` `ch04_attention_graph.png` | [extensions](extensions/graph_theory_basics.md) |

### 实验1：自注意力机制
- **文件：** [`code/ch04_transformer/self_attention.py`](../../code/ch04_transformer/self_attention.py)
- **内容：** 用 NumPy 实现自注意力并可视化注意力权重
- **运行：** `python code/ch04_transformer/self_attention.py`
- **输出：** 注意力权重热力图、权重分布、注意力模式分析

![Self Attention](/assets/ch04_self_attention.png)

*图4.1：自注意力矩阵示意。展示每个 token 如何根据相关性聚合其他位置的信息。*

**代码文件：** [`code/ch04_transformer/self_attention.py`](../../code/ch04_transformer/self_attention.py)
**运行方式：** `python code/ch04_transformer/self_attention.py`

### 实验2：位置编码
- **文件：** [`code/ch04_transformer/positional_encoding.py`](../../code/ch04_transformer/positional_encoding.py)
- **内容：** 生成 Transformer 位置编码，展示周期性和频率特性
- **运行：** `python code/ch04_transformer/positional_encoding.py`
- **输出：** 位置编码热力图、周期性分析、相邻位置相似度

![Positional Encoding](/assets/ch04_positional_encoding.png)

*图4.2：位置编码热力图与周期性结构。展示不同频率如何共同表示序列中的位置。*

**代码文件：** [`code/ch04_transformer/positional_encoding.py`](../../code/ch04_transformer/positional_encoding.py)
**运行方式：** `python code/ch04_transformer/positional_encoding.py`

### 实验3：缩放点积注意力
- **文件：** [`code/ch04_transformer/scaled_attention_demo.py`](../../code/ch04_transformer/scaled_attention_demo.py)
- **内容：** 展示为什么注意力分数需要除以 `sqrt(d_k)`
- **运行：** `python code/ch04_transformer/scaled_attention_demo.py`
- **输出：** 原始/缩放分数分布、标准差对比、softmax尖锐度变化

![Scaled Attention](/assets/ch04_scaled_attention.png)

*图4.3：缩放前后注意力分数分布对比。展示维度增大时，未缩放点积如何让 softmax 变尖。*

**代码文件：** [`code/ch04_transformer/scaled_attention_demo.py`](../../code/ch04_transformer/scaled_attention_demo.py)
**运行方式：** `python code/ch04_transformer/scaled_attention_demo.py`

### 实验4：因果掩码
- **文件：** [`code/ch04_transformer/causal_mask_demo.py`](../../code/ch04_transformer/causal_mask_demo.py)
- **内容：** 对比双向注意力和 causal mask 对信息流的影响
- **运行：** `python code/ch04_transformer/causal_mask_demo.py`
- **输出：** 原始分数矩阵、双向注意力、因果掩码、因果注意力热力图

![Causal Mask](/assets/ch04_causal_mask.png)

*图4.4：因果掩码如何限制注意力只能看到当前位置及其之前内容，这是 decoder-only LLM 的关键机制。*

**代码文件：** [`code/ch04_transformer/causal_mask_demo.py`](../../code/ch04_transformer/causal_mask_demo.py)
**运行方式：** `python code/ch04_transformer/causal_mask_demo.py`

## 推荐学习路径

### 路径1：快速入门（25分钟）
- 阅读 4.1-4.4 的正文
- 查看图表和核心公式
- 理解 Transformer 相比 RNN 的关键工程优势

### 路径2：标准学习（60分钟）
- 阅读所有内容
- 运行注意力与位置编码实验
- 理解多头、位置和架构如何协同工作

### 路径3：深度学习（90分钟）
- 阅读所有内容和补充材料
- 运行全部代码实验
- 对比双向与因果注意力
- 回答”核心问题”中的10个问题

## 关键概念速查

| 概念 | 核心思想 | 典型作用 |
|------|----------|---------|
| Self-Attention | 根据相似度动态聚合全局上下文 | 建模长距离依赖 |
| Q / K / V | 把“查询”“匹配”“内容”分离 | 学习更灵活的关系表示 |
| Scaled Dot-Product | 对注意力分数做缩放稳定 softmax | 提升训练稳定性 |
| Multi-Head Attention | 在多个子空间并行学习关系 | 捕捉多种依赖模式 |
| Positional Encoding | 显式注入顺序信息 | 让序列顺序可被利用 |
| Residual + LayerNorm | 稳定深层训练与信号传播 | 提升可训练性 |
| FFN | 对每个 token 做非线性变换 | 增强表示能力 |
| Decoder-only | 只保留自回归生成路径 | 构成现代 LLM 主干 |

## 常见问题

**Q: 为什么自注意力比RNN更适合长序列依赖？**
A: 因为自注意力允许任意两个位置直接交互，而RNN必须通过逐步递推传递信息，路径更长，也更难并行训练。

**Q: 为什么注意力分数要除以 `sqrt(d_k)`？**
A: 因为维度变大时点积幅值会变大，softmax 容易过于尖锐，缩放可以让分布更稳定、梯度更健康。

**Q: 为什么 Transformer 还需要位置编码？**
A: 因为注意力本身只比较内容相似度，不天然知道谁在前谁在后。位置编码负责把顺序信息注入表示中。

**Q: 多头注意力到底带来了什么？**
A: 它允许模型在不同表示子空间里同时学习不同关系，比如局部依赖、长距离依赖、语法关系或对齐关系。

**Q: 为什么现代 LLM 更常用 decoder-only？**
A: 因为它天然适合 next-token prediction，自回归训练目标简单统一，也更适合扩展到大规模文本生成任务。

## 扩展内容

### 注意力机制变种 — [📖 阅读](extensions/attention_variants.md)
- 高效 Transformer（线性注意力、稀疏注意力）
- 多查询注意力（MQA）与分组查询注意力（GQA）
- 旋转位置编码（RoPE）与相对位置编码
- Flash Attention：IO 复杂度优化原理
- 推荐论文（Transformer、BERT、GPT、RoPE、Flash Attention）

### 向量空间基础 — [📖 阅读](extensions/vector_space_basics.md)
- 向量空间、内积、范数、正交性
- 投影与线性变换
- 距离度量（余弦相似度、欧氏距离）
- 向量空间在 Transformer 中的应用（QKV 投影、Embedding 空间）

### 图论基础 — [📖 阅读](extensions/graph_theory_basics.md)
- 图的基本概念与遍历算法
- 图神经网络（GNN）与注意力的关系
- Transformer 作为全连接图的视角
- 与后续章节的连接（知识图谱、Agent 工具调用图）

## 关键连接点

### 向量空间 → 注意力打分

```text
向量内积：衡量两个方向是否相近
                    ↓
QK^T：衡量 token 之间是否相关
```

**启示：** 注意力不是魔法，可以用线性代数来理解“该关注谁”。

### 傅里叶思想 → 位置编码

```text
傅里叶变换：用不同频率的正弦波表示信号
                    ↓
位置编码：用不同频率的正弦波表示位置
```

**启示：** 位置编码不是随意设计的，可以理解为用频率结构表示顺序。

### RNN → Transformer

```text
RNN：逐步递推记忆
    ↓
Transformer：直接全局建模依赖
```

**启示：** Transformer 的核心突破，不是“去掉循环”本身，而是把依赖建模从串行传播改成了并行关联。

### Transformer → LLM

```text
Transformer：通用序列建模架构
                    ↓
decoder-only Transformer：自回归生成
                    ↓
LLM：大规模预训练的生成式模型
```

**启示：** LLM并不是独立的新范式，而是 Transformer 在生成任务和规模化训练上的自然延伸。

---

**下一步：** 阅读 [4.1 自注意力机制](01_attention.md)
