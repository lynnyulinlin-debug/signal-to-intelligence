# 第4章扩展：注意力机制的变种

**版本：** v2.0  
**最后更新：** 2026-05-26

本文档包含第4章的深度扩展内容，介绍注意力机制的各种变种和优化。

---

## E4.1 注意力机制的数学基础

### 注意力的本质

**注意力就是加权求和：**
$$\text{Attention}(Q, K, V) = \sum_i w_i v_i$$

其中权重 $w_i$ 由查询和键的相似度决定。

### 不同的相似度函数

**点积注意力（Dot-Product Attention）：**
$$w_i = \text{softmax}\left(\frac{q \cdot k_i}{\sqrt{d_k}}\right)$$

**加法注意力（Additive Attention）：**
$$w_i = \text{softmax}(v^T \tanh(W_q q + W_k k_i))$$

**乘法注意力（Multiplicative Attention）：**
$$w_i = \text{softmax}(q^T W k_i)$$

**优点对比：**
- 点积：计算快，内存高效
- 加法：表达能力强，但计算慢
- 乘法：平衡两者

---

## E4.2 高效Transformer

### 问题：二次复杂度

标准Transformer的注意力复杂度是 $O(n^2)$，其中 $n$ 是序列长度。

**对于长序列很贵：**
- 序列长度1000：100万次操作
- 序列长度10000：1亿次操作

### 解决方案1：稀疏注意力

**思想：** 不是每个位置都关注所有位置，只关注相关位置。

**方法：**
- **局部注意力**：只关注附近的位置
- **步长注意力**：每隔k个位置关注一次
- **块对角注意力**：分块处理

**复杂度：** $O(n \log n)$ 或 $O(n)$

### 解决方案2：低秩近似

**思想：** 注意力矩阵可以用低秩矩阵近似。

**方法：**
- **Linformer**：用线性投影降维
- **Performer**：用随机特征近似softmax

**复杂度：** $O(n)$

### 解决方案3：分层注意力

**思想：** 先在局部做注意力，再在全局做注意力。

**例子：**
- **Longformer**：局部 + 全局注意力
- **BigBird**：块对角 + 全局注意力

---

## E4.3 多查询注意力（Multi-Query Attention）

### 问题

标准多头注意力中，每个头都有独立的键和值投影，导致参数多、内存占用大。

### 解决方案

**多查询注意力：** 所有头共享同一个键和值。

$$\text{MQA}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

其中所有 $\text{head}_i$ 使用同一个 $K$ 和 $V$。

### 优势

- 参数减少 $h$ 倍
- 内存占用减少
- 推理速度快
- 性能基本不变

### 应用

- **PaLM**：Google的大模型
- **Falcon**：开源大模型

---

## E4.4 因果注意力（Causal Attention）

### 问题

在生成任务中，模型不应该看到未来的token。

### 解决方案

**因果掩码：** 在计算注意力权重前，把未来位置的分数设为 $-\infty$。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}} + M\right) V$$

其中 $M_{ij} = -\infty$ 如果 $i < j$（未来位置）。

### 实现

```python
# 创建因果掩码
mask = torch.tril(torch.ones(seq_len, seq_len))
scores = scores.masked_fill(mask == 0, float('-inf'))
```

---

## E4.5 相对位置编码

### 问题

绝对位置编码有局限：
- 对于超长序列，位置编码可能超出范围
- 模型难以泛化到训练时没见过的长度

### 解决方案

**相对位置编码：** 编码位置之间的相对距离，而不是绝对位置。

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T + R}{\sqrt{d_k}}\right) V$$

其中 $R$ 是相对位置偏置。

### 优势

- 可以泛化到更长的序列
- 更符合人类的相对位置理解

### 应用

- **T5**：Google的预训练模型
- **DeBERTa**：微软的预训练模型

---

## E4.6 旋转位置编码（RoPE）

### 思想

用复数旋转编码位置。

$$PE(pos, 2i) = \cos(pos \cdot \theta_i)$$
$$PE(pos, 2i+1) = \sin(pos \cdot \theta_i)$$

其中 $\theta_i = 10000^{-2i/d}$。

### 优势

- 自然的相对位置编码
- 可以外推到更长的序列
- 计算高效

### 应用

- **LLaMA**：Meta的大模型
- **Falcon**：开源大模型

---

## E4.7 分组查询注意力（Grouped Query Attention）

### 思想

介于多头注意力和多查询注意力之间。

**多头注意力：** 每个查询头有独立的K、V  
**分组查询注意力：** 多个查询头共享一个K、V  
**多查询注意力：** 所有查询头共享一个K、V

### 优势

- 比多头注意力参数少
- 比多查询注意力表达能力强
- 性能和效率的平衡

### 应用

- **Llama 2**：Meta的改进版大模型

---

## E4.8 闪电注意力（Flash Attention）

### 问题

标准注意力实现在GPU上不高效：
- 需要多次访问内存
- 内存带宽是瓶颈

### 解决方案

**Flash Attention：** 重新组织计算，减少内存访问。

**关键思想：**
1. 分块计算注意力
2. 在GPU高速缓存中完成计算
3. 减少全局内存访问

### 性能提升

- 速度快2-4倍
- 内存占用减少
- 精度不变

### 应用

- **PyTorch 2.0**：官方支持
- **Hugging Face Transformers**：集成支持

---

## E4.9 推荐论文

### 注意力机制的经典论文

1. **Vaswani et al. (2017)** - "Attention Is All You Need"
   - Transformer的原始论文
   - 引入多头自注意力

2. **Bahdanau et al. (2014)** - "Neural Machine Translation by Jointly Learning to Align and Translate"
   - 注意力机制的开创性工作

### 高效Transformer的论文

1. **Kitaev et al. (2020)** - "Reformer: The Efficient Transformer"
   - 局部敏感哈希注意力

2. **Choromanski et al. (2020)** - "Rethinking Attention with Performers"
   - 用随机特征近似softmax

3. **Dao et al. (2022)** - "FlashAttention: Fast and Memory-Efficient Exact Attention with IO-Awareness"
   - Flash Attention的论文

### 位置编码的论文

1. **Su et al. (2021)** - "RoFormer: Enhanced Transformer with Rotary Position Embedding"
   - 旋转位置编码

2. **Shaw et al. (2018)** - "Self-Attention with Relative Position Representations"
   - 相对位置编码

---

## E4.10 进一步学习

### 书籍

- **"Attention Is All You Need" 论文详解**
  - 很多博客和教程详细讲解
  - 推荐 Jay Alammar 的可视化讲解

- **"The Illustrated Transformer"**
  - 在线免费资源
  - 非常直观的讲解

### 在线资源

- **Stanford CS224N** - NLP with Deep Learning
- **CMU 11-747** - Neural Language Models
- **Hugging Face Course** - NLP with Transformers

### 实践项目

1. **实现自注意力**
   - 从零开始实现
   - 理解每一步的计算

2. **实现多头注意力**
   - 理解多头的作用
   - 对比单头和多头的性能

3. **实现完整Transformer**
   - 编码器-解码器结构
   - 在简单任务上训练

4. **优化注意力**
   - 实现Flash Attention
   - 测试性能提升

---

**返回：** [第4章：Transformer详解](../README.md)
