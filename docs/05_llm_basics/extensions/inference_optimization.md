# 扩展：推理机制优化

**所属章节：** [第5章：LLM基础](../README.md)  
**相关内容：** 工程部署层面见 [第6章 E6.x 推理部署](../../06_llm_applications/extensions/inference_deployment.md)

---

## 概览

本节从**原理层**解释 LLM 推理的核心瓶颈和优化思路。工程选型（用哪个框架、哪种量化方案）见第6章扩展。

---

## 自回归生成的瓶颈

LLM 推理是自回归的：每次生成一个 token，需要对整个已生成序列做一次完整的前向传播。

```
生成第 t 个 token 时：
  输入：[token₁, token₂, ..., token_{t-1}]
  计算：对每个 token 做自注意力（O(t²) 复杂度）
  输出：token_t 的概率分布

生成第 t+1 个 token 时：
  输入：[token₁, token₂, ..., token_t]
  重新计算所有注意力...
```

**问题：** 每次生成新 token，都要重新计算之前所有 token 的 Key 和 Value 矩阵，大量重复计算。

---

## KV Cache

**核心思想：** 缓存已计算的 Key 和 Value 矩阵，避免重复计算。

### 原理

在自注意力中，每个 token 会生成三个向量：Q（Query）、K（Key）、V（Value）。

```
注意力计算：
  Attention(Q, K, V) = softmax(QKᵀ/√d) × V

生成第 t 个 token 时：
  新 token 的 Q 需要和所有历史 token 的 K 做点积
  但历史 token 的 K、V 在之前已经计算过了！
```

**KV Cache 的做法：**

```
第 1 步：计算 token₁ 的 K₁、V₁，存入缓存
第 2 步：计算 token₂ 的 K₂、V₂，存入缓存；用 Q₂ 和 [K₁,K₂] 做注意力
第 t 步：只计算新 token 的 K_t、V_t；用 Q_t 和缓存中的 [K₁,...,K_t] 做注意力
```

**效果：** 将每步的计算复杂度从 O(t²) 降低到 O(t)，推理速度提升数倍到数十倍。

### KV Cache 的代价

缓存占用显存：

```
KV Cache 大小 = 2 × 层数 × 头数 × 头维度 × 序列长度 × batch_size × 精度字节数

示例（LLaMA-3-8B，序列长度 4096，batch=1，fp16）：
  = 2 × 32 × 32 × 128 × 4096 × 1 × 2 bytes
  ≈ 2GB
```

序列越长，KV Cache 越大。这是长上下文推理的主要显存瓶颈。

### MLA（Multi-head Latent Attention）

DeepSeek-V2 提出的改进：将 KV Cache 压缩到低维潜空间，显存占用降低 5-13 倍。

```
标准 MHA：缓存每层的完整 K、V 矩阵
MLA：缓存压缩后的潜向量 c，推理时动态解压
```

---

## Flash Attention

**问题：** 标准注意力计算需要将 O(n²) 大小的注意力矩阵写入 GPU HBM（高带宽内存），IO 成为瓶颈。

### GPU 内存层次

```
SRAM（片上缓存）：~20MB，带宽极高（~19TB/s），但容量小
HBM（显存）：~40-80GB，带宽较低（~2TB/s），容量大

标准注意力：
  计算 QKᵀ → 写入 HBM（n² 大小）
  读取 HBM → 做 softmax → 写回 HBM
  读取 HBM → 乘以 V → 写回 HBM
  大量 HBM 读写，IO 成为瓶颈
```

### Flash Attention 的思路

**分块计算（Tiling）：** 将注意力矩阵分成小块，每块在 SRAM 中完成计算，避免写入 HBM。

```
Flash Attention：
  将 Q、K、V 分成小块，逐块加载到 SRAM
  在 SRAM 中完成注意力计算（包括 softmax）
  只将最终结果写回 HBM
  
  HBM 读写次数：O(n²/M)，其中 M 是 SRAM 大小
  vs 标准注意力：O(n²)
```

**重计算（Recomputation）：** 反向传播时不存储注意力矩阵，而是重新计算。

```
标准反向传播：存储前向的注意力矩阵（O(n²) 显存）
Flash Attention：丢弃注意力矩阵，反向传播时重新计算
  代价：多一次前向计算
  收益：显存从 O(n²) 降到 O(n)
```

**效果：** 速度提升 2-4 倍，显存降低 5-20 倍（取决于序列长度）。

### Flash Attention 2 & 3

- **Flash Attention 2**：改进并行化策略，速度再提升 2 倍
- **Flash Attention 3**：针对 H100 的异步流水线优化，速度再提升 1.5-2 倍

---

## 注意力的 O(n²) 瓶颈

自注意力的计算复杂度是 O(n²)，其中 n 是序列长度。

```
序列长度 1K：注意力矩阵 1M 元素
序列长度 4K：注意力矩阵 16M 元素（增加 16 倍）
序列长度 128K：注意力矩阵 16B 元素（增加 16384 倍）
```

这是长上下文 LLM 的核心挑战。解决方案：

| 方法 | 思路 | 代表工作 |
|------|------|---------|
| Flash Attention | IO 优化，不改变复杂度 | Flash Attention 1/2/3 |
| Sliding Window | 只关注局部窗口，O(n×w) | Mistral、Longformer |
| 线性注意力 | 近似计算，O(n) | Mamba、RWKV |
| 稀疏注意力 | 只计算重要的注意力对 | BigBird、Longformer |

---

## 量化原理

量化是将模型权重从高精度（FP32/FP16）压缩到低精度（INT8/INT4）的技术。

### 精度与存储

```
FP32：4 字节/参数，精度最高
FP16/BF16：2 字节/参数，训练常用
INT8：1 字节/参数，推理常用
INT4：0.5 字节/参数，极限压缩
```

7B 模型的存储需求：
```
FP16：~14GB
INT8：~7GB
INT4：~3.5GB
```

### 量化误差的来源

```
量化过程：
  FP16 值（连续）→ 映射到 INT8 范围（离散）→ 反量化回 FP16

误差来源：
  1. 截断误差：超出量化范围的值被截断
  2. 舍入误差：连续值映射到离散格点的误差
  3. 异常值问题：少数极大值会压缩其他值的精度
```

### 主流量化方法

| 方法 | 精度 | 特点 |
|------|------|------|
| LLM.int8() | INT8 | 混合精度，异常值用 FP16 | 
| GPTQ | INT4 | 逐层量化，精度损失小 |
| AWQ | INT4 | 保护重要权重，精度更好 |
| GGUF | INT4/INT8 | llama.cpp 格式，CPU 推理 |

---

**返回：** [第5章：LLM基础](../README.md)
