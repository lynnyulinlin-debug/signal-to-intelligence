# 第5章扩展：LLM训练的深度细节

**版本：** v2.0  
**最后更新：** 2026-05-26

本文档包含第5章的深度扩展内容，介绍LLM训练的技术细节。

---

## E5.1 预训练的具体过程

### 数据准备

**数据来源：**
- 网络文本（Common Crawl）
- 书籍（Books3）
- 代码（GitHub）
- 学术论文（arXiv）

**数据清理：**
1. 去重
2. 去除低质量文本
3. 去除个人信息
4. 语言过滤

**数据量：**
- GPT-3：300B tokens
- Chinchilla：1.4T tokens
- LLaMA：1.4T tokens

### 分词（Tokenization）

**目的：** 把文本转换为token序列。

**常见方法：**
- BPE（Byte Pair Encoding）
- WordPiece
- SentencePiece

**词表大小：**
- GPT-2：50K tokens
- GPT-3：50K tokens
- LLaMA：32K tokens

### 预训练目标

**因果语言模型（Causal Language Modeling）：**
$$L = -\sum_{t=1}^{T} \log P(w_t | w_1, ..., w_{t-1})$$

**优点：**
- 自然的生成任务
- 无需标注数据
- 学习因果关系

### 训练配置

**典型配置：**
- 优化器：AdamW
- 学习率：1e-4 到 1e-3
- 预热步数：总步数的1-10%
- 学习率衰减：余弦衰减
- 批大小：1000-4000
- 梯度累积：多个小批次累积

---

## E5.2 Scaling Laws的数学基础

### 幂律关系

**模型大小与性能：**
$$L(N) = aN^{-\alpha}$$

**数据量与性能：**
$$L(D) = bD^{-\beta}$$

**计算量与性能：**
$$L(C) = cC^{-\gamma}$$

其中 $\alpha \approx \beta \approx \gamma \approx 0.07$。

### Chinchilla缩放律

**最优配置：**
- 计算量应该平均分配给模型大小和数据量
- 最优的数据量 ≈ 20倍的参数量

**公式：**
$$N_{\text{opt}} = \frac{C}{6D_{\text{opt}}}$$
$$D_{\text{opt}} = \frac{C}{6N_{\text{opt}}}$$

其中 $C$ 是总计算量。

### 计算量的定义

**FLOPs（浮点运算数）：**
- 前向传播：$2NTB$
- 反向传播：$4NTB$
- 总计：$6NTB$

其中：
- $N$：模型参数数
- $T$：序列长度
- $B$：批大小

---

## E5.3 涌现能力的机制

### In-Context Learning的原理

**假设1：隐状态编码**
- 模型在隐状态中编码示例的模式
- 在推理时应用这个模式

**假设2：梯度下降的类比**
- In-Context Learning类似于快速的梯度下降
- 模型在推理时"学习"新任务

### Chain-of-Thought的原理

**为什么有效：**
1. 分解复杂问题为简单步骤
2. 每一步都更容易预测
3. 减少推理错误

**例子：**
```
问题：15 + 27 = ?

直接预测：困难（需要计算）
一步步推理：容易（每步都是简单的加法）
```

### 其他涌现能力

**代码生成：**
- 小模型：无法生成正确的代码
- 大模型：能生成相对复杂的代码

**多语言能力：**
- 小模型：主要是英文
- 大模型：能处理多种语言

**推理能力：**
- 小模型：无法进行多步推理
- 大模型：能进行复杂的逻辑推理

---

## E5.4 Prompt工程的高级技巧

### 思维链（Chain-of-Thought）

**基本形式：**
```
问题：...
让我一步步思考：
1. ...
2. ...
3. ...
答案：...
```

**变种：**
- **自洽性（Self-Consistency）**：生成多个推理路径，投票选择
- **思维树（Tree of Thought）**：探索多个推理分支

### 角色扮演（Role-Playing）

**效果：**
- 让模型采用特定的"人设"
- 改变输出的风格和质量

**例子：**
```
你是一个资深的机器学习工程师，有10年的经验。
请解释什么是Transformer。
```

### 上下文学习的优化

**示例选择：**
- 选择与输入相似的示例
- 选择多样化的示例
- 示例的顺序很重要

**示例数量：**
- 0-shot：无示例
- 1-shot：1个示例
- Few-shot：3-5个示例
- 通常5个示例后效果不再显著提升

### 指令微调（Instruction Tuning）

**思想：** 在预训练后，用指令-回答对进行微调。

**数据格式：**
```
指令：翻译英文到中文
输入：Hello
输出：你好
```

**效果：** 显著提升模型的可用性和安全性。

---

## E5.5 LLM的局限

### 幻觉（Hallucination）

**定义：** 模型生成看起来合理但实际上错误的内容。

**原因：**
- 模型没有真正的知识，只是学会了模式
- 在不确定时，模型倾向于生成看起来合理的文本

**缓解方法：**
- 提供准确的上下文
- 要求模型说明不确定性
- 使用检索增强生成（RAG）

### 知识截断

**问题：** 模型的知识截止于训练数据的时间。

**解决方案：**
- 定期重新训练
- 使用检索增强生成
- 微调新知识

### 推理能力有限

**问题：** 模型在复杂推理上仍然有限。

**例子：**
- 数学计算：容易出错
- 逻辑推理：在复杂情况下失败
- 常识推理：有时违反常识

---

## E5.6 与其他AI技术的比较

### LLM vs 传统NLP

**传统NLP：**
- 特征工程
- 任务特定的模型
- 需要大量标注数据

**LLM：**
- 自动特征学习
- 通用模型
- 需要大量无标注数据

### LLM vs 符号AI

**符号AI：**
- 显式的知识表示
- 可解释性强
- 推理能力强

**LLM：**
- 隐式的知识表示
- 可解释性弱
- 泛化能力强

### LLM vs 强化学习

**强化学习：**
- 需要环境交互
- 样本效率低
- 适合决策任务

**LLM：**
- 只需要文本数据
- 样本效率高
- 适合理解和生成任务

---

## E5.7 推荐论文

### 预训练的论文

1. **Devlin et al. (2018)** - "BERT: Pre-training of Deep Bidirectional Transformers"
   - 掩码语言模型的开创性工作

2. **Radford et al. (2019)** - "Language Models are Unsupervised Multitask Learners"
   - GPT-2论文
   - 展示了预训练的强大能力

### Scaling Laws的论文

1. **Kaplan et al. (2020)** - "Scaling Laws for Neural Language Models"
   - 发现幂律关系

2. **Hoffmann et al. (2022)** - "Training Compute-Optimal Large Language Models"
   - Chinchilla缩放律

### In-Context Learning的论文

1. **Brown et al. (2020)** - "Language Models are Few-Shot Learners"
   - GPT-3论文
   - 展示了In-Context Learning能力

2. **Wei et al. (2022)** - "Emergent Abilities of Large Language Models"
   - 涌现能力的系统研究

### Prompt工程的论文

1. **Wei et al. (2022)** - "Chain-of-Thought Prompting Elicits Reasoning in Large Language Models"
   - Chain-of-Thought的开创性工作

2. **Wang et al. (2022)** - "Self-Consistency Improves Chain of Thought Reasoning in Language Models"
   - 自洽性方法

---

## E5.8 进一步学习

### 书籍

- **"Attention Is All You Need" 论文详解**
  - 理解Transformer的基础

- **"The Illustrated GPT-2"**
  - 在线免费资源
  - 直观讲解GPT-2

### 在线资源

- **OpenAI的博客**
  - GPT系列模型的介绍
  - 最新的研究进展

- **Hugging Face的教程**
  - 如何使用预训练模型
  - Prompt工程的实践

### 实践项目

1. **调用LLM API**
   - 使用OpenAI或Anthropic的API
   - 理解API的使用方式

2. **Prompt工程实验**
   - 对比不同的Prompt
   - 观察输出的差异

3. **In-Context Learning演示**
   - 用不同数量的示例
   - 观察性能的变化

4. **微调实验**
   - 在小数据集上微调模型
   - 对比微调和Prompt工程

---

---

## E5.9 对齐训练的深度细节

> 本节是第5章主线 [5.3 对齐：SFT/RLHF/DPO](../03_alignment.md) 的深度扩展，补充 PPO 算法细节、KL 散度约束的数学含义、DPO 推导，以及 RL 在 Agent 推理中的新应用。

### PPO 在 RLHF 中的具体流程

主线文档介绍了 RLHF 的四步流程（预训练→SFT→奖励模型→PPO优化）。这里展开 PPO 这一步的内部机制。

**为什么用 PPO 而不是普通梯度下降？**

RLHF 的优化目标是最大化奖励模型的分数。直接用梯度下降的问题：每次更新后策略变了，旧数据就不再适用，需要重新采样——效率极低。PPO（Proximal Policy Optimization）的核心思想是**限制每次更新的幅度**，让旧数据可以被多次复用。

**PPO 在 RLHF 中的 4 步循环：**

```
Step 1：采样（Rollout）
  用当前策略 π_θ 对每个 prompt 生成回答
  → 得到 (prompt, response) 对

Step 2：打分（Scoring）
  用奖励模型 r_φ 对每个回答打分
  → 得到标量奖励 r(x, y)

Step 3：计算优势（Advantage Estimation）
  优势 A = r(x, y) - baseline
  baseline 通常是 value network 的预测值
  → 衡量"这个回答比平均水平好多少"

Step 4：PPO 更新
  最大化 clip(π_θ/π_old, 1-ε, 1+ε) × A
  clip 操作限制策略变化幅度（ε 通常取 0.2）
  → 更新策略参数 θ，回到 Step 1
```

**PPO 的 clip 操作直觉：**

```
如果 A > 0（好的回答）：
  鼓励增大该回答的概率，但不超过 1+ε 倍
如果 A < 0（差的回答）：
  鼓励减小该回答的概率，但不低于 1-ε 倍
```

这保证了每次更新不会太激进，策略保持稳定。

---

### KL 散度约束：防止模型"走偏"

RLHF 的实际目标函数不只是最大化奖励，还包含一个 KL 散度惩罚项：

$$\max_{\pi_\theta} \mathbb{E}_{x \sim D, y \sim \pi_\theta} \left[ r_\phi(x, y) - \beta \cdot \text{KL}(\pi_\theta \| \pi_{\text{SFT}}) \right]$$

**为什么需要 KL 约束？**

没有约束时，模型会"钻空子"——找到奖励模型的漏洞，生成奖励分数高但实际质量差的回答（奖励黑客，reward hacking）。

**KL 散度的含义：**

$$\text{KL}(\pi_\theta \| \pi_{\text{SFT}}) = \mathbb{E}_{y \sim \pi_\theta} \left[ \log \frac{\pi_\theta(y|x)}{\pi_{\text{SFT}}(y|x)} \right]$$

- KL = 0：当前策略和 SFT 模型完全一样
- KL 增大：当前策略偏离 SFT 模型越来越远

**β 系数的作用：**

| β 值 | 效果 |
|------|------|
| β = 0 | 纯粹最大化奖励，容易 reward hacking |
| β 很大 | 策略几乎不变，对齐效果差 |
| β 适中（0.1-0.5） | 在奖励提升和稳定性之间取平衡 |

---

### DPO 的数学推导（简化版）

DPO（Direct Preference Optimization）的核心洞察：RLHF 的最优解有闭合形式，可以直接从偏好数据训练，不需要单独的奖励模型。

**Step 1：RLHF 最优策略的闭合形式**

对上面的 RLHF 目标函数求最优解，可以证明：

$$\pi^*(y|x) = \frac{1}{Z(x)} \pi_{\text{SFT}}(y|x) \exp\left(\frac{r(x,y)}{\beta}\right)$$

其中 $Z(x)$ 是归一化常数。

**Step 2：反推奖励函数**

从上式反解出奖励：

$$r(x, y) = \beta \log \frac{\pi^*(y|x)}{\pi_{\text{SFT}}(y|x)} + \beta \log Z(x)$$

**Step 3：代入 Bradley-Terry 偏好模型**

人类偏好数据满足 Bradley-Terry 模型：

$$P(y_w \succ y_l | x) = \sigma(r(x, y_w) - r(x, y_l))$$

其中 $y_w$ 是偏好回答，$y_l$ 是非偏好回答，$\sigma$ 是 sigmoid 函数。

代入 Step 2 的奖励表达式，$Z(x)$ 项相消，得到 DPO 的训练目标：

$$\mathcal{L}_{\text{DPO}} = -\mathbb{E} \left[ \log \sigma \left( \beta \log \frac{\pi_\theta(y_w|x)}{\pi_{\text{ref}}(y_w|x)} - \beta \log \frac{\pi_\theta(y_l|x)}{\pi_{\text{ref}}(y_l|x)} \right) \right]$$

**直觉解读：**

```
DPO 在做什么：
  增大偏好回答 y_w 相对于参考模型的概率
  减小非偏好回答 y_l 相对于参考模型的概率
  β 控制偏离参考模型的幅度（和 RLHF 的 KL 系数作用相同）
```

**RLHF vs DPO 对比：**

| 维度 | RLHF | DPO |
|------|------|-----|
| 训练步骤 | 3步（SFT→RM→PPO） | 1步（直接优化） |
| 奖励模型 | 需要单独训练 | 不需要 |
| 计算成本 | 高（需要4个模型同时在线） | 低（只需2个模型） |
| 稳定性 | 较难调参 | 更稳定 |
| 效果 | 在某些场景更好 | 接近 RLHF，更简单 |

---

### RL for Agent：超越对齐的强化学习

对齐（RLHF/DPO）用 RL 让模型"更有用、更安全"。但 RL 在 LLM 中还有另一个方向：**让模型学会推理**。

#### GRPO（Group Relative Policy Optimization）

DeepSeek-R1 使用的算法，专为 LLM 推理设计。

**核心思想：** 对同一个问题采样多个回答，用组内相对排名代替绝对奖励。

```
对 prompt x，采样 G 个回答 {y_1, y_2, ..., y_G}
用规则（数学题答案是否正确）给每个回答打分 r_i
组内归一化：A_i = (r_i - mean(r)) / std(r)
用 A_i 作为优势函数更新策略
```

**优点：** 不需要 value network（PPO 需要），显存占用更小；奖励是规则性的（对/错），不需要奖励模型。

**适用场景：** 有明确正确答案的任务（数学、代码、逻辑推理）。

#### o1 风格的推理强化学习

OpenAI o1 和 DeepSeek-R1 的核心思想：用 RL 训练模型在回答前进行**长链推理**（Chain-of-Thought）。

**过程奖励 vs 结果奖励：**

```
结果奖励（Outcome Reward）：
  只看最终答案是否正确
  → 简单，但不鼓励正确的推理过程

过程奖励（Process Reward）：
  对推理链中每一步打分
  → 更精细，但需要人工标注推理步骤
```

**训练流程：**

```
1. 基础模型（预训练 + SFT）
2. 用 GRPO/PPO 训练：
   - 让模型生成长推理链（<think>...</think>）
   - 用最终答案的正确性作为奖励
   - 模型自发学会"想更多步骤"
3. 结果：模型在难题上显著提升
```

**为什么有效？** 更长的推理链 = 更多的中间计算步骤 = 更高的正确率。RL 发现了这个规律并强化了它。

---

**返回：** [第5章：LLM基础](../README.md)
