# 扩展：对齐训练深度细节

**所属章节：** [第5章：LLM基础](../README.md)  
**前置阅读：** [5.5 强化学习对齐：RLHF 与 DPO](../05_rl_alignment.md)

---

## 概览

5.5 节介绍了 RLHF 和 DPO 的核心思路。本节补充 PPO 算法的内部机制、KL 散度约束的数学含义、DPO 的完整推导，以及 RL 在推理模型（o1/R1）中的新应用。

---

## PPO 在 RLHF 中的具体流程

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

## KL 散度约束：防止模型"走偏"

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

## DPO 的数学推导（简化版）

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

## RL for Agent：超越对齐的强化学习

对齐（RLHF/DPO）用 RL 让模型"更有用、更安全"。但 RL 在 LLM 中还有另一个方向：**让模型学会推理**。

### GRPO（Group Relative Policy Optimization）

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

### o1 风格的推理强化学习

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
