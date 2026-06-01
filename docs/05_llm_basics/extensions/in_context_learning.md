---
name: in_context_learning
description: In-Context Learning 原理与应用，第5章扩展内容
metadata:
  type: reference
---

# 扩展：In-Context Learning

**所属章节：** [第5章：LLM基础](../README.md)

---

## 什么是 In-Context Learning

In-Context Learning（ICL，上下文学习）是指 LLM 在不更新参数的情况下，仅通过 prompt 中的示例就能完成新任务的能力。

```
Zero-shot：
  指令：将以下句子翻译成法语：Hello
  输出：Bonjour

Few-shot（3-shot）：
  示例1：Hello → Bonjour
  示例2：Thank you → Merci
  示例3：Good morning → Bonjour matin
  待翻译：How are you?
  输出：Comment allez-vous?
```

## 为什么 LLM 能做到

ICL 不是真正的"学习"（参数没有更新），更准确的理解是：

**假设1：模式识别**  
模型在推理时从 prompt 中识别任务模式，利用预训练时学到的元学习能力来完成任务。

**假设2：梯度下降类比**  
有研究表明，ICL 在数学上等价于在隐状态空间中执行隐式梯度下降。

## 影响 ICL 效果的因素

| 因素 | 影响 |
|------|------|
| 示例数量 | 通常 3-5 个效果最好，更多不一定更好 |
| 示例质量 | 高质量示例 > 随机示例 |
| 示例顺序 | 顺序对结果有影响（最后一个示例影响最大） |
| 示例多样性 | 覆盖不同情况的示例效果更好 |
| 模型规模 | 小模型 ICL 效果差，大模型才能充分利用 |

## 与微调的对比

| 维度 | ICL | 微调 |
|------|-----|------|
| 参数更新 | 否 | 是 |
| 数据需求 | 几条示例 | 数百到数千条 |
| 推理成本 | 高（示例占用 context） | 低（示例已编码进参数） |
| 灵活性 | 高（随时换任务） | 低（需要重新训练） |
| 效果上限 | 低于微调 | 高于 ICL |

## 实践建议

- 先用 ICL 快速验证任务可行性
- 如果 ICL 效果不够，再考虑微调
- 示例选择：与输入相似、覆盖边界情况、格式一致

---

**返回：** [第5章：LLM基础](../README.md)
