# 5.3 In-Context Learning

**核心问题：** 什么是In-Context Learning？为什么LLM能做到？

---

## 什么是In-Context Learning

### 定义

模型在推理时，根据输入中的示例学习新任务，无需参数更新。

### 例子

```
输入：
翻译英文到法文的例子：
- "Hello" → "Bonjour"
- "Good morning" → "Bon matin"

现在翻译："Good night"

输出：
"Bonne nuit"
```

模型看了两个例子，就能翻译新的句子。

---

## Few-Shot Learning

### 定义

用少量示例（通常3-5个）让模型学习新任务。

### 与传统机器学习的对比

**传统机器学习：**
- 需要数千个标注样本
- 需要重新训练模型

**LLM的Few-Shot：**
- 只需要几个示例
- 无需重新训练

---

## Zero-Shot Learning

### 定义

不给任何示例，直接让模型完成任务。

### 例子

```
输入：
将以下文本分类为正面或负面情感：
"这个产品很好用，我很满意。"

输出：
正面
```

模型没有看过任何分类示例，但仍能完成任务。

---

## 为什么LLM能做到

### 1. 预训练学到了通用知识

LLM在预训练时学到了大量的语言知识和世界知识。

### 2. 上下文编码

Transformer的自注意力机制能有效编码上下文信息。

### 3. 大模型的涌现能力

只有足够大的模型才能展现In-Context Learning能力。

---

## 本节小结

In-Context Learning是LLM的关键能力：
- Few-Shot：用少量示例学习新任务
- Zero-Shot：无需示例直接完成任务
- 这是大模型的涌现能力

---

**下一节：** [5.4 Prompt工程基础](04_prompt_engineering.md)
