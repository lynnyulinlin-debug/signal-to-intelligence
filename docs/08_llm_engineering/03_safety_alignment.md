# 8.3 安全与对齐

**核心问题：** 如何确保LLM应用的安全性？

---

## 问题

LLM可能生成有害内容：
- 虚假信息
- 有偏见的内容
- 有害的建议

---

## 安全措施

### 1. 输入验证

检查用户输入，防止恶意提示。

```python
def validate_input(prompt):
    if len(prompt) > 10000:
        raise ValueError("提示太长")
    if contains_harmful_keywords(prompt):
        raise ValueError("包含有害内容")
    return prompt
```

### 2. 输出过滤

检查模型输出，过滤有害内容。

```python
def filter_output(response):
    if contains_harmful_content(response):
        return "无法生成该内容"
    return response
```

### 3. 用户反馈

收集用户反馈，识别问题。

```python
def collect_feedback(response, user_feedback):
    if user_feedback == "有害":
        log_harmful_response(response)
```

---

## 对齐

### 定义

让模型的行为与人类价值观一致。

### 方法

- **指令微调**：用高质量指令微调
- **RLHF**：用人类反馈强化学习
- **宪法AI**：用一组原则指导模型

---

## 本节小结

安全与对齐的关键：
- 输入验证
- 输出过滤
- 用户反馈
- 指令微调和RLHF

---

**下一节：** [8.4 最佳实践](04_best_practices.md)
