# 扩展：微调工程实践

**所属章节：** [第6章：LLM应用](../README.md)  
**前置阅读：** [6.2 微调：让 LLM 适配你的任务](../02_finetuning.md)  
**原理层内容：** 见 [第5章 E5.x 微调方法调研](../../05_llm_basics/extensions/finetuning_survey.md)

---

## 概览

6.2 节介绍了微调的原理和 LoRA 的数学基础。本节聚焦**工程实践**：如何准备数据、选择工具、配置训练、评估效果。

---

## 数据准备

### 数据格式

SFT 数据的标准格式是指令-回答对，主流格式有两种：

**Alpaca 格式（单轮）：**
```json
{
  "instruction": "将下面的句子翻译成英文",
  "input": "今天天气很好",
  "output": "The weather is very nice today."
}
```

**ShareGPT 格式（多轮对话）：**
```json
{
  "conversations": [
    {"from": "human", "value": "你好，帮我写一首关于春天的诗"},
    {"from": "gpt", "value": "春风轻抚柳丝长..."},
    {"from": "human", "value": "能改成七言绝句吗？"},
    {"from": "gpt", "value": "春风拂柳绿如烟..."}
  ]
}
```

**选择建议：** 单轮任务用 Alpaca，对话场景用 ShareGPT。

### 数据质量检查

微调数据的质量比数量重要，上线前检查：

```
数据质量检查清单：
  ✓ 格式一致（所有样本字段完整，input 为空时用 "" 而非省略字段）
  ✓ 无重复（用 MinHash 或精确哈希去重，去重率 < 5%）
  ✓ 长度分布合理（过长样本截断，过短样本过滤）
  ✓ 无有害内容（色情、暴力、歧视）
  ✓ 标注一致性（同类问题回答风格统一）
  ✓ 覆盖目标场景（不同难度、不同子任务均有样本）
```

### 数据增强

数据量不足时，可通过以下方式扩充：

- **指令改写**：用 LLM 对同一任务生成多种表述方式（"翻译" → "请将…译为英文" / "用英语表达…"）
- **回译**：中文 → 英文 → 中文，产生语义等价但表述不同的样本
- **难度分层**：在已有样本基础上，让 LLM 生成更难/更简单的变体

**混合任务时的数据配比：** 多任务微调时，各任务样本量差异过大会导致模型偏向大任务。常见做法是按任务数量取平方根后归一化（temperature sampling），或直接手动设定各任务上限。

### 数据量参考

| 任务类型 | 最小数据量 | 推荐数据量 |
|---------|-----------|-----------|
| 格式/风格适配 | 200-500 条 | 1000-3000 条 |
| 领域知识注入 | 1000-2000 条 | 5000-10000 条 |
| 指令遵循 | 500-1000 条 | 3000-5000 条 |
| 代码生成 | 2000-5000 条 | 10000+ 条 |

---

## 主流微调工具

### PEFT（Hugging Face）

最基础的 LoRA 实现，与 Transformers 生态无缝集成：

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType

model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")

config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=8,                          # LoRA rank
    lora_alpha=16,                # 缩放系数
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
)
model = get_peft_model(model, config)
model.print_trainable_parameters()
# trainable params: 3,407,872 || all params: 7,241,732,096 || trainable%: 0.047
```

### LLaMA-Factory

一站式微调框架，支持 100+ 模型，提供 Web UI：

```bash
pip install llamafactory

# 启动 Web UI
llamafactory-cli webui

# 命令行训练
llamafactory-cli train \
  --model_name_or_path Qwen/Qwen2.5-7B-Instruct \
  --dataset alpaca_zh \
  --finetuning_type lora \
  --lora_rank 8 \
  --output_dir ./output
```

**适合：** 快速上手，不想写训练代码。  
**局限：** 高度封装，复杂实验（自定义 loss、多阶段训练）难以实现。

### Unsloth

速度优化版 LoRA，训练速度比标准 PEFT 快 2-5 倍，显存减少 60%：

```python
from unsloth import FastLanguageModel

model, tokenizer = FastLanguageModel.from_pretrained(
    model_name="unsloth/Qwen2.5-7B-Instruct",
    max_seq_length=2048,
    load_in_4bit=True,   # QLoRA
)
model = FastLanguageModel.get_peft_model(
    model, r=16, lora_alpha=16,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
)
```

**适合：** 显存有限（消费级 GPU），追求训练速度。  
**局限：** 只支持部分主流模型（Llama、Qwen、Mistral 等），非通用方案；不支持自定义模型架构。

### 工具选型建议

| 场景 | 推荐工具 | 局限 |
|------|---------|------|
| 快速验证，不想写代码 | LLaMA-Factory（Web UI） | 复杂实验难以定制 |
| 消费级 GPU（RTX 3090/4090） | Unsloth + QLoRA | 仅支持部分模型 |
| 需要自定义训练逻辑 | PEFT + Transformers | 需要手写训练循环 |
| 大规模分布式训练 | DeepSpeed + PEFT | 配置复杂，调试成本高 |

---

## 训练配置参考

### 关键超参数

```yaml
# 典型 LoRA 微调配置（7B 模型，单卡 A100 80G）
model: Qwen2.5-7B-Instruct
lora_rank: 8
lora_alpha: 16
lora_dropout: 0.05
target_modules: [q_proj, v_proj, k_proj, o_proj]

learning_rate: 2e-4
lr_scheduler: cosine
warmup_ratio: 0.03
num_epochs: 3
batch_size: 4
gradient_accumulation_steps: 4   # 等效 batch_size = 16
max_seq_length: 2048

# 显存估算（7B 模型，LoRA r=8）
# 模型权重（FP16）：~14GB
# LoRA 参数：~50MB
# 激活值（batch=4, seq=2048）：~8GB
# 优化器状态：~200MB
# 总计：~22GB（A100 40G 可运行）
```

### 训练监控

关注以下指标判断训练是否正常：

```
正常训练曲线：
  训练 loss：持续下降，最终趋于平稳
  验证 loss：下降后趋于平稳（若上升则过拟合）
  梯度范数：稳定在 0.1-10 之间（过大则梯度爆炸）

异常信号：
  loss 不下降 → 学习率太小或数据格式错误
  loss 突然变 NaN → 学习率太大或数值溢出
  验证 loss 上升 → 过拟合，减少 epoch 或增加 dropout
```

---

## 评估与合并

### 评估微调效果

**定性评估**（必做）：对比微调前后的输出，覆盖目标场景的典型样本和边界情况。

```python
# 定性评估：对比微调前后的输出
prompts = ["你的测试问题1", "你的测试问题2"]
for prompt in prompts:
    base_output = base_model.generate(prompt)
    ft_output = finetuned_model.generate(prompt)
    print(f"Base: {base_output}")
    print(f"Finetuned: {ft_output}")
```

**定量评估**：在保留的测试集上计算任务指标。

| 任务类型 | 推荐指标 | 注意事项 |
|---------|---------|---------|
| 分类/抽取 | 准确率、F1 | 最可靠 |
| 代码生成 | pass@k（执行通过率） | 比 BLEU 更有意义 |
| 摘要/翻译 | ROUGE-L、BLEU | 与人类判断相关性有限，仅供参考 |
| 开放生成 | 人工评估 | 无法自动化，但最准确 |

**人工评估**（关键场景必做）：BLEU/ROUGE 对生成任务的相关性有限，领域专家验收是最终标准。建议设计评分表（流畅性、准确性、格式合规性），至少抽样 50-100 条。

### 合并 LoRA 权重

推理时可以将 LoRA 权重合并到基础模型，消除推理开销：

```python
from peft import PeftModel

# 加载基础模型 + LoRA 权重
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B")
model = PeftModel.from_pretrained(model, "./lora_output")

# 合并并保存
model = model.merge_and_unload()
model.save_pretrained("./merged_model")
```

合并后的模型与原始模型大小相同，推理速度无损失。

---

## 常见坑与避坑指南

| 坑 | 现象 | 解决方法 |
|----|------|---------|
| `input` 字段为空时格式不对 | loss 不降，模型输出乱码 | Alpaca 格式中 `input` 为空时必须保留字段并设为 `""`，不能省略 |
| `max_seq_length` 设置过大 | 显存 OOM，训练极慢 | 先统计数据长度分布，取 95 分位数作为上限；超长样本截断而非扩大窗口 |
| `target_modules` 选错 | 参数量异常（过多或过少），效果差 | 不要选 embedding 层（`embed_tokens`）；标准选择是 attention 的 `q/k/v/o_proj` |
| 用 base 模型而非 instruct 模型 | 模型不遵循指令，输出格式混乱 | SFT 任务应从 instruct 版本微调，base 模型需要更多数据才能学会指令遵循 |
| 学习率过大 | loss 突然变 NaN | 7B 模型 LoRA 微调建议 `lr=1e-4~2e-4`，从小值开始；出现 NaN 立即降低 10 倍 |
| 过拟合（epoch 过多） | 训练 loss 低，验证 loss 上升，测试效果差 | 监控验证 loss，早停；通常 1-3 个 epoch 足够 |

---

## 后续步骤

微调完成后的典型流程：

```
1. 合并 LoRA 权重（见上方代码）
   ↓
2. 量化（可选）
   INT4/INT8 量化减少显存，适合部署
   工具：llama.cpp（GGUF）、AutoGPTQ、AWQ
   ↓
3. 部署推理服务
   本地：Ollama（GGUF）、vLLM（HuggingFace 格式）
   云端：HuggingFace Inference Endpoints、自建 API
   ↓
4. 持续迭代
   收集线上失败案例 → 补充到训练集 → 重新微调
   建议每 1-2 个月评估一次模型效果
```

**部署详情**见 [推理部署优化](inference_deployment.md)。

---

**返回：** [第6章：LLM应用](../README.md)
