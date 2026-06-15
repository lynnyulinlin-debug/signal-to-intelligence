# 附录B：模型获取与数据集完全指南

**版本：** v1.0（2026-06-03）  
**对标：** 第0章第4小节 — [当前LLM生态的主流模型](../00_introduction/04_tools_and_infrastructure.md#当前llm生态的主流模型)

---

## 快速导航

| 我想... | 跳转位置 | 预计时间 |
|--------|---------|---------|
| 快速了解模型网站 | [主流模型网站](#主流模型网站导航) | 5 min |
| 下载我的第一个模型 | [快速开始：下载模型](#快速开始-下载模型) | 10 min |
| 理解模型文件格式 | [模型文件格式详解](#模型文件格式详解) | 15 min |
| 获取数据集 | [数据集获取指南](#数据集获取指南) | 20 min |
| 解决下载问题 | [常见问题与解决](#常见问题与解决) | As needed |
| 量化模型对比 | [量化模型选择](#量化模型选择) | 10 min |

---

## 前置说明

> **参考第0章第4小节：** [主流模型对比表](../00_introduction/04_tools_and_infrastructure.md#当前llm生态的主流模型)
>
> 如果你还不确定选择哪个模型，**先查看第0章的模型对比表**（2分钟）
>
> 本章补充：如何在各个网站上找到和下载这些模型，不同格式的区别，数据集获取方法。

---

## 主流模型网站导航

### 1. Hugging Face（最全面）

**网址：** https://huggingface.co  
**特点：**
- 模型数量最多（500,000+）
- 社区最活跃
- 支持在线推理试用
- 模型卡详细完整

**如何使用：**

```bash
# 安装官方工具
pip install huggingface-hub

# 使用CLI下载模型
huggingface-cli download \
    meta-llama/Llama-2-7b-hf \
    --local-dir ./models/llama2-7b \
    --local-dir-use-symlinks False

# 使用Python API
from huggingface_hub import snapshot_download

snapshot_download(
    repo_id="Qwen/Qwen2-7B-Instruct",
    local_dir="./models/qwen2-7b",
)
```

**国内加速（中国用户）：**

```bash
# 使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

huggingface-cli download \
    meta-llama/Llama-2-7b-hf \
    --local-dir ./models/llama2-7b
```

---

### 2. ModelScope（国内首选）

**网址：** https://modelscope.cn  
**特点：**
- 中文支持最好
- 下载速度快（国内服务器）
- 阿里巴巴维护
- Qwen系列官方源

**如何使用：**

```bash
# 安装工具
pip install modelscope

# 下载模型
from modelscope import snapshot_download

model_dir = snapshot_download(
    'qwen/Qwen2-7B-Instruct',
    cache_dir='./models'
)
```

**浏览器下载：**
1. 访问 https://modelscope.cn
2. 搜索模型（如 "Qwen2-7B"）
3. 进入模型页面
4. 点击"克隆或下载"
5. 复制 git 链接并使用 git-lfs 下载

---

### 3. WiseModel（国内备选）

**网址：** https://wisemodel.cn  
**特点：**
- 中文模型资源丰富
- 国内镜像源
- 速度稳定

**如何使用：**
```bash
# 类似 HuggingFace 的 API
# 需要先下载 git-lfs
# git clone https://wisemodel.cn/...
```

---

### 4. Replicate（云端试用）

**网址：** https://replicate.com  
**特点：**
- 无需本地部署，即可试用模型
- 按需付费推理
- API 调用方便

**适用场景：**
- 快速测试模型效果
- 不想部署本地推理
- 需要 API 接口调用

---

## 模型文件格式详解

### 常见格式对比

| 格式 | 全名 | 优点 | 缺点 | 文件大小 | 加载速度 |
|------|------|------|------|---------|---------|
| **SafeTensors** | Safe Tensors | 安全、快速 | 支持有限 | 原始大小 | 最快 |
| **GGUF** | GGML Unified Format | 量化、兼容性好 | 仅推理 | 小（量化） | 快 |
| **PyTorch** | .pt / .bin | 兼容性好 | 大、不安全 | 原始大小 | 中等 |
| **TensorFlow** | .pb / .savedmodel | TensorFlow生态 | PyTorch用户少用 | 很大 | 慢 |
| **PEFT** | Parameter-Efficient Fine-Tuning | 微调权重小 | 需基础模型 | 很小（几MB） | - |

### 选择建议

```
我想要什么？

├─ 本地推理，追求速度
│  └─ 选择：GGUF（量化版本）
│
├─ 本地推理，保证精度
│  └─ 选择：SafeTensors 或 PyTorch
│
├─ 微调模型
│  ├─ 完整微调：PyTorch
│  └─ 参数高效微调：PEFT
│
└─ API调用或框架推理
   └─ 选择：任意（框架会处理）
```

---

## 快速开始：下载模型

### 方案1：Ollama（最简单）

```bash
# 列出可用模型
ollama list

# 下载模型
ollama pull qwen:7b          # 中文最优
ollama pull llama2:7b        # 英文
ollama pull mistral:7b       # 快速推理

# 检查下载
ollama list
```

**优点：** 一条命令搞定，自动优化  
**缺点：** 功能有限

---

### 方案2：HuggingFace CLI

```bash
# 安装工具
pip install huggingface-hub

# 下载特定模型
huggingface-cli download \
    Qwen/Qwen2-7B-Instruct \
    --local-dir ./models/qwen2-7b

# 下载特定文件
huggingface-cli download \
    Qwen/Qwen2-7B-Instruct \
    model.safetensors \
    --local-dir ./models/qwen2-7b
```

---

### 方案3：Python API

```python
from huggingface_hub import snapshot_download

# 下载整个仓库
model_path = snapshot_download(
    repo_id="Qwen/Qwen2-7B-Instruct",
    cache_dir="./models",
    resume_download=True,  # 断点续传
    local_dir_use_symlinks=False,  # 避免符号链接
)

print(f"模型已下载到: {model_path}")

# 列出文件
import os
files = os.listdir(model_path)
print(files)
```

---

## 数据集获取指南

### 主流数据集来源

#### Hugging Face Datasets

```python
from datasets import load_dataset

# 加载流行数据集
dataset = load_dataset("wikitext", "wikitext-2")

# 查看数据
print(dataset)
print(dataset["train"][0])

# 保存到本地
dataset.save_to_disk("./data/wikitext-2")

# 查看可用数据集
# https://huggingface.co/datasets
```

#### 常用数据集

| 数据集 | 规模 | 用途 | 下载命令 |
|--------|------|------|---------|
| Wikitext-2 | 3.8M | 语言模型 | `load_dataset("wikitext", "wikitext-2")` |
| OpenWebText | 37GB | 预训练 | `load_dataset("openwebtext")` |
| ALPACA | 52K | 指令微调 | `load_dataset("tatsu-lab/alpaca")` |
| Chinese-LLAMA | 20M | 中文 | ModelScope下载 |

#### 国内数据源

```bash
# ModelScope数据集
python -c "
from modelscope.msdatasets import MsDataset
ds = MsDataset.load(
    'chinese-medical-dialog',
    cache_dir='./data'
)
"

# 中文 LLAMA 数据集
# https://modelscope.cn/datasets
```

---

## 模型文件夹结构详解

下载模型后，文件夹通常包含：

```
qwen2-7b/
├── config.json              # 模型配置
├── model.safetensors        # 模型权重（最大文件，通常 14GB）
├── tokenizer.model          # 分词器（sentencepiece）
├── tokenizer.json           # 分词器配置
├── tokenizer_config.json    # 分词器参数
├── generation_config.json   # 生成参数
├── special_tokens_map.json  # 特殊 token 映射
├── README.md                # 模型说明
└── .gitattributes           # Git 配置（用于 LFS）
```

### 各文件说明

**config.json：** 模型结构配置
```json
{
  "model_type": "qwen",
  "num_hidden_layers": 32,
  "hidden_size": 4096,
  "num_attention_heads": 32,
  "intermediate_size": 11008,
  ...
}
```

**model.safetensors：** 模型权重（最关键）
- 这是最大的文件
- 包含模型的所有参数
- 通常占 70-90% 的空间

**tokenizer.model：** 分词器
- 将文本转换为 token
- 不同模型可能不同
- 必须配对使用

---

## 量化模型选择

### GGUF 格式量化模型

许多模型提供预量化的 GGUF 版本，极大减小文件大小：

```bash
# 搜索量化模型
# HuggingFace 上搜索 "GGUF" 关键词
# 例：TheBloke/Mistral-7B-Instruct-v0.1-GGUF

# 下载量化模型
huggingface-cli download \
    TheBloke/Qwen2-7B-Instruct-GGUF \
    qwen2-7b-instruct.Q4_K_M.gguf \
    --local-dir ./models/
```

**量化版本说明：**

| 版本 | 大小 | 精度 | 推理速度 |
|------|------|------|---------|
| Q2_K | 2.3GB | 低 | 最快 |
| Q3_K | 3.3GB | 中低 | 快 |
| Q4_K | 4.3GB | 中 | 中等 |
| Q5_K | 5.3GB | 中高 | 较慢 |
| Q6_K | 6.3GB | 高 | 慢 |

---

## 常见问题与解决

### 问题1：下载速度慢

**症状：** 每个文件只有几 KB/s

**解决方案：**

```bash
# 1. 如果在中国，使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 2. 或使用 ModelScope
from modelscope import snapshot_download
snapshot_download("qwen/Qwen2-7B-Instruct")

# 3. 或分成多个并行下载
# 使用 aria2 工具
aria2c -x 16 https://huggingface.co/...
```

---

### 问题2：磁盘空间不足

**症状：** "No space left on device"

**解决方案：**

```bash
# 1. 检查磁盘使用
du -sh ./models/

# 2. 下载量化版本（更小）
huggingface-cli download \
    TheBloke/Qwen2-7B-GGUF \
    qwen2-7b.Q4_K_M.gguf

# 3. 删除旧模型
rm -rf ./models/old_model/

# 4. 使用流式加载（不下载完整模型）
# 某些框架支持动态下载
```

---

### 问题3：无法访问 HuggingFace

**症状：** Connection refused / Connection timeout

**解决方案：**

```bash
# 方案1：使用代理
export HTTP_PROXY=http://proxy.example.com:8080
huggingface-cli download ...

# 方案2：使用国内镜像
export HF_ENDPOINT=https://hf-mirror.com

# 方案3：手动下载
# 1. 使用浏览器访问模型页面
# 2. 下载所有文件到本地
# 3. 使用本地路径加载模型
```

---

### 问题4：下载中断如何恢复？

**解决方案：**

```bash
# 大多数工具支持断点续传
huggingface-cli download \
    model_name \
    --local-dir ./models \
    --resume-download  # 关键参数
```

---

## 模型许可证与使用

### 常见许可证

| 许可证 | 商用 | 修改 | 可用性 |
|--------|------|------|--------|
| MIT | ✅ | ✅ | 完全开放 |
| Apache 2.0 | ✅ | ✅ | 完全开放 |
| OpenRAIL | ✅ | ✅ | 有限制 |
| 商业许可 | ❌ | ❌ | 需付费 |

**查看许可证：**
- 模型 README.md 中通常有说明
- HuggingFace 页面顶部显示许可证

---

## 相关链接

**第0章参考：**
- [第0章：模型生态对比](../00_introduction/04_tools_and_infrastructure.md)

**其他附录：**
- [附录B：本地推理指南](B_local_inference.md)
- [附录B：Docker容器化](B_docker_cnb.md)
- [附录C：代码运行指南](C_code_guide.md)

**外部资源：**
- HuggingFace Hub：https://huggingface.co
- ModelScope：https://modelscope.cn
- HuggingFace Datasets：https://huggingface.co/datasets

---

**版本：** v1.0（2026-06-03）
