# 环境配置指南

**版本：** v2.1  
**最后更新：** 2026-06-03

本指南说明如何配置《Signals to Intelligence》教程的开发环境。

> **我是新手** → 直接看 [快速上手](#快速上手10分钟)（10 分钟）  
> **我有 GPU / 要用本地模型** → 看完快速上手 + [GPU 加速](#gpu-加速可选) + [本地模型](#开源模型下载可选)  
> **我用云端环境** → 跳到 [云端环境方案](#云端环境方案)

---

## 目录

**核心路径（必需）**
1. [快速上手](#快速上手10分钟)
2. [本地环境配置](#本地环境配置)
3. [验证安装](#验证安装)

**可选扩展（需要时）**
4. [GPU 加速](#gpu-加速可选)
5. [开源模型下载](#开源模型下载可选)
6. [显存需求速查](#显存需求速查)
7. [IDE 配置](#ide-配置可选)

**云端方案**
8. [云端环境方案](#云端环境方案)

**问题排查**
9. [常见问题](#常见问题)

---

## 快速上手（10分钟）

### 前置条件

**操作系统：** Linux / macOS / Windows (WSL2)  
**Python 版本：** 3.10+ （推荐 3.11）

检查 Python：
```bash
python3 --version
```

### 第1步：选择环境

**选项 A：conda（推荐，尤其有 GPU）**
```bash
conda create -n sti python=3.11 --no-default-packages -y
conda activate sti
pip install --upgrade pip setuptools wheel
```

**选项 B：venv（标准 Python）**
```bash
git clone <repo-url>
cd signals-to-intelligence
python3 -m venv venv
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\Activate.ps1  # Windows
pip install --upgrade pip setuptools wheel
```

### 第2步：安装核心依赖
```bash
pip install -e .
```
✅ 完成 — 可以开始学习第 1-6 章，运行所有离线实验

### 第3步：验证安装
```bash
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
pytest tests/ -v
```

---

**接下来呢？**
- ✅ 想运行实验 → 见 [代码运行指南](C_code_guide.md)
- 🎮 有 GPU → 下翻到 [GPU 加速](#gpu-加速可选)
- 🤖 要本地推理 → 下翻到 [开源模型下载](#开源模型下载可选)
- ☁️ 用云端 → 跳到 [云端环境方案](#云端环境方案)

---

## 本地环境配置

### 系统要求

磁盘空间：最小 500 MB（不含模型），推荐 2 GB（含可选依赖）。

---

## 验证安装

### 检查核心依赖

```bash
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
python3 -c "import scipy; print(f'SciPy {scipy.__version__}')"
python3 -c "import pytest; print(f'pytest {pytest.__version__}')"
```

### 运行第一个实验

```bash
python3 code/ch01_dsp/fft_spectrum.py
```

预期输出：
```
==================================================
FFT 频谱分析
==================================================
信号长度: 1000
频率分量: [5.0, 10.0]
==================================================
```

### 运行所有测试

```bash
pytest tests/ -v
```

✅ **完成** — 可以开始学习第 1-6 章的所有实验

---

## ⭐ 可选扩展（需要时再看）

### GPU 加速（可选）

如果你有 NVIDIA GPU 且计划运行深度学习/模型推理实验（第3-7章），需要安装 GPU 版本的 PyTorch。

**检查 GPU：**
```bash
nvidia-smi  # 查看 CUDA 版本和 GPU 型号
```

**安装 GPU 版本：**

推荐使用自动检测：
```bash
pip install light-the-torch
ltt install torch torchvision
```

或手动指定（如果自动检测失败）：
```bash
# RTX 30/40 系（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121

# RTX 50 系 Blackwell（CUDA 12.8，需 PyTorch 2.7+）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**显存需求：** 见下方 [显存需求速查](#显存需求速查)

---

### 开源模型下载（可选）

本教程第5-6章的代码实验使用轻量级计算（BPE、矩阵运算、模拟数据），**不需要下载真实模型**即可运行。

如果需要体验本地推理或微调，以下是下载开源模型的方法。

#### 方式一：Hugging Face Hub（推荐）

```bash
# 安装 HF 工具
pip install -e ".[hf]"

# 下载模型（命令行）
huggingface-cli download Qwen/Qwen2.5-7B-Instruct --local-dir ./models/Qwen2.5-7B-Instruct
```

**国内加速（设置 HF 镜像）：**
```bash
export HF_ENDPOINT=https://hf-mirror.com
```

#### 方式二：ModelScope（国内备选）

```bash
pip install modelscope

# 下载模型
modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models
```

#### 方式三：Ollama（最简单，本地推理）

```bash
# 安装 Ollama（https://ollama.com）
ollama run qwen2.5:7b

# code/ch05_llm_basics/llm_api_demo.py 会自动检测并使用
python code/ch05_llm_basics/llm_api_demo.py
```

**推荐入门模型：**

| 模型 | 参数量 | 磁盘 | 说明 |
|------|--------|------|------|
| Qwen2.5-0.5B | 0.5B | ~1GB | CPU 可运行 |
| Qwen2.5-7B | 7B | ~14GB | 推荐，能力均衡 |
| Qwen2.5-7B-GPTQ-Int4 | 7B | ~4GB | 量化版，显存友好 |

---

### LLM API 密钥配置（可选）

如果你要用 API 而不是本地模型，需要配置 API 密钥。

**推荐优先级：**

| 提供商 | 模型 | 优点 | 地址 |
|--------|------|------|------|
| DeepSeek | deepseek-chat | 国内首选，极低价格 | [platform.deepseek.com](https://platform.deepseek.com) |
| 阿里云百炼 | qwen-plus | Qwen 系列，中文强 | [bailian.aliyun.com](https://bailian.aliyun.com) |
| 智谱 AI | glm-4-flash | 有免费额度 | [open.bigmodel.cn](https://open.bigmodel.cn) |
| Anthropic | claude-sonnet-4-6 | 综合能力强 | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | gpt-4o-mini | 生态最广 | [platform.openai.com](https://platform.openai.com) |

**设置环境变量：**

```bash
# 临时设置（当前终端）
export DEEPSEEK_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

# 永久设置（写入 ~/.bashrc 或 ~/.zshrc）
echo 'export DEEPSEEK_API_KEY="your-key"' >> ~/.bashrc
source ~/.bashrc
```

**使用 API：**

```python
from openai import OpenAI

# DeepSeek
client = OpenAI(
    api_key="your-key",
    base_url="https://api.deepseek.com"
)

response = client.chat.completions.create(
    model="deepseek-chat",
    messages=[{"role": "user", "content": "你好"}]
)
```

---

### 显存需求速查

运行开源模型的主要瓶颈是 GPU 显存。以下是常见规模模型的显存需求：

### 推理显存需求

| 模型规模 | FP16 | INT8 | INT4 (GPTQ/AWQ) |
|---------|------|------|-----------------|
| 0.5B | ~1GB | ~0.5GB | ~0.3GB |
| 1.5B | ~3GB | ~1.5GB | ~1GB |
| 7B | ~14GB | ~7GB | ~4GB |
| 13B | ~26GB | ~13GB | ~7GB |
| 70B | ~140GB | ~70GB | ~35GB |

### 微调显存需求（LoRA）

微调需要额外存储梯度和优化器状态：

| 模型规模 | 全量微调 | LoRA (r=8) | QLoRA (r=8, INT4) |
|---------|---------|-----------|-------------------|
| 7B | ~60GB | ~20GB | ~6GB |
| 13B | ~120GB | ~40GB | ~10GB |
| 70B | 需多卡 | ~160GB | ~48GB |

### 消费级 GPU 对应方案

| GPU | 显存 | CUDA | 推荐方案 |
|-----|------|------|---------|
| RTX 3060 / 3070 | 12GB / 8GB | 12.x | 7B INT4 推理，1.5B FP16 微调 |
| RTX 3090 | 24GB | 12.x | 7B FP16 推理，7B QLoRA 微调 |
| RTX 4060 / 4070 | 8~12GB | 12.x | 7B INT4 推理，1.5B FP16 微调 |
| RTX 4080 / 4090 | 16~24GB | 12.x | 7B FP16 推理，7B QLoRA 微调 |
| RTX 5060 / 5070 | 8~12GB | **12.8+** | 7B INT4 推理，1.5B FP16 微调 |
| RTX 5080 / 5090 | 16~32GB | **12.8+** | 7B FP16 推理，7B QLoRA 微调 |
| A100 40G | 40GB | 任意 | 13B FP16 推理，7B LoRA 微调 |
| A100 80G | 80GB | 任意 | 70B INT4 推理，13B LoRA 微调 |

**没有 GPU？** 以下方案可以在 CPU 上运行小模型：
- Ollama + `qwen2.5:0.5b`（0.5B 模型，CPU 可接受）
- llama.cpp + GGUF 格式（专为 CPU 优化）
- 使用 API（DeepSeek / 阿里云百炼 / OpenAI / Anthropic）代替本地模型

---

## 验证安装

### 验证 Python 和虚拟环境

```bash
which python3  # 应该显示 venv 目录下的 python
python3 --version
```

### 验证核心依赖

```bash
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
python3 -c "import scipy; print(f'SciPy {scipy.__version__}')"
python3 -c "import pytest; print(f'pytest {pytest.__version__}')"
```

### 验证可选依赖

```bash
# 深度学习（第3-4章）
python3 -c "import torch; print(f'PyTorch {torch.__version__}, CUDA: {torch.cuda.is_available()}')"

# LLM API（第5-6章）
python3 -c "import openai; print(f'openai {openai.__version__}')"
python3 -c "import anthropic; print(f'anthropic {anthropic.__version__}')"

# RAG（第6章）
python3 -c "import langchain; print(f'langchain {langchain.__version__}')"
python3 -c "import faiss; print('faiss OK')"
```

### 验证 API Key 配置

```bash
# 检查环境变量是否已设置（只显示是否存在，不显示值）
python3 -c "
import os
keys = ['DEEPSEEK_API_KEY', 'DASHSCOPE_API_KEY', 'ZHIPUAI_API_KEY',
        'ANTHROPIC_API_KEY', 'OPENAI_API_KEY']
for k in keys:
    status = '✓ 已设置' if os.environ.get(k) else '✗ 未设置'
    print(f'{k}: {status}')
"
```

### 一键验证脚本

```bash
# 运行第一个离线实验（不需要 API Key）
python3 code/ch05_llm_basics/bpe_tokenization.py
# 预期：Saved: ../../assets/ch05_bpe_tokenization.png

# 运行所有离线实验
make run-exp-ch05

# 运行 API 实验（需要至少一个 API Key）
make run-exp-ch05-api

# 运行所有测试
make test
```

### 运行第一个实验

```bash
python3 code/ch01_dsp/fft_spectrum.py
```

预期输出：
```
==================================================
FFT 频谱分析
==================================================
信号长度: 1000
频率分量: [5.0, 10.0]
==================================================
```

### 运行测试

```bash
pytest tests/ -v
```

预期输出：
```
tests/conftest.py::test_seed PASSED
tests/conftest.py::test_sample_signal PASSED
```

---

### IDE 配置（可选）

**VS Code：**
1. 安装 Python 扩展
2. Ctrl+Shift+P → "Python: Select Interpreter" → `./venv/bin/python`

**PyCharm：**
1. Settings → Project → Python Interpreter → Add
2. 选择 "Existing Environment" → `./venv/bin/python`

**Jupyter Lab：**
```bash
pip install jupyterlab
jupyter lab
```

---

## 云端环境方案

无本地 GPU 或不想配置本地环境？直接用云端方案。

### Google Colab（免费）

适合学习第 1-6 章，免费 T4 GPU：

```python
# 在 Colab notebook 中运行
!git clone <repo-url>
%cd signals-to-intelligence
!pip install -e . -q
!python code/ch01_dsp/fft_spectrum.py
```

### CNB 云原生开发环境（推荐）

> **即将开放**：CNB 项目空间配置完成后，将提供：
> - 一键启动链接
> - 预配置的 Linux + NVIDIA GPU 环境
> - 已安装所有依赖的镜像
> - 持久化存储

### Docker（本地隔离环境）

```bash
# CPU 版本
make docker-up

# GPU 版本
make docker-up-gpu
```

详见 `deploy/docker-compose.yml` 和 `deploy/docker-compose.gpu.yml`

---

## 常见问题

### PyTorch 安装失败？

检查网络连接，或使用镜像源：

```bash
pip install torch torchvision -i https://pypi.tsinghua.edu.cn/simple
```

### 下载 Hugging Face 模型速度慢？

设置国内镜像：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或改用 ModelScope：`modelscope download --model Qwen/Qwen2.5-7B-Instruct --local_dir ./models`

### 没有 GPU 能运行第5-6章实验吗？

**可以。** 这些实验使用模拟数据，无需真实模型。要体验模型推理，用 Ollama（CPU 可运行 0.5B 模型）或 API（DeepSeek / 阿里云百炼）。

### 在 Windows 上遇到权限错误？

以管理员身份运行 PowerShell。

### 如何重新安装依赖？

```bash
pip install --force-reinstall -e .
```

---

## 获取帮助

- 环境配置问题：查看上方 [常见问题](#常见问题)
- 代码运行问题：见 [代码运行指南](C_code_guide.md)
- 提交 Bug：[GitHub Issue](https://github.com/your-repo/issues)