# 环境配置指南

**版本：** v2.0  
**最后更新：** 2026-05-30

本指南详细说明如何配置《Signals to Intelligence》教程的开发环境。

---

## 目录

1. [系统要求](#系统要求)
2. [Python 环境配置](#python-环境配置)
3. [依赖安装](#依赖安装)
4. [开源模型下载](#开源模型下载)
5. [显存需求速查](#显存需求速查)
6. [验证安装](#验证安装)
7. [常见问题](#常见问题)
8. [IDE 配置（可选）](#ide-配置可选)
9. [云端环境方案](#云端环境方案)

---

## 系统要求

### 操作系统

- **Linux** — Ubuntu 20.04 LTS 或更新版本
- **macOS** — 10.14 或更新版本
- **Windows** — Windows 10 或更新版本（推荐使用 WSL2）

### Python 版本

- **Python 3.10+**（推荐 3.11 或 3.12）

检查 Python 版本：
```bash
python3 --version
```

如果系统中有多个 Python 版本，使用特定版本创建虚拟环境：
```bash
python3.10 -m venv venv  # 或 python3.11, python3.12
```

### 磁盘空间

- 最小：500 MB（不含数据集）
- 推荐：2 GB（含可选依赖）

---

## Python 环境配置

两种方式二选一，效果等价：

### 方式A：conda（推荐）

适合有 conda 的环境，尤其是系统存在 ROS 等环境污染、或需要 GPU 实验的场景：

```bash
conda create -n sti python=3.11 --no-default-packages -y
conda activate sti
pip install --upgrade pip setuptools wheel
```

完成后跳到 [依赖安装](#依赖安装)。

### 方式B：venv（标准 Python）

**步骤1：** 克隆仓库
```bash
git clone <repo-url>
cd signals-to-intelligence
```

**步骤2：** 创建虚拟环境
```bash
python3 -m venv venv
```

**步骤3：** 激活虚拟环境

Linux / macOS：
```bash
source venv/bin/activate
```

Windows (PowerShell)：
```powershell
venv\Scripts\Activate.ps1
```

**步骤4：** 升级 pip
```bash
pip install --upgrade pip setuptools wheel
```

---

## 依赖安装

### 安装核心依赖

```bash
pip install -e .
```

这会安装：numpy、matplotlib、scipy、pytest（依赖由 `pyproject.toml` 统一管理）。

### 安装可选依赖（第3-4章：深度学习）

如果你计划学习深度学习章节（第3-4章），还需要安装 PyTorch：

```bash
# CPU 版本（适合本教程所有学习实验，无需额外配置）
pip install -e ".[ml]"
```

如需 GPU 加速，在此基础上再覆盖安装对应的 torch：

```bash
# 推荐：自动检测已安装的 CUDA 驱动，选择匹配的 torch wheel
pip install light-the-torch && ltt install torch torchvision

# 手动指定（ltt 无法识别时的备选）
# 30系 / 40系 Ada（RTX 3060~4090）— CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
# 50系 Blackwell（RTX 5060~5090）— CUDA 12.8，需 PyTorch 2.7+
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

**选择 CPU 还是 GPU？**
- **CPU 版本** — 适合本教程所有学习实验，无需额外配置
- **GPU 版本** — 本地运行开源模型时有明显加速（第5-6章扩展实验）

### 安装可选依赖（第5-8章：LLM）

如果你计划学习 LLM 相关章节（第5-8章），需要安装 LLM API 客户端和相关工具：

```bash
# LLM API（OpenAI、Anthropic）
pip install -e ".[llm]"

# RAG 框架（LangChain、FAISS，第6章）
pip install -e ".[rag]"
```

如果需要在本地运行开源模型（第5-6章实验），还需要安装：

```bash
# Hugging Face 生态（transformers、datasets、PEFT 微调、量化、加速推理）
pip install -e ".[hf]"
```

**配置 API 密钥：**

国内外主流 API 均可用于本教程的实验，推荐按以下优先级选择：

| 提供商 | 模型 | 特点 | 获取地址 |
|--------|------|------|---------|
| DeepSeek | deepseek-chat / deepseek-reasoner | 国内首选，价格极低，兼容 OpenAI 格式 | [platform.deepseek.com](https://platform.deepseek.com) |
| 阿里云百炼 | qwen-plus / qwen-max | Qwen 系列，中文强，兼容 OpenAI 格式 | [bailian.aliyun.com](https://bailian.aliyun.com) |
| 智谱 AI | glm-4 / glm-4-flash | GLM 系列，有免费额度 | [open.bigmodel.cn](https://open.bigmodel.cn) |
| Anthropic | claude-sonnet-4-6 | 综合能力强，需境外网络 | [console.anthropic.com](https://console.anthropic.com) |
| OpenAI | gpt-4o / gpt-4o-mini | 生态最广，需境外网络 | [platform.openai.com](https://platform.openai.com) |

> **国内用户推荐：** DeepSeek 或阿里云百炼。两者均兼容 OpenAI SDK 格式，只需修改 `base_url` 和 `api_key`，代码无需改动。

**设置环境变量：**

```bash
# 方式1：命令行临时设置
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"

# 国内模型（兼容 OpenAI 格式）
export DEEPSEEK_API_KEY="your-key"
export DASHSCOPE_API_KEY="your-key"   # 阿里云百炼
export ZHIPUAI_API_KEY="your-key"     # 智谱 AI

# 方式2：写入 .env 文件（推荐，永久生效）
cp .env.example .env
# 编辑 .env 填入真实值
```

**使用国内模型（兼容 OpenAI SDK）：**

```python
from openai import OpenAI

# DeepSeek
client = OpenAI(
    api_key="your-deepseek-key",
    base_url="https://api.deepseek.com"
)

# 阿里云百炼（Qwen）
client = OpenAI(
    api_key="your-dashscope-key",
    base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
)

# 调用方式与 OpenAI 完全相同
response = client.chat.completions.create(
    model="deepseek-chat",   # 或 "qwen-plus"
    messages=[{"role": "user", "content": "你好"}]
)
print(response.choices[0].message.content)
```

### 安装开发工具（可选）

```bash
# Jupyter 笔记本（可选）
pip install jupyter ipython

# 代码检查工具（可选）
pip install flake8 black
```

---

## 开源模型下载

本教程第5-6章的代码实验使用轻量级计算（BPE、矩阵运算、模拟数据），**不需要下载真实模型**即可运行。

如果你想进一步实验（如本地推理、微调），本节介绍如何下载开源模型。

### 方式一：Hugging Face Hub（推荐）

Hugging Face Hub 是最主要的开源模型托管平台。

**安装客户端：**
```bash
pip install -e ".[hf]"
```

**下载模型：**
```python
from huggingface_hub import snapshot_download

# 下载 Qwen2.5-7B-Instruct（推荐入门模型）
snapshot_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct",
    local_dir="./models/Qwen2.5-7B-Instruct",
    ignore_patterns=["*.bin"],   # 只下载 safetensors 格式
)
```

或使用命令行：
```bash
huggingface-cli download Qwen/Qwen2.5-7B-Instruct \
  --local-dir ./models/Qwen2.5-7B-Instruct
```

**国内访问加速（HF 镜像）：**

Hugging Face 在国内访问较慢，设置镜像站可大幅提速：

```bash
# 方式1：设置环境变量（推荐，临时生效）
export HF_ENDPOINT=https://hf-mirror.com

# 方式2：写入 ~/.bashrc（永久生效）
echo 'export HF_ENDPOINT=https://hf-mirror.com' >> ~/.bashrc
source ~/.bashrc
```

设置后，所有 `huggingface_hub` 和 `transformers` 的下载请求都会走镜像。

### 方式二：ModelScope（国内备选）

ModelScope 是阿里云的模型平台，国内下载速度快，Qwen 系列模型优先在此发布：

```bash
pip install modelscope
```

```python
from modelscope import snapshot_download

snapshot_download(
    model_id="Qwen/Qwen2.5-7B-Instruct",
    cache_dir="./models"
)
```

### 模型存储管理

模型文件较大（7B 模型约 14GB），建议统一管理存储路径：

```bash
# 设置 HF 缓存目录（默认是 ~/.cache/huggingface）
export HF_HOME=/data/models/huggingface

# 设置 ModelScope 缓存目录
export MODELSCOPE_CACHE=/data/models/modelscope
```

**推荐入门模型：**

| 模型 | 参数量 | 磁盘占用 | 特点 |
|------|--------|---------|------|
| Qwen2.5-0.5B-Instruct | 0.5B | ~1GB | 极轻量，CPU 可运行 |
| Qwen2.5-1.5B-Instruct | 1.5B | ~3GB | 轻量，适合学习 |
| Qwen2.5-7B-Instruct | 7B | ~14GB | 能力均衡，推荐 |
| Qwen2.5-7B-Instruct-GPTQ-Int4 | 7B (INT4) | ~4GB | 量化版，显存友好 |

### 使用 Ollama（最简单的本地运行方式）

Ollama 封装了模型下载和推理，一条命令即可运行：

```bash
# 安装 Ollama（Linux）
curl -fsSL https://ollama.com/install.sh | sh

# 下载并运行 Qwen2.5-7B
ollama run qwen2.5:7b

# 下载量化版（显存需求更低）
ollama run qwen2.5:7b-instruct-q4_K_M
```

Ollama 适合**快速体验**，不适合微调或批量推理。

---

## 显存需求速查

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

## 常见问题

### Q1: 如何退出虚拟环境？

```bash
deactivate
```

### Q2: 如何删除虚拟环境？

```bash
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Q3: 如何重新安装依赖？

```bash
pip install --force-reinstall -e .
```

### Q4: 如何检查已安装的包？

```bash
pip list
```

### Q5: 如何升级某个包？

```bash
pip install --upgrade numpy
```

### Q6: 在 Windows 上遇到权限错误？

如果遇到 `Permission denied` 错误，尝试以管理员身份运行 PowerShell 或 CMD。

### Q7: PyTorch 安装失败？

检查网络连接，或使用清华大学镜像：

```bash
pip install torch torchvision -i https://pypi.tsinghua.edu.cn/simple
```

### Q8: 如何使用 Jupyter 笔记本？

```bash
# 安装 Jupyter
pip install jupyter

# 启动 Jupyter
jupyter notebook

# 在浏览器中打开 http://localhost:8888
```

### Q9: 下载 Hugging Face 模型速度很慢？

设置国内镜像站：

```bash
export HF_ENDPOINT=https://hf-mirror.com
```

或改用 ModelScope 下载同款模型（Qwen 系列在 ModelScope 上有完整镜像）。

### Q10: 没有 GPU，能运行第5-6章的代码实验吗？

**可以。** 第5-6章的代码实验（BPE、Scaling Laws、LoRA 可视化等）全部使用模拟数据和轻量计算，不需要真实模型，CPU 即可运行。

如果想体验真实模型推理，可以：
1. 使用 DeepSeek / 阿里云百炼 API（国内可直接访问，见上方 API 配置说明）
2. 用 Ollama 在 CPU 上运行 0.5B 小模型（速度慢但可用）

---

## IDE 配置（可选）

### VS Code

1. 安装 Python 扩展
2. 打开命令面板（Ctrl+Shift+P）
3. 搜索 "Python: Select Interpreter"
4. 选择 `./venv/bin/python`

### PyCharm

1. 打开 Settings → Project → Python Interpreter
2. 点击齿轮图标 → Add
3. 选择 "Existing Environment"
4. 选择 `./venv/bin/python`

### Jupyter Lab（可选）

```bash
pip install jupyterlab
jupyter lab
```

---

## 环境变量（可选）

如果需要调试，可以设置以下环境变量：

```bash
# 启用 NumPy 调试模式
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

# 启用 PyTorch 调试模式
export TORCH_DISTRIBUTED_DEBUG=INFO
```

---

## 下一步

1. 阅读 [README.md](../../README.md) 了解项目概览
2. 查看 [C_code_guide.md](C_code_guide.md) 了解如何运行代码实验
3. 开始阅读 [第0章：导论](../00_introduction/README.md)

---

## 云端环境方案

没有 GPU 或不想配置本地环境？以下云端方案可以直接在浏览器中运行实验。

### Google Colab / Kaggle（免费）

适合第 1-6 章的所有实验，免费提供 GPU：

```python
# 在 Colab/Kaggle 中运行
!git clone <repo-url>
%cd signal-to-intelligence
!pip install -e . -q

# 运行任意实验
!python code/ch05_llm_basics/bpe_tokenization.py
```

| 平台 | 免费 GPU | 时长限制 | 适合场景 |
|------|---------|---------|---------|
| Google Colab | T4 | 每天约 4-6 小时 | 快速体验，轻量实验 |
| Kaggle Notebooks | P100 | 每周 30 小时 | 较长时间的实验 |

### CNB 云原生开发环境（推荐）

> **待补充**：CNB 项目空间配置完成后，将在此提供：
> - 一键启动链接
> - 预配置的 Linux + NVIDIA GPU 环境
> - 已安装所有依赖的镜像
> - 持久化存储配置

CNB 是本教程推荐的云端方案，提供完整的 Linux 环境和 GPU 支持，适合：
- 没有 NVIDIA GPU 的 Windows/Mac 用户
- 需要运行开源模型微调实验的学习者
- 希望环境与本地完全一致的团队协作场景

### Docker（本地隔离环境）

见 [deploy/docker-compose.yml](../../deploy/docker-compose.yml)（CPU）和 [deploy/docker-compose.gpu.yml](../../deploy/docker-compose.gpu.yml)（GPU），或直接运行 `make docker-up` / `make docker-up-gpu`。

---

## 获取帮助

- 遇到问题？查看 [常见问题](#常见问题) 部分
- 需要更多帮助？提交 [GitHub Issue](https://github.com/your-repo/issues)
