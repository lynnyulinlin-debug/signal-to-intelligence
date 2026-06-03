# 代码运行指南

**版本：** v1.0  
**最后更新：** 2026-05-13

本指南说明如何运行《Signals to Intelligence》教程中的所有代码实验。

---

## 目录

1. [快速开始](#快速开始)
2. [代码组织结构](#代码组织结构)
3. [运行单个实验](#运行单个实验)
4. [运行所有测试](#运行所有测试)
5. [代码实验清单](#代码实验清单)
6. [常见问题](#常见问题)

---

## 快速开始

### 前置条件

确保已完成环境配置（见 [B_environment_setup.md](B_environment_setup.md)）：

```bash
# 激活虚拟环境
source venv/bin/activate  # Linux/macOS
# 或 venv\Scripts\activate  # Windows

# 验证依赖已安装
pip list | grep numpy
```

### 运行第一个实验（30秒）

```bash
# 进入项目目录
cd signals-to-intelligence

# 运行 FFT 频谱分析实验
python code/ch01_dsp/fft_spectrum.py
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

---

## 代码组织结构

所有代码实验位于 `code/` 目录，按章节组织：

```
code/
├── ch01_dsp/                    # 第1章：DSP基础
│   ├── fft_spectrum.py          # 实验1.1：FFT频谱分析
│   └── positional_encoding.py   # 实验1.2：位置编码
├── ch02_optimization/           # 第2章：优化与机器学习
│   └── lms_vs_adam.py           # 实验2.1：LMS vs Adam
├── ch03_deep_learning_fast/     # 第3章：深度学习快速通道
│   ├── polynomial_vs_mlp.py     # 实验3.1：多项式 vs MLP
│   └── mnist_cnn.py             # 实验3.2：MNIST CNN
├── ch04_transformer/            # 第4章：Transformer详解
│   └── self_attention.py        # 实验4.1：自注意力
├── ch05_llm_basics/             # 第5章：LLM基础
│   └── llm_api_demo.py          # 实验5.1：LLM API调用
├── ch06_llm_applications/       # 第6章：LLM应用
│   └── rag_demo.py              # 实验6.1：RAG系统演示
├── ch07_multimodal_llm/         # 第7章：多模态LLM
│   ├── vit_patches.py           # 实验7.1：ViT patches
│   └── clip_similarity.py       # 实验7.2：CLIP相似度
└── utils/                       # 工具函数
    ├── plotting.py              # 绘图工具
    └── data_gen.py              # 数据生成工具
```

---

## 运行单个实验

### 基本命令

```bash
# 运行某个实验
python code/ch{N}_{topic}/{experiment}.py

# 例子
python code/ch01_dsp/fft_spectrum.py
python code/ch02_optimization/lms_vs_adam.py
python code/ch03_deep_learning_fast/polynomial_vs_mlp.py
```

### 带参数运行（可选）

某些实验支持命令行参数：

```bash
# 查看实验支持的参数
python code/ch01_dsp/fft_spectrum.py --help

# 使用自定义参数
python code/ch01_dsp/fft_spectrum.py --seed 123 --noise-level 0.2
```

### 保存输出

```bash
# 保存输出到文件
python code/ch01_dsp/fft_spectrum.py > output.txt

# 保存图表（如果实验生成图表）
# 通常会自动保存到 assets/ch{N}/ 目录
```

---

## 运行所有测试

### 运行全部测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行并显示详细输出
pytest tests/ -vv

# 运行并显示打印语句
pytest tests/ -v -s
```

### 运行特定章节的测试

```bash
# 运行第1章的测试
pytest tests/test_ch01.py -v

# 运行第2章的测试
pytest tests/test_ch02.py -v
```

### 生成覆盖率报告

```bash
# 生成覆盖率报告
pytest tests/ --cov=code --cov-report=html

# 查看覆盖率报告
open htmlcov/index.html  # macOS
xdg-open htmlcov/index.html  # Linux
start htmlcov/index.html  # Windows
```

### 运行特定测试

```bash
# 运行特定测试函数
pytest tests/test_ch01.py::test_fft_output_shape -v

# 运行匹配模式的测试
pytest tests/ -k "fft" -v
```

---

## 代码实验清单

### 第1章：DSP基础

#### 基础部分（1.1-1.4）

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 1.1 FFT频谱分析 | `ch01_dsp/fft_spectrum.py` | NumPy, Matplotlib | <1s |
| 1.2 位置编码 | `ch01_dsp/positional_encoding.py` | NumPy, Matplotlib | <1s |

```bash
python code/ch01_dsp/fft_spectrum.py
python code/ch01_dsp/positional_encoding.py
```

#### 理论与应用部分（1.5-1.8）

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 1.5 随机信号分析 | `ch01_dsp/random_signals.py` | NumPy, Matplotlib | 1-2s |
| 1.6 信号检测器 | `ch01_dsp/signal_detection.py` | NumPy, Matplotlib, SciPy | 2-3s |
| 1.7 参数估计 | `ch01_dsp/parameter_estimation.py` | NumPy, Matplotlib, SciPy | 2-3s |
| 1.8 MUSIC算法 | `ch01_dsp/music_algorithm.py` | NumPy, Matplotlib, SciPy | 1-2s |

```bash
python code/ch01_dsp/random_signals.py
python code/ch01_dsp/signal_detection.py
python code/ch01_dsp/parameter_estimation.py
python code/ch01_dsp/music_algorithm.py
```

**注：** 新增的代码实验（1.5-1.8）待创建，详见 [Task #7 创建第1章代码实验](../../README.md)

### 第2章：优化与机器学习

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 2.1 LMS vs Adam | `ch02_optimization/lms_vs_adam.py` | NumPy, Matplotlib | <1s |

```bash
python code/ch02_optimization/lms_vs_adam.py
```

### 第3章：深度学习快速通道

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 3.1 多项式 vs MLP | `ch03_deep_learning_fast/polynomial_vs_mlp.py` | NumPy, PyTorch, Matplotlib | 1-2s |
| 3.2 MNIST CNN | `ch03_deep_learning_fast/mnist_cnn.py` | NumPy, PyTorch, Matplotlib | 30-60s |

```bash
python code/ch03_deep_learning_fast/polynomial_vs_mlp.py
python code/ch03_deep_learning_fast/mnist_cnn.py
```

### 第4章：Transformer详解

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 4.1 自注意力 | `ch04_transformer/self_attention.py` | NumPy, Matplotlib | <1s |

```bash
python code/ch04_transformer/self_attention.py
```

### 第5章：LLM基础

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 5.1 LLM API调用 | `ch05_llm_basics/llm_api_demo.py` | OpenAI/Anthropic SDK | 1-5s |

```bash
# 需要先设置 API 密钥
export OPENAI_API_KEY="your-key"
export ANTHROPIC_API_KEY="your-key"

python code/ch05_llm_basics/llm_api_demo.py
```

### 第6章：LLM应用与微调

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 6.1 RAG系统 | `ch06_llm_applications/rag_demo.py` | LangChain, FAISS, LLM SDK | 5-10s |

```bash
python code/ch06_llm_applications/rag_demo.py
```

### 第7章：多模态LLM

| 实验 | 文件 | 依赖 | 运行时间 |
|------|------|------|---------|
| 7.1 ViT Patches | `ch07_multimodal_llm/vit_patches.py` | NumPy, Matplotlib | 1-2s |
| 7.2 CLIP相似度 | `ch07_multimodal_llm/clip_similarity.py` | NumPy, Matplotlib | 1-2s |

```bash
python code/ch07_multimodal_llm/vit_patches.py
python code/ch07_multimodal_llm/clip_similarity.py
```

---

## 常见问题

### Q1: 运行实验时出现 `ModuleNotFoundError`？

**原因：** 依赖未安装或虚拟环境未激活

**解决方案：**

确保虚拟环境已激活，然后参考 [附录B：环境配置](B_environment_setup.md) 重新安装依赖。

### Q2: 实验运行很慢？

**原因：** 可能是 CPU 密集型操作或数据集过大

**解决方案：**
- 第6章的 MNIST CNN 训练可能需要 30-60 秒，这是正常的
- 如果需要加速，可以减少训练轮数（编辑代码中的 `epochs` 参数）

### Q3: 如何修改实验参数？

**方法1：** 编辑代码文件中的参数

```python
# 例如，在 fft_spectrum.py 中
SIGNAL_LENGTH = 1000  # 修改这个值
NOISE_LEVEL = 0.1     # 修改这个值
```

**方法2：** 使用命令行参数（如果实验支持）

```bash
python code/ch01_dsp/fft_spectrum.py --seed 123
```

### Q4: 如何保存实验结果？

```bash
# 保存输出到文件
python code/ch01_dsp/fft_spectrum.py > results.txt

# 图表通常会自动保存到 assets/ 目录
# 或者在代码中修改 plt.savefig() 的路径
```

### Q5: 如何在 Jupyter 中运行实验？

```bash
# 启动 Jupyter
jupyter notebook

# 在 Jupyter 中创建新单元格
%run code/ch01_dsp/fft_spectrum.py

# 或者复制代码到单元格中运行
```

### Q6: 如何调试实验代码？

```bash
# 使用 Python 调试器
python -m pdb code/ch01_dsp/fft_spectrum.py

# 或者在代码中添加 breakpoint()
# 然后运行
python code/ch01_dsp/fft_spectrum.py
```

### Q7: 如何运行所有实验？

```bash
# 运行所有实验（按顺序）
for f in code/ch*/*.py; do
  echo "运行 $f..."
  python "$f" || echo "失败：$f"
done
```

### Q8: 如何检查代码风格？

```bash
# 使用 flake8 检查代码风格
flake8 code/ --max-line-length=100

# 使用 black 格式化代码
black code/
```

### Q9: 如何配置 LLM API 密钥？

**方法1：** 设置环境变量

```bash
# Linux/macOS
export OPENAI_API_KEY="sk-..."
export ANTHROPIC_API_KEY="sk-ant-..."

# Windows (PowerShell)
$env:OPENAI_API_KEY="sk-..."
$env:ANTHROPIC_API_KEY="sk-ant-..."
```

**方法2：** 在代码中设置

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-..."
os.environ["ANTHROPIC_API_KEY"] = "sk-ant-..."
```

**方法3：** 使用 .env 文件

创建 `.env` 文件：
```
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
```

然后在代码中加载：
```python
from dotenv import load_dotenv
load_dotenv()
```

### Q10: LLM API 调用超时或失败？

**原因：** 网络连接问题、API 配额限制或密钥无效

**解决方案：**
```bash
# 检查网络连接
ping api.openai.com

# 检查 API 密钥是否正确
python -c "import os; print(os.environ.get('OPENAI_API_KEY'))"

# 查看 API 使用情况和配额
# 访问 https://platform.openai.com/account/usage/overview
```

### Q11: 如何降低 LLM API 的成本？

- 使用更便宜的模型（如 GPT-3.5 而不是 GPT-4）
- 减少 token 数量（缩短 prompt 和 response）
- 使用缓存和批处理
- 参考 [第6章 6.5 LLM 系统设计](../06_llm_applications/05_system_design.md) 的成本估算

### Q12: 如何在离线环境中运行代码？

某些实验需要网络连接（LLM API 调用），但可以：
- 运行第1-4章的所有实验（不需要网络）
- 使用本地模型替代 API（如 Ollama、LLaMA）
- 预先缓存 API 响应

---

## 下一步

1. 运行第一个实验：`python code/ch01_dsp/fft_spectrum.py`
2. 阅读对应的章节：`docs/01_dsp_basics.md`
3. 修改实验参数，观察输出变化
4. 运行所有测试：`pytest tests/ -v`

---

## 获取帮助

- 遇到问题？查看 [常见问题](#常见问题) 部分
- 需要更多帮助？提交 [GitHub Issue](https://github.com/your-repo/issues)
- 查看 [B_environment_setup.md](B_environment_setup.md) 了解环境配置
