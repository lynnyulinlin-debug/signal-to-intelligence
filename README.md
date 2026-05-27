# 《Signals to Intelligence》教程

从数字信号处理到大语言模型的思想演化与代码实践

## 📖 项目简介

这是一份面向工程师的**思想对照式**快速入门教程。我们不重复造轮子，而是帮你把已有的信号处理/统计知识迁移到现代AI（深度学习、Transformer、大语言模型、多模态）的理解中。

**核心理念：** 智能不是凭空产生，而是从信号处理的基本原则中一步步生长出来的。从DSP到优化算法，从深度学习到Transformer，再到LLM的预训练、应用和工程实践。

## 🎯 目标读者

- 具有本科/硕士理工科背景，工作1~5年的工程师
- 学过至少一门：数字信号处理、通信、自动化、控制、计算机视觉或机器学习基础
- 希望快速理解"Transformer 是怎么来的？""LLM 如何工作？""如何部署LLM应用？"这类本质问题
- 不想读大部头教材，也不想只看概念幻灯片，希望有代码跑起来验证想法
- 对LLM应用感兴趣，想了解从基础到工程实践的完整路径

## 📋 前置知识（最低要求）

- 会用 Python + NumPy（或愿意边学边用）
- 知道什么是向量、矩阵、导数、概率
- 听说过傅里叶变换、卷积、梯度下降

## 📚 完整目录

### 第1章：DSP基础
- **文档**: [docs/01_dsp/](docs/01_dsp/) — 信号处理基础
- **代码**: [code/ch01_dsp/](code/ch01_dsp/) — FFT、位置编码实验
- **Notebook**: [notebooks/ch01_dsp_interactive.ipynb](notebooks/ch01_dsp_interactive.ipynb) — 交互式学习
- **学习时间**: 30分钟

### 第2章：优化与机器学习
- **文档**: [docs/02_optimization/](docs/02_optimization/) — 优化算法与传统ML详解
- **代码**: [code/ch02_optimization/](code/ch02_optimization/) — 优化器对比、线性回归、SVM、决策树
- **学习时间**: 60-90分钟

### 第3章：深度学习快速通道
- **文档**: [docs/03_deep_learning_fast/](docs/03_deep_learning_fast/) — CNN/RNN基础
- **代码**: [code/ch03_deep_learning_fast/](code/ch03_deep_learning_fast/) — 多项式拟合、MNIST、RNN
- **学习时间**: 40分钟

### 第4章：Transformer详解
- **文档**: [docs/04_transformer/](docs/04_transformer/) — 自注意力、位置编码
- **代码**: [code/ch04_transformer/](code/ch04_transformer/) — 自注意力机制实现
- **学习时间**: 50分钟

### 第5章：LLM基础
- **文档**: [docs/05_llm_basics/](docs/05_llm_basics/) — 预训练、Scaling Laws、Prompt工程
- **代码**: [code/ch05_llm_basics/](code/ch05_llm_basics/) — LLM API调用演示
- **学习时间**: 50分钟

### 第6章：LLM应用与微调
- **文档**: [docs/06_llm_applications/](docs/06_llm_applications/) — RAG、Agent、微调
- **代码**: [code/ch06_llm_applications/](code/ch06_llm_applications/) — RAG系统演示
- **学习时间**: 60分钟

### 第7章：多模态LLM
- **文档**: [docs/07_multimodal_llm/](docs/07_multimodal_llm/) — 视觉-语言对齐、Qwen VL
- **代码**: [code/ch07_multimodal_llm/](code/ch07_multimodal_llm/) — ViT、CLIP、高分辨率处理
- **学习时间**: 50分钟

### 第8章：LLM工程实践
- **文档**: [docs/08_llm_engineering/](docs/08_llm_engineering/) — 部署、成本优化、安全
- **代码**: [code/ch08_llm_engineering/](code/ch08_llm_engineering/) — 工程实践演示
- **学习时间**: 40分钟

### 附录
- **A. 数学参考**: [docs/appendix/A_math_reference.md](docs/appendix/A_math_reference.md)
- **B. 环境配置**: [docs/appendix/B_environment_setup.md](docs/appendix/B_environment_setup.md)
- **C. 代码运行指南**: [docs/appendix/C_code_guide.md](docs/appendix/C_code_guide.md)

## 🚀 快速开始

### 1. 环境配置

**系统要求：** Python 3.10+

```bash
# 克隆仓库
git clone <repo-url>
cd signals-to-intelligence

# 创建虚拟环境
python3.10 -m venv venv  # 或 python3.11, python3.12
source venv/bin/activate  # Linux/Mac
# 或 venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt

# 可选：安装LLM API依赖（第5-8章）
export ANTHROPIC_API_KEY="your-key"
export OPENAI_API_KEY="your-key"
```

### 2. 运行第一个实验

```bash
# 运行 FFT 频谱分析实验（第1章）
python code/ch01_dsp/fft_spectrum.py

# 运行 LLM API 演示（第5章）
python code/ch05_llm_basics/llm_api_demo.py

# 运行 LLM 工程实践演示（第8章）
python code/ch08_llm_engineering/llm_engineering_demo.py

# 运行所有测试
pytest tests/ -v
```

### 3. 阅读教程

所有章节位于 `docs/` 目录，按以下结构组织：

```
docs/
├── 00_introduction/              # 导论与学习路径
│   ├── README.md
│   ├── 01_why_llm_era.md
│   ├── 02_learning_paths.md
│   └── 03_overview.md
├── 01_dsp/                       # 第1章：DSP基础
│   ├── README.md
│   ├── 01_signals_basics.md
│   ├── 02_fourier_analysis.md
│   ├── 03_filtering.md
│   ├── 04_time_freq.md
│   └── extensions/advanced_topics.md
├── 02_optimization/              # 第2章：优化算法
│   ├── README.md
│   ├── 01_gradient_descent.md
│   ├── 02_adaptive_methods.md
│   ├── 03_why_matters_for_llm.md
│   └── extensions/advanced_optimization.md
├── 03_deep_learning_fast/        # 第3章：深度学习快速入门
│   ├── README.md
│   ├── 01_neural_networks.md
│   ├── 02_cnn_rnn.md
│   ├── 03_training_tricks.md
│   ├── 04_why_transformer_better.md
│   └── extensions/deep_learning_theory.md
├── 04_transformer/               # 第4章：Transformer架构
│   ├── README.md
│   ├── 01_self_attention.md
│   ├── 02_multi_head_attention.md
│   ├── 03_positional_encoding.md
│   ├── 04_architecture.md
│   └── extensions/attention_variants.md
├── 05_llm_basics/                # 第5章：LLM基础 (NEW)
│   ├── README.md
│   ├── 01_pretraining.md
│   ├── 02_scaling_laws.md
│   ├── 03_in_context_learning.md
│   ├── 04_prompt_engineering.md
│   └── extensions/llm_training_details.md
├── 06_llm_applications/          # 第6章：LLM应用与微调 (NEW)
│   ├── README.md
│   ├── 01_rag_systems.md
│   ├── 02_agent_frameworks.md
│   ├── 03_finetuning.md
│   ├── 04_case_studies.md
│   └── extensions/advanced_techniques.md
├── 07_multimodal_llm/            # 第7章：多模态LLM (NEW)
│   ├── README.md
│   ├── 01_vision_language.md
│   ├── 02_alignment.md
│   ├── 03_case_studies.md
│   └── extensions/multimodal_architectures.md
├── 08_llm_engineering/           # 第8章：LLM工程实践 (NEW)
│   ├── README.md
│   ├── 01_model_selection.md
│   ├── 02_cost_optimization.md
│   ├── 03_safety_alignment.md
│   ├── 04_best_practices.md
│   └── extensions/production_guide.md
└── appendix/
    ├── A_math_reference.md       # 数学备忘（含Transformer和LLM）
    ├── B_environment_setup.md    # 详细环境配置
    └── C_code_guide.md           # 代码运行指南
```

## 📚 教程结构

全书共 **8 章 + 导论 + 3 附录**，每章正文控制在 8 页以内。

### 第一部分：基础（第1-4章）

| 章节 | 标题 | 核心概念 | 现代AI映射 | 代码实验 |
|------|------|---------|----------|---------|
| 1 | DSP基础 | 傅里叶变换、滤波器、时频分析 | CNN卷积核、Transformer位置编码 | FFT频谱、位置编码 |
| 2 | 优化算法 | 梯度下降、自适应方法 | SGD、Adam、学习率调度 | LMS vs Adam收敛对比 |
| 3 | 深度学习快速入门 | 神经网络、CNN、RNN | 特征学习、序列建模 | 多项式拟合 vs MLP、MNIST CNN |
| 4 | Transformer架构 | 自注意力、多头注意力、位置编码 | 序列对齐、并行处理 | 自注意力权重可视化 |

### 第二部分：LLM（第5-8章）NEW

| 章节 | 标题 | 核心概念 | 关键技术 | 代码实验 |
|------|------|---------|---------|---------|
| 5 | LLM基础 | 预训练、缩放律、上下文学习 | 自回归建模、Prompt工程 | API调用、Prompt工程、In-context Learning |
| 6 | LLM应用与微调 | RAG系统、Agent框架、微调 | 检索增强、工具调用、参数高效微调 | RAG演示、Agent框架、微调对比 |
| 7 | 多模态LLM | 视觉-语言对齐、ViT、CLIP | 跨模态表示、对比学习 | Qwen2.5-VL分析、高分辨率处理 |
| 8 | LLM工程实践 | 模型选择、成本优化、监控 | 部署、错误处理、性能基准 | 模型对比、成本分析、监控告警 |

### 导论与附录

| 部分 | 标题 | 内容 |
|------|------|------|
| 0 | 导论 | 为什么是LLM时代、学习路径、全书概览 |
| A | 数学参考 | 线性代数、概率、Transformer数学、LLM基础 |
| B | 环境配置 | Python环境、依赖安装、LLM API配置 |
| C | 代码运行指南 | 所有实验的运行方法、参数说明、常见问题 |

## 🛠️ 代码实验

全书共 **12 个代码实验**，每个都是独立可运行的Python脚本。

### 实验清单

| 章节 | 实验 | 文件 | 运行时间 | 依赖 |
|------|------|------|---------|------|
| 1 | FFT频谱分析 | `code/ch01_dsp/fft_spectrum.py` | <1s | NumPy, Matplotlib |
| 1 | 位置编码 | `code/ch01_dsp/positional_encoding.py` | <1s | NumPy, Matplotlib |
| 2 | LMS vs Adam | `code/ch02_optimization/lms_vs_adam.py` | <1s | NumPy, Matplotlib |
| 3 | 多项式 vs MLP | `code/ch05_statistical_learning/polynomial_vs_mlp.py` | 1-2s | NumPy, PyTorch, Matplotlib |
| 3 | MNIST CNN | `code/ch06_deep_learning/mnist_cnn.py` | 30-60s | NumPy, PyTorch, Matplotlib |
| 4 | 自注意力 | `code/ch07_transformer/self_attention.py` | <1s | NumPy, Matplotlib |
| 5 | LLM API演示 | `code/ch05_llm_basics/llm_api_demo.py` | 1-5s | Anthropic/OpenAI SDK |
| 6 | RAG系统 | `code/ch06_llm_applications/rag_demo.py` | <1s | LangChain, FAISS |
| 7 | Qwen2.5-VL分析 | `code/ch07_multimodal/qwen_vl_analysis.py` | <1s | NumPy |
| 7 | 高分辨率处理 | `code/ch07_multimodal/high_resolution_processing.py` | <1s | NumPy, Matplotlib |
| 8 | LLM工程实践 | `code/ch08_llm_engineering/llm_engineering_demo.py` | <1s | 无外部依赖 |

### 代码实验特点

- **独立性** — 每个实验可单独运行，只需 `python xxx.py`
- **最小化** — 核心逻辑清晰，易于理解和修改
- **可复现** — 固定随机种子，输出确定
- **合成数据** — 除非必要，不用真实大数据集
- **清晰输出** — 打印数值结果 + 可选绘图

### 快速运行所有实验

```bash
# 运行所有实验
for f in code/ch*/*.py; do
  echo "运行 $f..."
  python "$f" || echo "失败：$f"
done

# 或使用pytest运行测试
pytest tests/ -v --tb=short
```

## 📖 阅读路径指引

根据你的背景和目标选择阅读路径：

### 路径1：快速入门（2小时）
适合：想快速了解LLM的工程师

导论 → 第5章（LLM基础）→ 第6章（LLM应用）→ 第8章（工程实践）

### 路径2：完整学习（8小时）
适合：想从基础到应用完整理解的工程师

导论 → 第1-4章（基础）→ 第5-8章（LLM）

### 路径3：信号处理背景（4小时）
适合：有DSP/控制背景，想快速转向LLM的工程师

导论 → 第2-4章（优化和Transformer）→ 第5-8章（LLM）

### 路径4：AI背景（4小时）
适合：有深度学习背景，想了解LLM工程实践的工程师

导论 → 第4章（Transformer回顾）→ 第5-8章（LLM）

### 路径5：深度研究（12小时）
适合：想完全掌握所有细节的研究者

导论 → 第1-8章（全部）+ 所有扩展内容 + 所有代码实验

### 学习时间估计

| 部分 | 快速版 | 深度版 |
|------|--------|--------|
| 导论 | 10分钟 | 15分钟 |
| 第1-4章（每章） | 10分钟 | 30分钟 |
| 第5-8章（每章） | 15分钟 | 40分钟 |
| 代码实验（每个） | 5分钟 | 15分钟 |
| 扩展内容（每章） | - | 20分钟 |

## 📦 依赖管理

**核心依赖（第1-4章）：**
```
numpy>=1.20
matplotlib>=3.3
pytest>=7.0
pytest-cov>=4.0
```

**深度学习依赖（第3-4章）：**
```
torch>=2.0
torchvision>=0.15.0
```

**LLM依赖（第5-8章）：**
```
openai>=1.0
anthropic>=0.7
langchain>=0.1
faiss-cpu>=1.7
```

**开发工具（可选）：**
```
jupyter>=1.0
ipython>=8.0
flake8>=4.0
black>=22.0
```

详见 `requirements.txt`。


## 🧪 运行测试

```bash
# 运行所有测试
pytest tests/ -v

# 运行特定章节的测试
pytest tests/test_ch01.py -v

# 运行并生成覆盖率报告
pytest tests/ --cov=code --cov-report=html
```

## 📊 项目统计

- **文档文件：** 30+ 个Markdown文件
- **代码实验：** 12 个完整的Python脚本
- **代码行数：** ~3,500+ 行
- **文档行数：** ~10,000+ 行
- **章节结构：** 8 章 + 导论 + 3 附录
- **学习时间：** 快速版 2小时，深度版 8小时

## 📝 项目进度

详见 [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md) 和 [PROGRESS.md](PROGRESS.md)

## 🤝 贡献与反馈

- 发现问题？提交 [GitHub Issue](https://github.com/your-repo/issues)
- 有改进建议？提交 [Pull Request](https://github.com/your-repo/pulls)
- 想讨论想法？使用 [GitHub Discussions](https://github.com/your-repo/discussions)

## 📄 许可证

MIT License

## 🔗 相关资源

- [COMPLETION_SUMMARY.md](COMPLETION_SUMMARY.md) — v2.0 完成总结
- [PROGRESS.md](PROGRESS.md) — 项目进度跟踪
- [DECISIONS.md](DECISIONS.md) — 设计决策文档
- [docs/00_introduction/README.md](docs/00_introduction/README.md) — 导论与学习路径
- [docs/appendix/A_math_reference.md](docs/appendix/A_math_reference.md) — 数学参考
- [docs/appendix/B_environment_setup.md](docs/appendix/B_environment_setup.md) — 详细环境配置
- [docs/appendix/C_code_guide.md](docs/appendix/C_code_guide.md) — 代码运行指南

---

**版本：** v2.0 ✓ 完成
**最后更新：** 2026-05-26
**状态：** 生产就绪（Production Ready）
