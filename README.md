# 《Signals to Intelligence》教程

从数字信号处理到大模型的思想演化与代码实践

## 📖 项目简介

这是一份面向工程师的**思想对照式**快速入门教程。我们不重复造轮子，而是帮你把已有的信号处理/统计知识迁移到现代AI（深度学习、大模型、多模态）的理解中。

**核心理念：** 智能不是凭空产生，而是从信号处理的基本原则中一步步生长出来的。

## 🎯 目标读者

- 具有本科/硕士理工科背景，工作1~5年的工程师
- 学过至少一门：数字信号处理、通信、自动化、控制、计算机视觉或机器学习基础
- 希望快速理解"Transformer 是怎么来的？""Mamba 和卡尔曼滤波有什么关系？"这类本质问题
- 不想读大部头教材，也不想只看概念幻灯片，希望有代码跑起来验证想法

## 📋 前置知识（最低要求）

- 会用 Python + NumPy（或愿意边学边用）
- 知道什么是向量、矩阵、导数、概率
- 听说过傅里叶变换、卷积、梯度下降

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
```

### 2. 运行第一个实验

```bash
# 运行 FFT 频谱分析实验
python code/ch01_dsp/fft_spectrum.py

# 运行所有测试
pytest tests/ -v
```

### 3. 阅读教程

所有章节位于 `docs/` 目录：

```
docs/
├── 00_introduction.md          # 导论与路径指引
├── 01_dsp_basics.md            # 第1章：数字信号处理基础
├── 02_statistical_signal.md    # 第2章：统计信号处理
├── 03_adaptive_filtering.md    # 第3章：自适应滤波与优化
├── 04_kalman_and_state_space.md # 第4章：卡尔曼滤波与状态空间
├── 05_statistical_learning.md  # 第5章：统计学习（浅层）
├── 06_deep_learning.md         # 第6章：深度学习（表示学习）
├── 07_transformer.md           # 第7章：Transformer与自注意力
├── 08_multimodal.md            # 第8章：视觉与多模态（ViT）
└── appendix/
    ├── A_math_reference.md     # 数学备忘
    ├── B_environment_setup.md  # 详细环境配置
    └── C_code_guide.md         # 代码运行指南
```

## 📚 教程结构

全书共 8 章 + 附录，每章正文控制在 8 页以内。

| 章节 | 标题 | 传统思想 | 现代AI映射 | 代码实验 |
|------|------|---------|----------|---------|
| 0 | 导论 | 系统如何利用数据进行自适应推断 | 全书路线图 | 无 |
| 1 | DSP基础 | 傅里叶变换、滤波器、时频分析 | CNN卷积核、Transformer位置编码 | FFT频谱、位置编码 |
| 2 | 统计信号处理 | MMSE估计、似然比检测 | 回归/分类损失函数、贝叶斯DL | MMSE vs 神经网络 |
| 3 | 自适应滤波 | LMS / RLS 梯度下降 | SGD / Adam 优化器 | LMS vs Adam收敛对比 |
| 4 | 卡尔曼滤波 | 预测-更新递归 | RNN/LSTM门控、Mamba | 卡尔曼跟踪、RNN结构 |
| 5 | 统计学习 | 偏差-方差权衡、正则化 | 权重衰减、Dropout | 多项式拟合 vs MLP |
| 6 | 深度学习 | 多层特征抽象 | CNN、RNN、自编码器 | MNIST CNN + 可视化 |
| 7 | Transformer | 序列对齐、注意力机制 | 自注意力、位置编码、多头 | 自注意力权重可视化 |
| 8 | 多模态 | 图像作为序列、跨模态对齐 | ViT、CLIP | ViT patches、CLIP相似度 |

## 🛠️ 代码实验

每章都包含最小可运行的代码实验：

- **独立性** — 每个实验可单独运行，只需 `python xxx.py`
- **最小化** — 核心逻辑 ≤ 30 行（不含 import、绘图、注释）
- **可复现** — 固定随机种子，输出确定
- **合成数据** — 除非必要，不用真实大数据集
- **清晰输出** — 打印数值结果 + 可选绘图

## 📖 阅读路径指引

根据你的背景选择阅读路径：

```
你的背景？
├─ 信号/控制/通信背景 
│  → 重点阅读第5,6,7,8章的"现代映射"部分，跳过基础
├─ AI/计算机背景 
│  → 重点阅读第1,2,3,4章的"传统思想"部分
└─ 两者都熟 
   → 按顺序通读，关注每章的"对照表"和实验
```

## 📦 依赖管理

**核心依赖（第1-5章）：**
```
numpy>=1.20
matplotlib>=3.3
pytest>=7.0
```

**可选依赖（第6-8章）：**
```
torch>=2.0
torchvision>=0.15.0
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

## 📝 项目进度

详见 [PROGRESS.md](PROGRESS.md)

## 🤝 贡献与反馈

- 发现问题？提交 [GitHub Issue](https://github.com/your-repo/issues)
- 有改进建议？提交 [Pull Request](https://github.com/your-repo/pulls)
- 想讨论想法？使用 [GitHub Discussions](https://github.com/your-repo/discussions)

## 📄 许可证

MIT License

## 🔗 相关资源

- [DESIGN.md](DESIGN.md) — 原始设计文档
- [design_detail.md](design_detail.md) — 详细操作手册
- [docs/appendix/B_environment_setup.md](docs/appendix/B_environment_setup.md) — 详细环境配置
- [docs/appendix/C_code_guide.md](docs/appendix/C_code_guide.md) — 代码运行指南

---

**版本：** v0.1 (开发中)  
**最后更新：** 2026-05-13
