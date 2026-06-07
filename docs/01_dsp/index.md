# 第1章：传统数字信号处理

**版本：** v2.1  
**最后更新：** 2026-05-30

## 章节概览

本章介绍传统数字信号处理（DSP）的核心概念，为后续深度学习章节奠定基础。虽然DSP是经典领域，但其思想在现代AI中仍然随处可见：CNN中的卷积来自DSP的滤波，Transformer中的位置编码来自傅里叶变换。

从主线角度看，本章也可以理解为一个经典的信息处理链路：先定义信号的表示方式，再分析频率结构，然后通过滤波抑制噪声、利用统计特性建模不确定性，最后完成检测、估计与特征提取。这条链路既适用于通信接收机，也适用于图像、音频、时间序列以及后续深度学习和LLM中的表示学习问题。

## 在线 Notebook

本章提供交互式运行版本，适合边看边调参数、边观察频谱和位置编码效果。

- Google Colab: [打开本章 Notebook](https://colab.research.google.com/github/lynnyulinlin-debug/signal-to-intelligence/blob/main/notebooks/ch01_dsp_interactive.ipynb)
- 使用说明: [Notebook 使用方式](/signal-to-intelligence/00_introduction/05_how_to_use_this_tutorial.html)

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 1.1 信号的三种视角 | [01_signals.md](01_signals.md) | ⭐ | 10分钟 |
| 1.2 傅里叶变换与频谱分析 | [02_fourier.md](02_fourier.md) | ⭐⭐ | 15分钟 |
| 1.3 滤波器与卷积 | [03_filters.md](03_filters.md) | ⭐⭐ | 15分钟 |
| 1.4 时频分析 | [04_time_freq.md](04_time_freq.md) | ⭐⭐ | 10分钟 |
| 1.5 随机信号理论 | [05_random_signals.md](05_random_signals.md) | ⭐⭐⭐ | 20分钟 |
| 1.6 信号检测 | [06_signal_detection.md](06_signal_detection.md) | ⭐⭐⭐ | 20分钟 |
| 1.7 信号估计 | [07_signal_estimation.md](07_signal_estimation.md) | ⭐⭐⭐ | 20分钟 |
| 1.8 矩阵分解应用 | [08_matrix_decomposition.md](08_matrix_decomposition.md) | ⭐⭐⭐⭐ | 20分钟 |

## 小节目录

### 基础概念（1.1-1.4）

**1.1 信号的三种视角** — [📖 阅读](01_signals.md)
- 时域、频域、时频，理解信号的多个角度

**1.2 傅里叶变换与频谱分析** — [📖 阅读](02_fourier.md)
- 从时域到频域的转换，位置编码的数学基础

**1.3 滤波器与卷积** — [📖 阅读](03_filters.md)
- 信号处理的基本操作，CNN 卷积核的原型

**1.4 时频分析** — [📖 阅读](04_time_freq.md)
- 同时看到时间和频率信息，STFT 与小波变换

### 理论与应用（1.5-1.8）

**1.5 随机信号理论** — [📖 阅读](05_random_signals.md)
- 统计特性、平稳性、高斯过程、噪声模型

**1.6 信号检测** — [📖 阅读](06_signal_detection.md)
- 假设检验、似然比检验、ROC曲线、检测器设计

**1.7 信号估计** — [📖 阅读](07_signal_estimation.md)
- 参数估计、MLE、LSE、贝叶斯估计、Cramér-Rao界

**1.8 矩阵分解应用** — [📖 阅读](08_matrix_decomposition.md)
- SVD、EVD、子空间方法、MUSIC、PCA、去噪

## 学习时间

- **快速版**（仅阅读1.1-1.4）：15分钟
- **标准版**（阅读1.1-1.8）：60分钟
- **深度版**（包含代码实验和练习）：120分钟

## 核心问题

完成本章后，你应该能回答：

**基础部分（1.1-1.4）：**
1. 信号可以从哪些角度理解？
2. 傅里叶变换的本质是什么？
3. 卷积如何实现滤波？
4. CNN中的卷积核与DSP中的滤波器有什么关系？

**进阶部分（1.5-1.8）：**
5. 什么是随机信号？如何分析其统计特性？
6. 如何在噪声中检测信号？什么是ROC曲线？
7. 如何从观测中估计信号参数？什么是Cramér-Rao界？
8. 矩阵分解如何应用于信号处理？什么是MUSIC算法？

## 代码实验

本章共有 **12 个代码脚本**，生成 **16 张图片**，覆盖所有 8 个小节。

| 小节 | 脚本 | 生成图片 | 文档位置 |
|------|------|---------|---------|
| 1.1 信号的三种视角 | [`signal_dimensions.py`](../../code/ch01_dsp/signal_dimensions.py) | `ch01_signal_dimensions.png` | [1.1](01_signals.md) |
| 1.1 信号的三种视角 | [`signal_three_views.py`](../../code/ch01_dsp/signal_three_views.py) | `ch01_three_views.png` | [1.1](01_signals.md) |
| 1.2 傅里叶变换 | [`fft_spectrum.py`](../../code/ch01_dsp/fft_spectrum.py) | `ch01_fft_spectrum.png` | [README](README.md) |
| 1.2 傅里叶变换 | [`fourier_2d.py`](../../code/ch01_dsp/fourier_2d.py) | `ch01_fourier_2d.png` | [1.2](02_fourier.md) |
| 1.2 傅里叶变换（LLM） | [`positional_encoding.py`](../../code/ch01_dsp/positional_encoding.py) | `ch01_positional_encoding.png` | [1.2](02_fourier.md) |
| 1.3 滤波器与卷积 | [`random_signals.py`](../../code/ch01_dsp/random_signals.py) | `ch01_convolution_effect.png` | [1.3](03_filters.md) |
| 1.4 时频分析 | [`time_freq_analysis.py`](../../code/ch01_dsp/time_freq_analysis.py) | `ch01_time_freq_music.png` | [1.4](04_time_freq.md) |
| 1.4 时频分析 | [`time_freq_analysis.py`](../../code/ch01_dsp/time_freq_analysis.py) | `ch01_time_freq_comparison.png` | [1.4](04_time_freq.md) |
| 1.5 随机信号 | [`random_signals.py`](../../code/ch01_dsp/random_signals.py) | `ch01_random_signals.png` | [1.5](05_random_signals.md) |
| 1.6 信号检测 | [`signal_detection.py`](../../code/ch01_dsp/signal_detection.py) | `ch01_detection_performance.png` | [1.6](06_signal_detection.md) |
| 1.6 信号检测 | [`signal_detection.py`](../../code/ch01_dsp/signal_detection.py) | `ch01_signal_detection.png` | [1.6](06_signal_detection.md) |
| 1.7 信号估计 | [`parameter_estimation.py`](../../code/ch01_dsp/parameter_estimation.py) | `ch01_parameter_estimation.png` | [1.7](07_signal_estimation.md) |
| 1.8 矩阵分解 | [`music_algorithm.py`](../../code/ch01_dsp/music_algorithm.py) | `ch01_matrix_decomposition.png` | [1.8](08_matrix_decomposition.md) |
| 1.8 矩阵分解 | [`music_algorithm.py`](../../code/ch01_dsp/music_algorithm.py) | `ch01_matrix_decomposition_performance.png` | [1.8](08_matrix_decomposition.md) |
| 1.8 矩阵分解 | [`feature_extraction_comparison.py`](../../code/ch01_dsp/feature_extraction_comparison.py) | `ch01_feature_extraction.png` | [1.8](08_matrix_decomposition.md) |
| 扩展：随机过程 | [`stochastic_processes_demo.py`](../../code/ch01_dsp/stochastic_processes_demo.py) | `ch01_stochastic_processes.png` | [extensions](extensions/stochastic_processes.md) |

**运行所有实验：**
```bash
python code/ch01_dsp/fft_spectrum.py
python code/ch01_dsp/signal_three_views.py
python code/ch01_dsp/time_freq_analysis.py
# ... 其余脚本见上表
```

![FFT Spectrum Analysis](/assets/ch01_fft_spectrum.png)

*图1.1：FFT频谱分析。展示信号在频域的表示，不同频率分量的幅度和相位。*

**代码文件：** [`code/ch01_dsp/fft_spectrum.py`](../../code/ch01_dsp/fft_spectrum.py)  
**运行方式：** `python code/ch01_dsp/fft_spectrum.py`

## 推荐学习路径

### 路径1：快速入门（15分钟）
- 阅读 1.1-1.4 的正文
- 查看图表和公式
- 理解基础概念
- 回答"核心问题"中的前4个问题

### 路径2：标准学习（60分钟）
- 阅读 1.1-1.8 的所有内容
- 理解每个小节的核心概念
- 查看实际应用例子
- 回答"核心问题"中的所有8个问题

### 路径3：深度学习（120分钟）
- 阅读所有内容
- 运行代码实验
- 分析实验结果
- 自己实现简单的算法（如能量检测器、MUSIC算法）
- 完成练习题

## 关键概念速查

| 概念 | 公式 | 直观理解 | 章节 |
|------|------|---------|------|
| 傅里叶变换 | $X(f) = \int_{-\infty}^{\infty} x(t)e^{-j2\pi ft}dt$ | 信号从时域到频域的转换 | 1.2 |
| 卷积 | $y[n] = \sum_{m} x[m]h[n-m]$ | 信号与滤波器的组合 | 1.3 |
| 频率响应 | $H(f) = \|H(f)\|e^{j\angle H(f)}$ | 滤波器对不同频率的响应 | 1.3 |
| 自相关函数 | $R(\tau) = E[x(t)x(t+\tau)]$ | 信号与自身的相似性 | 1.5 |
| 功率谱密度 | $S(f) = \mathcal{F}\{R(\tau)\}$ | 信号功率在频域的分布 | 1.5 |
| 似然比检验 | $\Lambda(y) = \frac{p(y\|H_1)}{p(y\|H_0)}$ | 最优信号检测 | 1.6 |
| 检测概率 | $P_d = P(\text{判定}H_1\|H_1\text{真实})$ | 正确检测信号的概率 | 1.6 |
| 最大似然估计 | $\hat{\theta}_{ML} = \arg\max_{\theta} L(\theta;\mathbf{y})$ | 最可能的参数值 | 1.7 |
| Cramér-Rao界 | $\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$ | 估计器方差的下界 | 1.7 |
| 奇异值分解 | $\mathbf{A} = \mathbf{U}\boldsymbol{\Sigma}\mathbf{V}^H$ | 矩阵的标准分解 | 1.8 |

## 常见问题

**Q: 为什么需要傅里叶变换？**
A: 很多信号处理问题在频域更容易解决。例如，滤波在频域就是简单的乘法。

**Q: FFT和DFT有什么区别？**
A: FFT是DFT的快速算法，计算复杂度从O(n²)降低到O(n log n)。

**Q: 卷积为什么在CNN中这么重要？**
A: 卷积能提取局部特征，这正是图像处理所需要的。

**Q: 什么是随机信号？为什么要研究它？**
A: 随机信号是不能完全确定的信号（如噪声）。现实中的信号往往包含随机成分，需要用统计方法分析。

**Q: 如何判断信号是否存在？**
A: 使用假设检验和似然比检验。通过设置阈值，在虚警率和漏检率之间权衡。

**Q: 参数估计和信号检测有什么区别？**
A: 检测是判断信号是否存在（二元决策），估计是确定信号的具体参数值（连续值）。

**Q: MUSIC算法为什么能高分辨率估计频率？**
A: MUSIC利用信号子空间和噪声子空间的正交性，能分离接近的频率成分。

## 扩展内容

### 高级DSP话题 — [📖 阅读](extensions/advanced_topics.md)
- 窗函数详解（Hann、Hamming、Blackman 的频谱特性）
- 多速率信号处理（上采样、下采样、滤波器组）
- 滤波器设计（Butterworth、Chebyshev、椭圆滤波器）
- 与深度学习的连接（卷积、注意力、位置编码）

### 线性系统理论 — [📖 阅读](extensions/linear_systems.md)
- 线性时不变系统（LTI）：冲激响应与卷积
- 频率响应与系统稳定性分析
- 状态空间表示、可控性与可观测性
- 与深度学习的连接（RNN 状态方程、CNN 滤波器）

### 随机过程进阶 — [📖 阅读](extensions/stochastic_processes.md)
- 平稳过程、马尔可夫链、高斯过程
- AR/MA/ARMA 模型与谱分析
- 随机过程的滤波（Wiener 滤波、Kalman 滤波）
- 与 LLM 的连接（语言模型作为序列随机过程）

## 关键连接点

### DSP → CNN

```
DSP中的卷积：y[n] = Σ x[m] * h[n-m]
                    ↓
CNN中的卷积核：可学习的滤波器
```

**启示：** CNN就是学习最优的滤波器。

### 傅里叶变换 → 位置编码

```
傅里叶变换：用不同频率的正弦波表示信号
                    ↓
位置编码：用不同频率的正弦波表示位置
```

**启示：** 位置编码让Transformer能理解序列顺序。

---

**下一步：** 阅读 [1.1 信号的三种视角](01_signals.md)
