# 第2章：优化算法与传统机器学习

**版本：** v2.1  
**最后更新：** 2026-05-30

## 章节概览

本章把“优化”当作训练系统的底层机制来讲：先回答参数如何被更新、训练为什么会不稳定，再把线性模型、SVM 和树模型统一到“最小化目标函数”的视角下。这样读完之后，不只是知道算法名称，而是知道训练过程里应该同时关注目标、更新规则、学习率和数值稳定性。

本章重点关注：
- 优化问题如何被定义，以及梯度下降如何工作
- 动量、RMSprop、Adam 和学习率调度为什么能改变收敛行为
- 浮点数、条件数和梯度检查为什么会决定实现能否落地
- 线性回归、逻辑回归、SVM 和决策树如何统一到机器学习的优化视角
- 这些经验如何过渡到第 3 章的深度学习训练与第 5 章的 LLM 训练

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 2.1 梯度下降基础 | [01_gradient_descent.md](01_gradient_descent.md) | ⭐⭐ | 15分钟 |
| 2.2 自适应优化器 | [02_adaptive_optimizers.md](02_adaptive_optimizers.md) | ⭐⭐⭐ | 20分钟 |
| 2.3 优化与传统ML | [03_optimization_and_traditional_ml.md](03_optimization_and_traditional_ml.md) | ⭐⭐⭐ | 20分钟 |
| 2.4 数值方法基础 | [04_numerical_methods.md](04_numerical_methods.md) | ⭐⭐⭐ | 15分钟 |
| 2.5 线性/逻辑回归 | [05_linear_logistic_regression.md](05_linear_logistic_regression.md) | ⭐⭐ | 15分钟 |
| 2.6 SVM与核方法 | [06_svm_kernel_methods.md](06_svm_kernel_methods.md) | ⭐⭐⭐ | 20分钟 |
| 2.7 决策树与随机森林 | [07_decision_trees_random_forest.md](07_decision_trees_random_forest.md) | ⭐⭐⭐ | 20分钟 |

## 小节目录

### 优化算法基础

**2.1 梯度下降基础** — [📖 阅读](01_gradient_descent.md)
- 优化问题的定义和梯度下降的原理
- 反向传播和链式法则
- 为什么梯度下降对深度学习重要

**2.2 自适应优化器** — [📖 阅读](02_adaptive_optimizers.md)
- 动量、RMSprop、Adam等改进算法
- 为什么深度学习使用Adam
- 学习率调度的重要性

**2.3 优化算法与传统机器学习** — [📖 阅读](03_optimization_and_traditional_ml.md)
- 优化算法如何从“求最小值”扩展到“训练系统”
- 为什么训练大模型时必须同时关心算法、学习率和数值稳定性
- 传统 ML 为什么仍然是理解深度学习的重要前置
- 从传统 ML 到深度学习，再到 LLM 训练的统一视角

**2.4 数值方法基础** — [📖 阅读](04_numerical_methods.md)
- 浮点数精度、舍入误差和条件数
- 梯度检查、有限差分和反向传播的数值验证
- 梯度消失、梯度爆炸与训练稳定性

### 传统机器学习

**2.5 线性回归与逻辑回归** — [📖 阅读](05_linear_logistic_regression.md)
- 监督学习如何落到损失函数最小化
- 从回归到分类的目标函数变化
- 与神经网络训练的连续性

**2.6 支持向量机与核方法** — [📖 阅读](06_svm_kernel_methods.md)
- 最大间隔与结构化目标
- 核方法如何把非线性问题变回线性可分
- 与深度学习特征学习的对照关系

**2.7 决策树与随机森林** — [📖 阅读](07_decision_trees_random_forest.md)
- 贪心分裂和集成学习的基本思想
- 随机森林如何通过多模型降低方差
- 与多路表示融合思想的启发式类比

## 章节逻辑导图

```
优化问题的定义与更新规则（2.1-2.2）
    ↓
训练过程中的工程约束（2.3-2.4）
    ↓
传统机器学习如何落到目标函数最小化（2.5-2.7）
    ├─ 线性模型：可解释的基线
    ├─ 核方法：非线性建模的经典路线
    └─ 树模型：贪心分裂与集成学习
    ↓
深度学习训练（第3章）
    ↓
Transformer 与 LLM 训练（第4-5章）
```

## 学习时间

- **快速版**（仅阅读正文）：15分钟
- **标准版**（包含优化算法实验）：40分钟
- **完整版**（包含所有实验）：90分钟

## 核心问题

完成本章后，你应该能回答：

### 优化算法部分
1. 机器学习里的“优化问题”到底是在优化什么？
2. 梯度下降解决了什么问题，又有哪些局限？
3. 为什么动量、RMSprop 和 Adam 能显著影响训练行为？
4. 学习率调度为什么是训练流程的一部分，而不是附加选项？

### 传统机器学习部分
5. 数值稳定性为什么会决定一个训练方法能不能真正跑起来？
6. 线性回归和逻辑回归为什么都可以看作优化问题？
7. SVM 如何通过核方法处理非线性问题？
8. 决策树和随机森林分别解决了什么建模痛点？

## 代码实验

本章共有 **6 个代码脚本**，生成 **6 张图片**，覆盖优化算法和传统机器学习两大主题。

| 小节 | 脚本 | 生成图片 | 文档位置 |
|------|------|---------|---------|
| 2.1 梯度下降 | [`mmse_vs_nn.py`](../../code/ch02_optimization/mmse_vs_nn.py) | `ch02_mmse_vs_nn.png` | [README](README.md) |
| 2.2 自适应优化器 | [`lms_vs_adam.py`](../../code/ch02_optimization/lms_vs_adam.py) | `ch02_lms_vs_adam.png` | [2.2](02_adaptive_optimizers.md) / [README](README.md) |
| 2.5 线性/逻辑回归 | [`linear_logistic_regression.py`](../../code/ch02_optimization/linear_logistic_regression.py) | `ch02_linear_logistic_regression.png` | [2.5](05_linear_logistic_regression.md) / [README](README.md) |
| 2.6 SVM与核方法 | [`svm_kernel.py`](../../code/ch02_optimization/svm_kernel.py) | `ch02_svm_kernel.png` | [2.6](06_svm_kernel_methods.md) / [README](README.md) |
| 2.7 决策树与随机森林 | [`decision_tree_random_forest.py`](../../code/ch02_optimization/decision_tree_random_forest.py) | `ch02_decision_tree_random_forest.png` | [2.7](07_decision_trees_random_forest.md) / [README](README.md) |
| 扩展：凸分析 | [`convex_analysis_demo.py`](../../code/ch02_optimization/convex_analysis_demo.py) | `ch02_convex_analysis.png` | [extensions](extensions/convex_analysis.md) |

其中 5 个是核心实验，凸分析是扩展实验；它们共同帮助理解优化算法和传统机器学习的工作原理：

### 优化算法实验

**实验1：MMSE vs 神经网络**
- **文件：** [`code/ch02_optimization/mmse_vs_nn.py`](../../code/ch02_optimization/mmse_vs_nn.py)
- **内容：** 对比最小均方误差（MMSE）和神经网络的性能
- **运行：** `python code/ch02_optimization/mmse_vs_nn.py`
- **输出：** 性能对比图、收敛曲线、误差分析

![MMSE vs NN](/assets/ch02_mmse_vs_nn.png)

**实验2：LMS vs Adam优化器**
- **文件：** [`code/ch02_optimization/lms_vs_adam.py`](../../code/ch02_optimization/lms_vs_adam.py)
- **内容：** 对比LMS和Adam两种优化算法的收敛性
- **运行：** `python code/ch02_optimization/lms_vs_adam.py`
- **输出：** 收敛曲线、学习率效果、优化器对比

![LMS vs Adam](/assets/ch02_lms_vs_adam.png)

### 传统机器学习实验

**实验3：线性回归与逻辑回归**
- **文件：** [`code/ch02_optimization/linear_logistic_regression.py`](../../code/ch02_optimization/linear_logistic_regression.py)
- **内容：** 用梯度下降实现线性回归和逻辑回归，理解ML本质是优化问题
- **运行：** `python code/ch02_optimization/linear_logistic_regression.py`
- **输出：** 拟合曲线、决策边界、损失曲线

![Linear & Logistic Regression](/assets/ch02_linear_logistic_regression.png)

**实验4：SVM与核方法**
- **文件：** [`code/ch02_optimization/svm_kernel.py`](../../code/ch02_optimization/svm_kernel.py)
- **内容：** 实现线性SVM和核SVM，理解最大间隔分类和非线性分类
- **运行：** `python code/ch02_optimization/svm_kernel.py`
- **输出：** 决策边界、核函数效果、损失曲线

![SVM & Kernel Methods](/assets/ch02_svm_kernel.png)

**实验5：决策树与随机森林**
- **文件：** [`code/ch02_optimization/decision_tree_random_forest.py`](../../code/ch02_optimization/decision_tree_random_forest.py)
- **内容：** 实现决策树和随机森林，理解树模型和集成学习
- **运行：** `python code/ch02_optimization/decision_tree_random_forest.py`
- **输出：** 决策边界、树结构、集成效果

![Decision Tree & Random Forest](/assets/ch02_decision_tree_random_forest.png)

### 扩展实验：凸分析

**实验6：凸分析基础**
- **文件：** [`code/ch02_optimization/convex_analysis_demo.py`](../../code/ch02_optimization/convex_analysis_demo.py)
- **内容：** 凸函数与非凸函数的优化对比，条件数对收敛的影响
- **运行：** `python code/ch02_optimization/convex_analysis_demo.py`
- **输出：** 优化路径对比、Rosenbrock 函数等高线、条件数影响图

详见 [扩展：凸分析基础](extensions/convex_analysis.md)

## 推荐学习路径

### 路径1：快速入门（15分钟）
- 阅读 2.1-2.3 的正文
- 查看图表和公式
- 把优化问题、学习率和训练稳定性先建立起来

### 路径2：标准学习（40分钟）
- 阅读所有优化算法内容（2.1-2.4）
- 运行两个优化算法实验
- 阅读 2.5-2.7 传统ML部分

### 路径3：完整学习（90分钟）
- 阅读所有内容（优化算法 + 传统ML）
- 运行所有6个代码实验
- 回答"核心问题"中的8个问题
- 理解从传统ML到深度学习的演进

### 路径4：深入理论（2小时+）
- 完成路径3
- 阅读扩展文档：[高级优化话题](extensions/advanced_optimization.md)
- 阅读扩展文档：[凸分析基础](extensions/convex_analysis.md)
- 阅读扩展文档：[运筹学基础](extensions/operations_research_basics.md)，理解资源分配、调度、排队和系统约束优化

## 关键概念速查

### 优化算法

| 概念 | 公式 | 直观理解 |
|------|------|---------|
| 梯度下降 | $w_{t+1} = w_t - \alpha \nabla L(w_t)$ | 沿着梯度反方向更新参数 |
| 动量 | $v_t = \beta v_{t-1} + \nabla L(w_t)$ | 累积历史梯度方向 |
| Adam | $w_t = w_{t-1} - \alpha \frac{m_t}{\sqrt{v_t}+\epsilon}$ | 结合动量和自适应学习率 |

### 传统机器学习

| 算法 | 损失函数 | 优化方法 | 应用场景 |
|------|---------|---------|---------|
| 线性回归 | MSE | 梯度下降 | 连续值预测 |
| 逻辑回归 | 交叉熵 | 梯度下降 | 二分类 |
| SVM | Hinge损失 | 梯度下降/SMO | 二分类/多分类 |
| 决策树 | 基尼系数 | 贪心分割 | 分类/回归 |
| 随机森林 | 基尼系数 | Bootstrap+贪心 | 分类/回归 |

## 常见问题

**Q: 为什么梯度下降是最基础的训练方法？**
A: 因为它把“学习参数”统一成“沿着损失函数下降”的问题，配合反向传播后就能训练大规模模型。但它的稳定性强烈依赖学习率、曲率和数值实现。

**Q: Adam优化器为什么常作为强基线？**
A: Adam 把动量和自适应学习率结合起来，在很多任务上更容易得到可用收敛；但在大模型训练里，AdamW、学习率调度和权重衰减同样重要。

**Q: 学习率太大或太小会怎样？**
A: 太大会导致发散或震荡，太小会让训练拖得很慢。实际训练通常必须配合 warmup、衰减或周期性调度。

**Q: 机器学习的本质是什么？**
A: 可以把它看成在约束条件下最小化目标函数的过程，模型结构、正则化、数据分布和优化算法共同决定结果。

**Q: 为什么要学习传统ML算法？**
A: 因为它们提供了很多后面章节仍然在用的基本思想，比如损失函数、间隔、正则化、集成和结构化决策。

**Q: SVM的核方法有什么用？**
A: 核方法通过隐式地将数据映射到高维空间，使非线性问题变为线性问题，这个思想在深度学习中也很重要。

**Q: 决策树和随机森林与神经网络有什么关系？**
A: 决策树和随机森林体现了"多模型/多路径结果融合"的思想。Transformer 多头注意力也会融合多个注意力头的表示，但它是端到端学习的表示机制，不等同于随机森林。

## 扩展内容

本章提供四个深度扩展文档，适合想深入理解优化、机器学习和工程系统决策的读者：

### 高级优化话题 — [📖 阅读](extensions/advanced_optimization.md)
- **E2.1 二阶优化方法** — 牛顿法、拟牛顿法、L-BFGS
- **E2.2 随机优化理论** — 收敛速度分析、方差缩减
- **E2.3 非凸优化** — 深度学习中的非凸性、鞍点、过度参数化
- **E2.4 学习率调度** — 预热、余弦退火、周期性重启
- **E2.5 正则化与优化** — L1/L2正则化、批归一化
- **E2.6 优化与泛化** — 双重下降现象、隐式正则化
- **E2.7 分布式优化** — 数据并行、模型并行、联邦学习
- **E2.8 与LLM训练的连接** — 大规模训练实践、梯度检查、损失曲线分析
- **与深度学习的联系** — 为什么深度学习不使用二阶方法、何时使用、与Adam的对比

### 凸分析基础 — [📖 阅读](extensions/convex_analysis.md)
- **凸集和凸函数** — 定义、性质、判断方法
- **凸优化问题** — 标准形式、性质、例子
- **非凸优化** — 深度学习中的非凸性、为什么仍然有效
- **对偶问题** — Lagrange对偶、强对偶性、SVM应用
- **与深度学习的联系** — 为什么深度学习是非凸的、凸性的意义、具体应用例子

### 运筹学基础 — [📖 阅读](extensions/operations_research_basics.md)
- **线性规划** — 资源分配、成本最小化和约束表达
- **整数规划** — 路由、调度、开关和离散决策
- **动态规划** — 多步系统中的长期成本和序列决策
- **排队论** — 吞吐、等待时间、服务率和长尾延迟
- **与 LLM Serving 的联系** — 批处理、模型路由、缓存、容量规划和 SLO 成本优化

### 树数据结构基础 — [📖 阅读](extensions/tree_data_structures.md)
- **树的基本概念** — 节点、边、深度、高度
- **决策树的树结构** — 根节点、内部节点、叶子节点
- **树的遍历与预测** — 前序遍历、中序遍历、后序遍历
- **二叉树** — 定义、性质、与决策树的关系
- **树的复杂度分析** — 构建、查询、空间复杂度
- **树的递归实现** — 节点定义、递归构建、递归预测
- **树的剪枝** — 前剪枝、后剪枝、防止过拟合

## 关键连接点

### 梯度下降 → 优化器

```
基础梯度下降：w = w - α * ∇L
                    ↓
自适应优化器（Adam）：根据历史梯度调整学习率
                    ↓
深度学习训练：所有神经网络都用优化器训练
```

**启示：** 优化器是梯度下降的改进版本，是深度学习的核心。

### 传统ML → 深度学习

```
线性回归 → 神经网络（添加隐层和非线性激活）
逻辑回归 → 深度分类网络（多层堆叠）
SVM核方法 → 深度学习中的特征学习（自动学习特征）
集成学习 → 多头注意力的启发式类比（多路表示的融合）
```

**启示：** 深度学习是传统ML的自然延伸，通过自动特征学习和多层堆叠实现更强的表达能力。

### 优化 → LLM训练

```
优化算法（梯度下降、Adam）
        ↓
神经网络训练（反向传播）
        ↓
深度学习（CNN、RNN、Transformer）
        ↓
大语言模型（LLM）训练
```

**启示：** 理解优化算法是理解LLM训练的基础。


**目标4：准备深度学习学习（1.5小时）**
1. 阅读 [2.1 梯度下降基础](01_gradient_descent.md)
2. 阅读 [2.2 自适应优化器](02_adaptive_optimizers.md)
3. 阅读 [2.3 优化与传统ML](03_optimization_and_traditional_ml.md)
4. 运行优化算法实验（实验1、2）
5. 准备进入第3章：深度学习基础

### 按学习风格选择方式

**理论优先型：** 先读文档，再看代码
1. 阅读所有主文档
2. 理解核心概念
3. 运行代码实验验证理解

**实践优先型：** 先运行代码，再读文档
1. 运行代码实验
2. 观察实验结果
3. 阅读文档理解原理

**混合型：** 交替进行
1. 阅读一个小节
2. 运行相关代码实验
3. 分析实验结果
4. 继续下一个小节

---

**下一步：** 阅读 [2.1 梯度下降基础](01_gradient_descent.md)
