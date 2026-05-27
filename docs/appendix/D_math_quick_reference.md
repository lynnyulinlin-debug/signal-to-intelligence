# 附录D：数学基础速查表

**版本：** v1.0  
**最后更新：** 2026-05-27

本速查表提供了教程中所有数学基础概念的一页纸总结，方便快速查阅。

---

## 1. 复数基础

| 概念 | 公式 | 应用 |
|------|------|------|
| 复数 | $z = a + jb$ | 傅里叶变换、信号处理 |
| 极坐标 | $z = re^{j\theta}$ | 幅度和相位表示 |
| 欧拉公式 | $e^{j\theta} = \cos\theta + j\sin\theta$ | 复指数、旋转 |
| 共轭 | $z^* = a - jb$ | 实部提取 |
| 模 | $\|z\| = \sqrt{a^2 + b^2}$ | 幅度 |

**关键应用：** 傅里叶变换中的复指数 $e^{j2\pi ft}$ 同时编码幅度和相位

---

## 2. 数值方法

| 概念 | 公式/条件 | 应用 |
|------|----------|------|
| 浮点精度 | $\text{computed} = \text{true} \times (1 + \epsilon)$ | 数值稳定性 |
| 条件数 | $\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|$ | 问题难度 |
| 有限差分 | $\frac{\partial f}{\partial x} \approx \frac{f(x+h) - f(x-h)}{2h}$ | 梯度检查 |
| 最优步长 | $h \approx \sqrt{\epsilon}$ | 数值梯度 |
| 梯度消失 | $\frac{\partial L}{\partial w_1} = \prod_i \frac{\partial}{\partial w_i} < 1$ | 深层网络 |

**关键应用：** 理解为什么某些优化算法会失败，如何确保数值稳定性

---

## 3. 图论基础

| 概念 | 定义 | 应用 |
|------|------|------|
| 图 | $G = (V, E)$ | 网络、关系 |
| 邻接矩阵 | $A_{ij} = w_{ij}$ if $(i,j) \in E$ | 图表示 |
| 度 | $d_i = \sum_j A_{ij}$ | 节点连接数 |
| 拉普拉斯矩阵 | $\mathbf{L} = \mathbf{D} - \mathbf{A}$ | 图谱分析 |
| 注意力权重 | $\alpha_{ij} = \text{softmax}(Q_i K_j^T)$ | Transformer 中的图 |

**关键应用：** 理解注意力机制作为图上的消息传递

---

## 4. 向量空间基础

| 概念 | 公式 | 应用 |
|------|------|------|
| 内积 | $\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v}$ | 相似度 |
| L2 范数 | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ | 欧几里得距离 |
| 余弦相似度 | $\cos\theta = \frac{\mathbf{u}^T \mathbf{v}}{\|\mathbf{u}\| \|\mathbf{v}\|}$ | 方向相似度 |
| 投影 | $\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2} \mathbf{v}$ | 子空间投影 |
| 正交基 | $\langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0$ ($i \neq j$) | 独立表示 |

**关键应用：** 理解 Transformer 中的嵌入空间和注意力计算

---

## 5. 信息论基础

| 概念 | 公式 | 应用 |
|------|------|------|
| 熵 | $H(X) = -\sum_x P(x) \log P(x)$ | 不确定性 |
| 交叉熵 | $H(P,Q) = -\sum_x P(x) \log Q(x)$ | LLM 损失函数 |
| KL 散度 | $D_{KL}(P\|Q) = \sum_x P(x) \log \frac{P(x)}{Q(x)}$ | 分布差异 |
| 互信息 | $I(X;Y) = H(X) - H(X\|Y)$ | 变量依赖 |
| 困惑度 | $\text{PPL} = e^{H(P,Q)}$ | 模型评估 |

**关键应用：** 理解 LLM 的训练目标和评估指标

---

## 6. 凸分析基础

| 概念 | 定义/条件 | 应用 |
|------|----------|------|
| 凸集 | $\lambda \mathbf{x} + (1-\lambda) \mathbf{y} \in C$ | 可行域 |
| 凸函数 | $f(\lambda \mathbf{x} + (1-\lambda) \mathbf{y}) \leq \lambda f(\mathbf{x}) + (1-\lambda) f(\mathbf{y})$ | 优化问题 |
| Hessian 半正定 | $\nabla^2 f(\mathbf{x}) \succeq 0$ | 凸性判断 |
| 凸优化 | 最小化凸函数在凸集上 | 全局最优解 |
| 非凸优化 | 深度学习中的优化 | 局部最优解 |

**关键应用：** 理解优化问题的难度和可解性

---

## 7. 随机过程进阶

| 概念 | 定义/公式 | 应用 |
|------|----------|------|
| 平稳过程 | $E[X(t)] = \mu$, $R(t_1, t_2) = R(\tau)$ | 时间序列 |
| 马尔可夫链 | $P(X_{n+1}\|X_n, \ldots) = P(X_{n+1}\|X_n)$ | 序列模型 |
| 转移矩阵 | $P_{ij} = P(X_{n+1}=j\|X_n=i)$ | 状态转移 |
| AR(p) 模型 | $X_t = \sum_i \phi_i X_{t-i} + \epsilon_t$ | 时间序列预测 |
| 自相关函数 | $\rho(\tau) = \frac{R(\tau) - \mu^2}{\sigma^2}$ | 相关结构 |

**关键应用：** 理解时间序列的动态特性和预测

---

## 8. 线性系统理论

| 概念 | 公式/定义 | 应用 |
|------|----------|------|
| 冲激响应 | $h[n] = T(\delta[n])$ | 系统特性 |
| 卷积 | $y[n] = \sum_k x[k]h[n-k]$ | 系统输出 |
| 频率响应 | $H(e^{j\omega}) = \sum_n h[n]e^{-j\omega n}$ | 频域特性 |
| 传递函数 | $H(z) = \frac{B(z)}{A(z)}$ | 系统表示 |
| 稳定性 | $\|p_i\| < 1$ (所有极点) | 系统可用性 |
| 状态空间 | $\mathbf{x}[n+1] = \mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n]$ | 现代控制 |

**关键应用：** 理解滤波器设计和系统稳定性

---

## 📊 概念关系图

```
线性代数 (矩阵、特征值)
    ↓
    ├─→ 向量空间基础 → Transformer 嵌入
    ├─→ 图论基础 → 注意力机制
    └─→ 凸分析基础 → 优化问题

微积分 (导数、梯度)
    ↓
    ├─→ 数值方法 → 数值稳定性
    └─→ 优化理论 → 梯度下降

概率论 (分布、期望)
    ↓
    ├─→ 信息论基础 → LLM 损失函数
    ├─→ 随机过程 → 时间序列
    └─→ 统计学 → 参数估计

信号处理 (傅里叶、卷积)
    ↓
    ├─→ 复数基础 → 复指数
    ├─→ 线性系统 → 滤波器
    └─→ 随机过程 → 信号分析
```

---

## 🔑 关键公式速查

### 线性代数
- 矩阵乘法：$(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik}B_{kj}$
- 特征值：$\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$
- 行列式：$\det(\mathbf{A}) = \prod_i \lambda_i$

### 微积分
- 链式法则：$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$
- 梯度：$\nabla f = [\frac{\partial f}{\partial x_1}, \ldots, \frac{\partial f}{\partial x_n}]^T$
- Hessian：$H_{ij} = \frac{\partial^2 f}{\partial x_i \partial x_j}$

### 概率论
- 期望：$E[X] = \sum_x x P(x)$
- 方差：$\text{Var}(X) = E[X^2] - (E[X])^2$
- 贝叶斯：$P(A|B) = \frac{P(B|A)P(A)}{P(B)}$

### 信息论
- 熵：$H(X) = -\sum_x P(x) \log_2 P(x)$ (bits)
- 交叉熵：$H(P,Q) = -\sum_x P(x) \log_2 Q(x)$
- KL 散度：$D_{KL}(P||Q) = \sum_x P(x) \log_2 \frac{P(x)}{Q(x)}$

### 信号处理
- 傅里叶变换：$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$
- 卷积：$y[n] = \sum_{m} x[m] h[n-m]$
- 功率谱：$S(f) = |X(f)|^2$

---

## 📚 各章节的数学基础

| 章节 | 主要数学基础 | 新增补充 |
|------|------------|---------|
| Ch1 DSP | 线性代数、微积分、傅里叶 | 复数基础、随机过程、线性系统 |
| Ch2 优化 | 微积分、线性代数 | 数值方法、凸分析 |
| Ch3 深度学习 | 微积分、线性代数 | 数值方法、向量空间 |
| Ch4 Transformer | 线性代数、微积分 | 图论、向量空间 |
| Ch5 LLM | 概率论、信息论 | 信息论基础 |
| Ch6-8 应用 | 所有基础 | 综合应用 |

---

## 💡 快速查找指南

**我想理解...**

- **Transformer 中的注意力** → 图论基础 + 向量空间基础
- **LLM 的损失函数** → 信息论基础
- **为什么梯度消失** → 数值方法
- **滤波器设计** → 线性系统理论
- **时间序列预测** → 随机过程进阶
- **优化问题的难度** → 凸分析基础
- **复数在信号中的作用** → 复数基础
- **向量的相似度** → 向量空间基础

---

## 🎯 学习建议

### 初学者路径
1. 复数基础（理解傅里叶变换）
2. 向量空间基础（理解嵌入）
3. 信息论基础（理解损失函数）

### 进阶学习路径
1. 数值方法（理解实现细节）
2. 图论基础（理解注意力机制）
3. 凸分析基础（理解优化）

### 深度学习路径
1. 线性系统理论（理解 RNN）
2. 随机过程进阶（理解时间序列）
3. 所有内容的综合应用

---

## 📖 符号速查

| 符号 | 含义 | 例子 |
|------|------|------|
| $\mathbf{x}$ | 向量 | $\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ |
| $\mathbf{A}$ | 矩阵 | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| $\mathbf{A}^T$ | 转置 | 行列互换 |
| $\mathbf{A}^{-1}$ | 逆矩阵 | $\mathbf{A}\mathbf{A}^{-1} = \mathbf{I}$ |
| $\|\mathbf{x}\|$ | 范数 | $\sqrt{\sum_i x_i^2}$ |
| $\langle \mathbf{u}, \mathbf{v} \rangle$ | 内积 | $\mathbf{u}^T \mathbf{v}$ |
| $\nabla f$ | 梯度 | 偏导数向量 |
| $\mathbb{E}[X]$ | 期望 | 平均值 |
| $\mathbb{P}(X)$ | 概率 | 0 到 1 之间 |
| $j$ 或 $i$ | 虚数单位 | $j^2 = -1$ |

---

## ✨ 使用建议

1. **快速查阅**：遇到陌生概念时，先查这个表
2. **复习**：学完一章后，用这个表复习关键概念
3. **对比**：比较不同概念之间的关系
4. **应用**：查找概念的实际应用场景

---

**版本历史：**
- v1.0 (2026-05-27)：初始版本，包含所有 Plan B 和 Plan C 的数学基础
