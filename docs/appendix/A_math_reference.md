# 附录A：数学备忘

\*\*版本：\*\* v1.1  
**最后更新：** 2026-05-13

本附录收集了全书中常用的数学符号、公式和概念，方便查阅。

---

## A.1 线性代数

### 向量和矩阵

| 符号 | 含义 | 例子 |
|------|------|------|
| $\mathbf{x}$ | 列向量 | $\begin{bmatrix} 1 \\ 2 \\ 3 \end{bmatrix}$ |
| $\mathbf{x}^T$ | 行向量（转置） | $\begin{bmatrix} 1 & 2 & 3 \end{bmatrix}$ |
| $\mathbf{A}$ | 矩阵 | $\begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$ |
| $\mathbf{A}^T$ | 矩阵转置 | 行列互换 |
| $\mathbf{A}^{-1}$ | 矩阵逆 | $\mathbf{A} \mathbf{A}^{-1} = \mathbf{I}$ |
| $\mathbf{I}$ | 单位矩阵 | 对角线为1，其他为0 |

### 矩阵运算

**矩阵乘法：**
$$(\mathbf{A}\mathbf{B})_{ij} = \sum_k A_{ik} B_{kj}$$

**向量点积：**
$$\mathbf{x}^T \mathbf{y} = \sum_i x_i y_i$$

**矩阵范数（Frobenius范数）：**
$$\|\mathbf{A}\|_F = \sqrt{\sum_{i,j} A_{ij}^2}$$

**向量范数（L2范数）：**
$$\|\mathbf{x}\|_2 = \sqrt{\sum_i x_i^2}$$

### 矩阵求导

**标量对向量求导：**
$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{a}^T \mathbf{x}) = \mathbf{a}$$

$$\frac{\partial}{\partial \mathbf{x}} (\mathbf{x}^T \mathbf{A} \mathbf{x}) = (\mathbf{A} + \mathbf{A}^T) \mathbf{x}$$

**链式法则：**
$$\frac{\partial L}{\partial \mathbf{x}} = \frac{\partial L}{\partial \mathbf{y}} \frac{\partial \mathbf{y}}{\partial \mathbf{x}}$$

---

## A.2 概率与统计

### 高斯分布

**概率密度函数（PDF）：**
$$p(x) = \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**多元高斯分布：**
$$p(\mathbf{x}) = \frac{1}{(2\pi)^{n/2}|\mathbf{\Sigma}|^{1/2}} \exp\left(-\frac{1}{2}(\mathbf{x}-\boldsymbol{\mu})^T \mathbf{\Sigma}^{-1} (\mathbf{x}-\boldsymbol{\mu})\right)$$

其中 $\boldsymbol{\mu}$ 是均值，$\mathbf{\Sigma}$ 是协方差矩阵。

### 期望和方差

**期望：**
$$E[X] = \sum_x x \cdot p(x) \quad \text{（离散）}$$
$$E[X] = \int x \cdot p(x) dx \quad \text{（连续）}$$

**方差：**
$$\text{Var}(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2$$

**协方差：**
$$\text{Cov}(X, Y) = E[(X - E[X])(Y - E[Y])]$$

### 常用分布

| 分布 | 参数 | 均值 | 方差 |
|------|------|------|------|
| 高斯 | $\mu, \sigma^2$ | $\mu$ | $\sigma^2$ |
| 均匀 | $a, b$ | $(a+b)/2$ | $(b-a)^2/12$ |
| 伯努利 | $p$ | $p$ | $p(1-p)$ |

---

## A.2.5 复数基础

复数在信号处理和深度学习中无处不在。本节介绍复数的基本概念和运算。

### 复数的定义和表示

**代数形式：**
$$z = a + jb$$

其中 $a$ 是实部，$b$ 是虚部，$j$ 是虚数单位（$j^2 = -1$）。

**极坐标形式：**
$$z = r e^{j\theta} = r(\cos\theta + j\sin\theta)$$

其中 $r = |z| = \sqrt{a^2 + b^2}$ 是模（幅度），$\theta = \arg(z) = \arctan(b/a)$ 是幅角（相位）。

### 复数的运算

**加法和减法：**
$$(a + jb) + (c + jd) = (a+c) + j(b+d)$$

**乘法：**
$$(a + jb)(c + jd) = (ac - bd) + j(ad + bc)$$

**除法：**
$$\frac{a + jb}{c + jd} = \frac{(a + jb)(c - jd)}{c^2 + d^2} = \frac{ac + bd}{c^2 + d^2} + j\frac{bc - ad}{c^2 + d^2}$$

**共轭：**
$$z^* = a - jb$$

**模和幅角：**
$$|z| = \sqrt{a^2 + b^2}, \quad \arg(z) = \arctan(b/a)$$

### 欧拉公式

**最重要的恒等式：**
$$e^{j\theta} = \cos\theta + j\sin\theta$$

**推论：**
$$\cos\theta = \frac{e^{j\theta} + e^{-j\theta}}{2}, \quad \sin\theta = \frac{e^{j\theta} - e^{-j\theta}}{2j}$$

### 复数在信号处理中的应用

**为什么傅里叶变换使用复指数？**

复指数 $e^{j2\pi ft}$ 是一个旋转的向量，其实部是 $\cos(2\pi ft)$，虚部是 $\sin(2\pi ft)$。这样可以同时编码信号的幅度和相位信息。

**复信号的表示：**
$$x(t) = A e^{j(2\pi ft + \phi)} = A[\cos(2\pi ft + \phi) + j\sin(2\pi ft + \phi)]$$

其中 $A$ 是幅度，$f$ 是频率，$\phi$ 是初相位。

**复信号的功率：**
$$P = |x(t)|^2 = x(t) \cdot x^*(t)$$

### 复数在深度学习中的应用

虽然深度学习通常使用实数，但复数的思想在以下场景中很有用：

- **傅里叶特征**：在频域处理信号时使用复数表示
- **相位信息**：某些应用需要保留信号的相位信息
- **复值神经网络**：研究领域，用于处理复信号

---

## A.3 信号处理

### 傅里叶变换

**离散傅里叶变换（DFT）：**
$$X[k] = \sum_{n=0}^{N-1} x[n] e^{-j2\pi kn/N}$$

**逆变换：**
$$x[n] = \frac{1}{N} \sum_{k=0}^{N-1} X[k] e^{j2\pi kn/N}$$

### 卷积

**离散卷积：**
$$y[n] = \sum_{m=-\infty}^{\infty} x[m] h[n-m]$$

**性质：**
- 交换律：$x * h = h * x$
- 结合律：$(x * h_1) * h_2 = x * (h_1 * h_2)$
- 频域乘法：$\mathcal{F}(x * h) = \mathcal{F}(x) \cdot \mathcal{F}(h)$

### 能量和功率

**信号能量：**
$$E = \sum_{n=-\infty}^{\infty} |x[n]|^2$$

**信号功率：**
$$P = \lim_{N \to \infty} \frac{1}{2N+1} \sum_{n=-N}^{N} |x[n]|^2$$

---

## A.4 优化

### 梯度下降

**更新规则：**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

其中 $\alpha$ 是学习率，$\nabla L$ 是损失函数的梯度。

### 常用损失函数

**均方误差（MSE）：**
$$L = \frac{1}{N} \sum_{i=1}^{N} (y_i - \hat{y}_i)^2$$

**交叉熵（分类）：**
$$L = -\frac{1}{N} \sum_{i=1}^{N} \sum_{c=1}^{C} y_{ic} \log(\hat{y}_{ic})$$

**L2正则化：**
$$L_{\text{reg}} = L + \lambda \sum_i w_i^2$$

### 优化器

**SGD（随机梯度下降）：**
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \nabla L(\mathbf{w}_t)$$

**Adam：**
$$m_t = \beta_1 m_{t-1} + (1-\beta_1) \nabla L(\mathbf{w}_t)$$
$$v_t = \beta_2 v_{t-1} + (1-\beta_2) (\nabla L(\mathbf{w}_t))^2$$
$$\mathbf{w}_{t+1} = \mathbf{w}_t - \alpha \frac{m_t}{\sqrt{v_t} + \epsilon}$$

---

## A.5 激活函数

| 函数 | 公式 | 导数 | 用途 |
|------|------|------|------|
| ReLU | $\max(0, x)$ | $\mathbb{1}_{x>0}$ | 隐层 |
| Sigmoid | $\frac{1}{1+e^{-x}}$ | $\sigma(x)(1-\sigma(x))$ | 二分类输出 |
| Tanh | $\frac{e^x - e^{-x}}{e^x + e^{-x}}$ | $1 - \tanh^2(x)$ | RNN隐层 |
| Softmax | $\frac{e^{x_i}}{\sum_j e^{x_j}}$ | $\text{diag}(p) - pp^T$ | 多分类输出 |

---

## A.6 常用符号

| 符号 | 含义 |
|------|------|
| $\sum$ | 求和 |
| $\prod$ | 求积 |
| $\int$ | 积分 |
| $\partial$ | 偏导数 |
| $\nabla$ | 梯度 |
| $\mathbb{E}$ | 期望 |
| $\mathbb{R}$ | 实数集 |
| $\mathbb{C}$ | 复数集 |
| $j$ 或 $i$ | 虚数单位（$j^2 = -1$） |
| $\approx$ | 近似等于 |
| $\propto$ | 正比于 |

---

## A.7 常用恒等式

**三角恒等式：**
$$\sin^2(x) + \cos^2(x) = 1$$
$$e^{jx} = \cos(x) + j\sin(x)$$

**对数恒等式：**
$$\log(ab) = \log(a) + \log(b)$$
$$\log(a^b) = b\log(a)$$

**指数恒等式：**
$$e^{a+b} = e^a e^b$$
$$(e^a)^b = e^{ab}$$

---

## A.8 快速参考

### 常见求导

$$\frac{d}{dx} x^n = nx^{n-1}$$
$$\frac{d}{dx} e^x = e^x$$
$$\frac{d}{dx} \log(x) = \frac{1}{x}$$
$$\frac{d}{dx} \sin(x) = \cos(x)$$

### 常见积分

$$\int x^n dx = \frac{x^{n+1}}{n+1} + C \quad (n \neq -1)$$
$$\int e^x dx = e^x + C$$
$$\int \frac{1}{x} dx = \log|x| + C$$

---

## A.9 树数据结构

### 树的基本定义

**树**是一种分层的数据结构，由节点和边组成。

**树的性质：**
- 有一个根节点（没有父节点）
- 除根节点外，每个节点有唯一的父节点
- $n$ 个节点的树有 $n-1$ 条边
- 树是连通的无环图

**树的基本概念：**

| 概念 | 定义 |
|------|------|
| 根节点 | 没有父节点的节点 |
| 叶子节点 | 没有子节点的节点 |
| 内部节点 | 有子节点的节点 |
| 深度 | 从根节点到该节点的边数 |
| 高度 | 从该节点到最深叶子节点的边数 |
| 度数 | 节点的子节点个数 |
| 树的高度 | 从根节点到最深叶子节点的边数 |

### 二叉树

**二叉树**是每个节点最多有两个子节点的树。

**二叉树的性质：**
- 第 $i$ 层最多有 $2^{i-1}$ 个节点
- 高度为 $h$ 的二叉树最多有 $2^h - 1$ 个节点
- 完全二叉树：除最后一层外，每层都是满的

**决策树通常是二叉树：**
- 每个内部节点代表一个特征判断
- 左子树：判断结果为"是"
- 右子树：判断结果为"否"

### 树的复杂度分析

**构建树的时间复杂度：**
- 最坏情况：$O(n^2)$（树退化为链表）
- 平衡树：$O(n \log n)$
- 决策树（贪心分割）：$O(n \log n)$

**查询树的时间复杂度：**
- 最坏情况：$O(h)$（h是树的高度）
- 平衡树：$O(\log n)$
- 决策树预测：$O(h)$

**空间复杂度：**
- $O(n)$（n是节点数）

### 树的遍历算法

**前序遍历（Pre-order）：** 先访问节点，再访问子节点
```
访问节点 → 遍历左子树 → 遍历右子树
```
用途：决策树预测、表达式求值

**中序遍历（In-order）：** 先访问左子树，再访问节点，最后访问右子树
```
遍历左子树 → 访问节点 → 遍历右子树
```
用途：二叉搜索树排序

**后序遍历（Post-order）：** 先访问子节点，再访问节点
```
遍历左子树 → 遍历右子树 → 访问节点
```
用途：删除树、计算树的高度

### 递归的数学基础

**递归的定义：** 函数调用自身来解决问题的方法

**递归的三个要素：**
1. **基础情况（Base case）**：递归的停止条件
2. **递归情况（Recursive case）**：函数调用自身
3. **递推关系（Recurrence relation）**：问题规模的缩小

**例子：计算树的高度**
```
height(node) = {
  0,                                    if node is leaf
  1 + max(height(left), height(right)), if node is internal
}
```

**递归的时间复杂度分析：**
- 使用递推关系求解
- 例如：$T(n) = 2T(n/2) + O(n)$ → $T(n) = O(n \log n)$（主定理）

---

**版本：** v0.1 (开发中)  
**最后更新：** 2026-05-13

---

## A.9 Transformer与注意力机制

### 缩放点积注意力

**注意力权重：**
$$\text{Attention}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{softmax}\left(\frac{\mathbf{Q}\mathbf{K}^T}{\sqrt{d_k}}\right) \mathbf{V}$$

其中 $d_k$ 是键向量的维度。

### 多头注意力

**多头注意力输出：**
$$\text{MultiHead}(\mathbf{Q}, \mathbf{K}, \mathbf{V}) = \text{Concat}(\text{head}_1, \ldots, \text{head}_h) \mathbf{W}^O$$

其中每个 head 计算：
$$\text{head}_i = \text{Attention}(\mathbf{Q}\mathbf{W}_i^Q, \mathbf{K}\mathbf{W}_i^K, \mathbf{V}\mathbf{W}_i^V)$$

### 位置编码

**正弦位置编码：**
$$PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$
$$PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{model}}}\right)$$

---

## A.10 LLM相关概念

### KL散度

**定义：**
$$D_{KL}(P \| Q) = \sum_x P(x) \log\frac{P(x)}{Q(x)}$$

用于衡量两个概率分布的差异。

### 困惑度（Perplexity）

**定义：**
$$\text{Perplexity} = e^{-\frac{1}{N}\sum_{i=1}^{N} \log P(x_i)}$$

用于评估语言模型的性能。

### 缩放律（Scaling Laws）

**Chinchilla缩放律：**
$$L(N, D) \approx E + \frac{A}{N^\alpha} + \frac{B}{D^\beta}$$

其中 $N$ 是模型参数数，$D$ 是训练数据量，$\alpha \approx \beta \approx 0.07$。

---

**版本：** v1.1 (v2.0 更新)
**最后更新：** 2026-05-26
