# 4.1 向量空间基础

**核心问题：** 什么是向量空间？为什么 Transformer 中的嵌入是向量？

---

## 为什么需要理解向量空间

在 Transformer 中：
- 每个 token 被表示为一个**向量**（嵌入）
- 注意力计算涉及**向量的内积**
- 不同层的表示存在于不同的**向量空间**中

理解向量空间的几何性质可以帮助我们理解 Transformer 如何处理信息。

---

## 向量空间的基本概念

### 向量空间的定义

一个向量空间 $V$ 是一个集合，满足以下性质：

1. **加法封闭性**：$\mathbf{u}, \mathbf{v} \in V \Rightarrow \mathbf{u} + \mathbf{v} \in V$
2. **标量乘法封闭性**：$\mathbf{v} \in V, c \in \mathbb{R} \Rightarrow c\mathbf{v} \in V$
3. **零向量**：存在 $\mathbf{0} \in V$，使得 $\mathbf{v} + \mathbf{0} = \mathbf{v}$
4. **加法逆元**：对每个 $\mathbf{v} \in V$，存在 $-\mathbf{v} \in V$

### 维度和基

**基（Basis）**：向量空间的一组线性无关的向量，可以生成整个空间。

**维度（Dimension）**：基中向量的个数。

**例子：**
- $\mathbb{R}^2$（平面）的维度是 2，基可以是 $\{(1,0), (0,1)\}$
- $\mathbb{R}^{768}$（BERT 的嵌入维度）的维度是 768

### 子空间

**子空间**是向量空间的一个子集，本身也是向量空间。

**例子：**
- 通过原点的直线是 $\mathbb{R}^2$ 的子空间
- 通过原点的平面是 $\mathbb{R}^3$ 的子空间

---

## 内积和范数

### 内积（Inner Product）

**定义：** 内积 $\langle \mathbf{u}, \mathbf{v} \rangle$ 是一个标量，满足：

1. **对称性**：$\langle \mathbf{u}, \mathbf{v} \rangle = \langle \mathbf{v}, \mathbf{u} \rangle$
2. **线性性**：$\langle a\mathbf{u} + b\mathbf{v}, \mathbf{w} \rangle = a\langle \mathbf{u}, \mathbf{w} \rangle + b\langle \mathbf{v}, \mathbf{w} \rangle$
3. **正定性**：$\langle \mathbf{v}, \mathbf{v} \rangle \geq 0$，等号成立当且仅当 $\mathbf{v} = \mathbf{0}$

**标准内积（点积）：**
$$\langle \mathbf{u}, \mathbf{v} \rangle = \mathbf{u}^T \mathbf{v} = \sum_i u_i v_i$$

### 范数（Norm）

**定义：** 范数 $\|\mathbf{v}\|$ 是向量的"长度"，满足：

1. **正定性**：$\|\mathbf{v}\| \geq 0$，等号成立当且仅当 $\mathbf{v} = \mathbf{0}$
2. **齐次性**：$\|c\mathbf{v}\| = |c| \|\mathbf{v}\|$
3. **三角不等式**：$\|\mathbf{u} + \mathbf{v}\| \leq \|\mathbf{u}\| + \|\mathbf{v}\|$

**常用范数：**

| 范数 | 定义 | 用途 |
|------|------|------|
| L2 范数 | $\|\mathbf{v}\|_2 = \sqrt{\sum_i v_i^2}$ | 欧几里得距离 |
| L1 范数 | $\|\mathbf{v}\|_1 = \sum_i \|v_i\|$ | 稀疏性 |
| L∞ 范数 | $\|\mathbf{v}\|_\infty = \max_i \|v_i\|$ | 最大分量 |
| Frobenius 范数 | $\|\mathbf{A}\|_F = \sqrt{\sum_{ij} A_{ij}^2}$ | 矩阵范数 |

### 内积与范数的关系

$$\|\mathbf{v}\|^2 = \langle \mathbf{v}, \mathbf{v} \rangle$$

---

## 正交性和投影

### 正交向量

**定义：** 两个向量 $\mathbf{u}$ 和 $\mathbf{v}$ 正交，当且仅当：

$$\langle \mathbf{u}, \mathbf{v} \rangle = 0$$

**几何意义：** 两个向量垂直。

### 正交基

**定义：** 一组向量 $\{\mathbf{v}_1, \mathbf{v}_2, \ldots, \mathbf{v}_n\}$ 是正交基，如果：

1. 它们两两正交：$\langle \mathbf{v}_i, \mathbf{v}_j \rangle = 0$ （$i \neq j$）
2. 它们生成整个空间

**标准正交基**：如果还满足 $\|\mathbf{v}_i\| = 1$，则称为标准正交基。

**优势：** 在标准正交基下，坐标计算很简单。

### 投影（Projection）

**向量在另一个向量上的投影：**

$$\text{proj}_{\mathbf{v}} \mathbf{u} = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{v}\|^2} \mathbf{v}$$

**几何意义：** $\mathbf{u}$ 在 $\mathbf{v}$ 方向上的"影子"。

**向量在子空间上的投影：**

如果 $\{\mathbf{v}_1, \ldots, \mathbf{v}_k\}$ 是子空间的正交基，则：

$$\text{proj}_V \mathbf{u} = \sum_{i=1}^{k} \frac{\langle \mathbf{u}, \mathbf{v}_i \rangle}{\|\mathbf{v}_i\|^2} \mathbf{v}_i$$

---

## 距离度量

### 欧几里得距离

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_2 = \sqrt{\sum_i (u_i - v_i)^2}$$

### 余弦相似度

$$\cos(\theta) = \frac{\langle \mathbf{u}, \mathbf{v} \rangle}{\|\mathbf{u}\|_2 \|\mathbf{v}\|_2}$$

**范围：** $[-1, 1]$
- 1：完全相同方向
- 0：正交
- -1：完全相反方向

**应用：** 在 Transformer 中，注意力权重基于余弦相似度。

### 曼哈顿距离

$$d(\mathbf{u}, \mathbf{v}) = \|\mathbf{u} - \mathbf{v}\|_1 = \sum_i |u_i - v_i|$$

---

## 线性变换

### 定义

**线性变换** $T: V \to W$ 满足：

$$T(a\mathbf{u} + b\mathbf{v}) = aT(\mathbf{u}) + bT(\mathbf{v})$$

### 矩阵表示

任何线性变换都可以用矩阵表示：

$$T(\mathbf{v}) = \mathbf{A}\mathbf{v}$$

### 特征值和特征向量

**定义：** 如果 $\mathbf{A}\mathbf{v} = \lambda \mathbf{v}$，则 $\mathbf{v}$ 是特征向量，$\lambda$ 是特征值。

**几何意义：** 特征向量在线性变换下只改变长度，不改变方向。

**应用：** 
- 主成分分析（PCA）
- 图的谱分析
- 矩阵分解

---

## 向量空间在 Transformer 中的应用

### 1. 嵌入空间

每个 token 被映射到一个 $d_{model}$ 维的向量空间：

```
token "hello" → [0.2, -0.5, 0.1, ..., 0.3]  (768维)
```

### 2. 注意力计算中的内积

注意力权重基于查询和键向量的内积：

$$\text{attention}_{ij} = \frac{Q_i \cdot K_j^T}{\sqrt{d_k}}$$

这衡量了两个向量的**相似度**。

### 3. 值的加权求和

输出是值向量的加权组合：

$$\text{output}_i = \sum_j \alpha_{ij} V_j$$

这是在向量空间中的**线性组合**。

### 4. 多头注意力中的子空间

每个注意力头在一个**子空间**中操作：

```
原始空间：d_model 维
    ↓ (投影)
子空间1：d_model/h 维
子空间2：d_model/h 维
...
子空间h：d_model/h 维
    ↓ (拼接)
原始空间：d_model 维
```

### 5. 嵌入空间的几何性质

在训练过程中，Transformer 学习一个嵌入空间，其中：
- **相似的 token 靠近**（小欧几里得距离）
- **相关的概念在同一方向**（高余弦相似度）
- **不同的概念正交**（低内积）

---

## 向量空间的可视化

### 二维投影

高维向量空间可以用 t-SNE 或 PCA 投影到 2D 进行可视化：

```python
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 嵌入矩阵：(n_tokens, d_model)
embeddings = model.get_embeddings()

# 投影到 2D
pca = PCA(n_components=2)
embeddings_2d = pca.fit_transform(embeddings)

# 绘制
plt.scatter(embeddings_2d[:, 0], embeddings_2d[:, 1])
plt.show()
```

### 相似度矩阵

可视化向量之间的相似度：

```python
import numpy as np

# 计算余弦相似度矩阵
similarity = embeddings @ embeddings.T
similarity /= np.linalg.norm(embeddings, axis=1, keepdims=True)
similarity /= np.linalg.norm(embeddings, axis=1, keepdims=True).T

# 绘制热力图
plt.imshow(similarity, cmap='hot')
plt.colorbar()
plt.show()
```

---

## 实践建议

### 向量空间的性质检查

在训练 Transformer 时，可以检查：

- [ ] **嵌入的范数**：是否稳定？
- [ ] **相似度分布**：是否合理？
- [ ] **子空间的独立性**：多头注意力的子空间是否不同？
- [ ] **投影的损失**：降维后信息损失多少？

### 常见问题

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 嵌入范数爆炸 | 学习率太大或初始化不当 | 使用层归一化，调整学习率 |
| 所有向量相似 | 模型未充分训练或容量不足 | 增加训练时间或模型大小 |
| 子空间重叠 | 多头注意力冗余 | 增加头数或使用正则化 |

---

## 关键要点

1. **向量空间是 Transformer 的基础**，所有计算都在向量空间中进行
2. **内积衡量相似度**，是注意力机制的核心
3. **范数衡量向量的大小**，影响梯度流
4. **正交性和投影**帮助理解多头注意力中的子空间
5. **距离度量**（欧几里得、余弦）决定了向量之间的关系

---

## 与后续章节的连接

- **4.1-4.4**：向量空间中的注意力计算
- **第5-8章（LLM）**：嵌入空间的学习和优化
- **扩展**：度量学习、对比学习
