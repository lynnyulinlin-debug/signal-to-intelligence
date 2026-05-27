# 4.1 自注意力机制

**核心问题：** 什么是自注意力？它如何工作？

---

## 注意力的直观理解

### 类比：阅读

读一句话时，你会关注不同单词的重要性。

```
"The cat sat on the mat"

当读"sat"时，你特别关注"cat"和"mat"
```

### 数学表达

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

其中：
- $Q$：查询（Query）— 当前位置想问什么
- $K$：键（Key）— 其他位置的特征
- $V$：值（Value）— 其他位置的内容

---

## 自注意力的步骤

### 1. 计算相似度

$$\text{scores} = QK^T$$

**含义：** 当前位置与其他位置的相似度。

### 2. 归一化

$$\text{weights} = \text{softmax}\left(\frac{\text{scores}}{\sqrt{d_k}}\right)$$

**含义：** 把相似度转换为权重（和为1）。

### 3. 加权求和

$$\text{output} = \text{weights} \cdot V$$

**含义：** 用权重加权其他位置的内容。

---

## 自注意力的优势

### 1. 完全可并行化

每个位置可以同时计算，不需要顺序处理。

### 2. 长期依赖容易

任意两个位置之间的距离都是1（不是序列长度）。

### 3. 可解释性强

注意力权重显示模型关注哪些位置。

---

## 实验结果

![Self-Attention Visualization](../../assets/ch04_self_attention.png)

*图4.1：自注意力权重可视化。左上：所有头的平均注意力权重热力图。其他三个子图：不同注意力头学到的不同模式。右下：注意力权重分布，显示不同头的权重差异。*

**代码实验：** 见 [`code/ch04_transformer/self_attention.py`](../../code/ch04_transformer/self_attention.py)

---

## 本节小结

自注意力机制：
- 计算位置之间的相似度
- 用相似度作为权重
- 完全可并行化

---

**下一节：** [4.2 多头注意力](02_multihead.md)
