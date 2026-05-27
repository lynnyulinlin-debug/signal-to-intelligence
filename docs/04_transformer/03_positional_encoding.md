# 4.3 位置编码

**核心问题：** Transformer如何理解序列顺序？

---

## 问题

自注意力机制没有位置信息。

```
"The cat sat on the mat"
"mat the on sat cat The"

对自注意力来说，这两个句子是一样的！
```

---

## 解决方案：位置编码

### 思想

给每个位置添加一个位置向量，编码位置信息。

### 数学表达

$$PE(pos, 2i) = \sin\left(\frac{pos}{10000^{2i/d}}\right)$$
$$PE(pos, 2i+1) = \cos\left(\frac{pos}{10000^{2i/d}}\right)$$

其中：
- $pos$：位置
- $i$：维度
- $d$：模型维度

### 直观理解

用不同频率的正弦和余弦波编码位置。

```
低频：变化慢，编码绝对位置
高频：变化快，编码相对位置
```

---

## 与傅里叶变换的联系

### 傅里叶变换

用不同频率的正弦波表示信号。

### 位置编码

用不同频率的正弦波表示位置。

**启示：** 位置编码就是把位置"傅里叶变换"。

---

## 实验结果

![Positional Encoding Visualization](../../assets/ch04_positional_encoding.png)

*图4.2：位置编码的结构。左上：热力图显示不同维度在不同位置的值（低频维度变化慢，高频维度变化快）。右上：不同维度的周期性曲线。左下：位置编码的范数（近似常数）。右下：相邻位置的相似度（编码了相对距离信息）。*

**代码实验：** 见 [`code/ch01_dsp/positional_encoding.py`](../../code/ch01_dsp/positional_encoding.py)

---

## 本节小结

位置编码用正弦波编码位置信息，让Transformer理解序列顺序。

---

**下一节：** [4.4 完整架构](04_architecture.md)
