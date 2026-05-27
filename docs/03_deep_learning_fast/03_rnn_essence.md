# 3.3 RNN的本质

**核心问题：** RNN如何处理序列？与状态空间模型有什么关系？

---

## 序列数据的挑战

### 问题

CNN假设输入是固定大小的，不适合处理序列。

**序列数据的特点：**
- 长度可变
- 有时间顺序
- 当前输出依赖于历史输入

**例子：**
- 文本：单词序列
- 音频：声波序列
- 时间序列：股票价格序列

---

## RNN的基本思想

### 循环结构

RNN在每个时刻都有一个隐状态，用来记忆历史信息。

```
h_t = f(h_{t-1}, x_t)
y_t = g(h_t)
```

其中：
- $h_t$：第t时刻的隐状态
- $x_t$：第t时刻的输入
- $y_t$：第t时刻的输出

### 直观理解

```
时刻1：h_1 = f(h_0, x_1)
时刻2：h_2 = f(h_1, x_2)  ← 依赖于h_1
时刻3：h_3 = f(h_2, x_3)  ← 依赖于h_2
...
```

隐状态像一个"记忆"，逐步更新。

---

## RNN与状态空间模型的联系

### 状态空间模型

**状态转移方程：**
$$\mathbf{x}_{t+1} = \mathbf{A} \mathbf{x}_t + \mathbf{B} \mathbf{u}_t$$

**观测方程：**
$$\mathbf{y}_t = \mathbf{C} \mathbf{x}_t + \mathbf{D} \mathbf{u}_t$$

### RNN

**隐状态更新：**
$$h_t = \tanh(\mathbf{W}_{hh} h_{t-1} + \mathbf{W}_{xh} x_t + b_h)$$

**输出：**
$$y_t = \mathbf{W}_{hy} h_t + b_y$$

### 联系

RNN就是可学习的状态空间模型：
- 状态转移矩阵 $\mathbf{A}$ → 可学习的 $\mathbf{W}_{hh}$
- 输入矩阵 $\mathbf{B}$ → 可学习的 $\mathbf{W}_{xh}$
- 观测矩阵 $\mathbf{C}$ → 可学习的 $\mathbf{W}_{hy}$

---

## RNN的问题

### 梯度消失（Vanishing Gradient）

反向传播时，梯度通过多个时刻相乘，容易变得很小。

$$\frac{\partial L}{\partial h_1} = \frac{\partial L}{\partial h_T} \prod_{t=2}^{T} \frac{\partial h_t}{\partial h_{t-1}}$$

如果 $\frac{\partial h_t}{\partial h_{t-1}} < 1$，乘积会指数衰减。

### 梯度爆炸（Exploding Gradient）

如果 $\frac{\partial h_t}{\partial h_{t-1}} > 1$，乘积会指数增长。

---

## LSTM和GRU

### LSTM（长短期记忆）

引入"门"机制，控制信息流。

**三个门：**
1. **遗忘门**：决定丢弃哪些信息
2. **输入门**：决定添加哪些信息
3. **输出门**：决定输出哪些信息

**优点：** 解决梯度消失问题，能学习长期依赖。

### GRU（门控循环单元）

LSTM的简化版本，参数更少。

---

## 本节小结

RNN处理序列数据：
- 隐状态记忆历史信息
- 与状态空间模型相关
- LSTM/GRU解决梯度问题

---

**下一节：** [3.4 为什么Transformer更好](04_why_transformer_better.md)
