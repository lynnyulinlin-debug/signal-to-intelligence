# 扩展：线性系统理论

**难度：** 高级  
**前置知识：** 线性代数、微积分、信号处理基础（第1章）

---

## 为什么需要理解线性系统

在信号处理和控制中：

- **系统稳定性**决定了滤波器是否可用
- **频率响应**决定了系统对不同频率的处理
- **状态空间表示**是现代控制理论的基础
- **RNN 和 LSTM** 可以看作非线性动态系统

---

## 线性时不变系统（LTI）

### 定义

**线性系统**：满足叠加原理
$$T(a x_1(t) + b x_2(t)) = a T(x_1(t)) + b T(x_2(t))$$

**时不变系统**：系统特性不随时间变化
$$y(t - \tau) = T(x(t - \tau))$$

### 系统的表示

**差分方程（离散时间）：**
$$y[n] = \sum_{k=0}^{M} b_k x[n-k] - \sum_{k=1}^{N} a_k y[n-k]$$

**微分方程（连续时间）：**
$$\sum_{k=0}^{N} a_k \frac{d^k y(t)}{dt^k} = \sum_{k=0}^{M} b_k \frac{d^k x(t)}{dt^k}$$

---

## 冲激响应和卷积

### 冲激响应

**定义**：系统对单位冲激的响应
$$h[n] = T(\delta[n])$$

其中 $\delta[n]$ 是 Kronecker delta 函数。

### 系统的输出

对于 LTI 系统，任意输入的输出可以表示为：
$$y[n] = \sum_{k=-\infty}^{\infty} x[k] h[n-k] = x[n] * h[n]$$

**直观理解**：输出是输入与冲激响应的卷积。

### 因果性

**因果系统**：输出只依赖于当前和过去的输入
$$h[n] = 0, \quad n < 0$$

---

## 频率响应

### 频率响应函数

**定义**：系统对复指数输入 $e^{j\omega n}$ 的响应
$$H(e^{j\omega}) = \sum_{n=-\infty}^{\infty} h[n] e^{-j\omega n}$$

**性质：**
- 幅度响应：$|H(e^{j\omega})|$
- 相位响应：$\angle H(e^{j\omega})$

### 频率响应与冲激响应的关系

$$H(e^{j\omega}) = \text{DTFT}(h[n])$$

**应用**：通过频率响应设计滤波器。

---

## 系统稳定性

### BIBO 稳定性

**定义**：有界输入产生有界输出
$$|x[n]| < \infty \Rightarrow |y[n]| < \infty$$

### 稳定性条件

**充要条件**：冲激响应绝对可和
$$\sum_{n=-\infty}^{\infty} |h[n]| < \infty$$

### 极点和零点

**传递函数：**
$$H(z) = \frac{B(z)}{A(z)} = \frac{\sum_{k=0}^{M} b_k z^{-k}}{1 + \sum_{k=1}^{N} a_k z^{-k}}$$

**稳定性条件**：所有极点在单位圆内
$$|p_i| < 1$$

---

## 状态空间表示

### 状态空间模型

**离散时间：**
$$\mathbf{x}[n+1] = \mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n]$$
$$\mathbf{y}[n] = \mathbf{C}\mathbf{x}[n] + \mathbf{D}\mathbf{u}[n]$$

**连续时间：**
$$\dot{\mathbf{x}}(t) = \mathbf{A}\mathbf{x}(t) + \mathbf{B}\mathbf{u}(t)$$
$$\mathbf{y}(t) = \mathbf{C}\mathbf{x}(t) + \mathbf{D}\mathbf{u}(t)$$

其中：
- $\mathbf{x}$：状态向量
- $\mathbf{u}$：输入向量
- $\mathbf{y}$：输出向量
- $\mathbf{A}, \mathbf{B}, \mathbf{C}, \mathbf{D}$：系统矩阵

### 从差分方程到状态空间

**例子**：$y[n] = 0.5 y[n-1] + x[n]$

**状态空间表示：**
$$\mathbf{x}[n+1] = \begin{bmatrix} 0.5 \end{bmatrix} \mathbf{x}[n] + \begin{bmatrix} 1 \end{bmatrix} u[n]$$
$$y[n] = \begin{bmatrix} 1 \end{bmatrix} \mathbf{x}[n]$$

### 传递函数与状态空间的关系

$$H(z) = \mathbf{C}(z\mathbf{I} - \mathbf{A})^{-1}\mathbf{B} + \mathbf{D}$$

---

## 系统的可控性和可观测性

### 可控性

**定义**：能否通过输入将系统从任意初始状态转移到任意目标状态。

**可控性矩阵：**
$$\mathcal{C} = [\mathbf{B}, \mathbf{A}\mathbf{B}, \mathbf{A}^2\mathbf{B}, \ldots, \mathbf{A}^{n-1}\mathbf{B}]$$

**条件**：$\text{rank}(\mathcal{C}) = n$（状态维数）

### 可观测性

**定义**：能否从输出推断出系统的初始状态。

**可观测性矩阵：**
$$\mathcal{O} = \begin{bmatrix} \mathbf{C} \\ \mathbf{C}\mathbf{A} \\ \mathbf{C}\mathbf{A}^2 \\ \vdots \\ \mathbf{C}\mathbf{A}^{n-1} \end{bmatrix}$$

**条件**：$\text{rank}(\mathcal{O}) = n$

---

## 系统的特征值和稳定性

### 特征值与稳定性

**离散时间系统稳定**当且仅当：
$$|\lambda_i| < 1, \quad \forall i$$

其中 $\lambda_i$ 是 $\mathbf{A}$ 的特征值。

**连续时间系统稳定**当且仅当：
$$\text{Re}(\lambda_i) < 0, \quad \forall i$$

### 特征值与系统响应

**特征值决定了系统的动态特性：**
- 实特征值：指数衰减或增长
- 复特征值对：振荡衰减或增长

---

## 滤波器设计

### IIR 滤波器

**无限脉冲响应**：冲激响应无限长

**优势：**
- 用较少的系数实现陡峭的频率响应
- 计算量小

**劣势：**
- 可能不稳定
- 相位响应非线性

### FIR 滤波器

**有限脉冲响应**：冲激响应有限长

**优势：**
- 总是稳定的
- 可以设计线性相位

**劣势：**
- 需要更多系数
- 计算量大

### 滤波器设计方法

1. **模拟滤波器设计**（Butterworth、Chebyshev）
2. **双线性变换**：将模拟滤波器转换为数字滤波器
3. **窗函数法**：设计 FIR 滤波器
4. **最优设计**：Remez 算法

---

## 系统识别

### 参数估计

**问题**：给定输入输出数据，估计系统参数。

**最小二乘法：**
$$\hat{\boldsymbol{\theta}} = \arg\min_{\boldsymbol{\theta}} \sum_{n=1}^{N} (y[n] - \hat{y}[n|\boldsymbol{\theta}])^2$$

### 递归最小二乘（RLS）

**优势：**
- 在线学习
- 适应时变系统

**算法：**
$$\hat{\boldsymbol{\theta}}[n] = \hat{\boldsymbol{\theta}}[n-1] + \mathbf{K}[n] (y[n] - \hat{y}[n])$$

其中 $\mathbf{K}[n]$ 是 Kalman 增益。

---

## 与深度学习的连接

### RNN 作为动态系统

**RNN 的隐状态更新：**
$$\mathbf{h}[n] = \sigma(\mathbf{W}_h \mathbf{h}[n-1] + \mathbf{W}_x \mathbf{x}[n])$$

**类似于非线性状态空间模型：**
$$\mathbf{x}[n+1] = f(\mathbf{A}\mathbf{x}[n] + \mathbf{B}\mathbf{u}[n])$$

### LSTM 和门控机制

**遗忘门、输入门、输出门**可以看作系统的**可控性和可观测性**机制。

### 梯度消失问题

**根本原因**：RNN 的特征值小于 1，导致梯度指数衰减。

**解决方案**：
- 残差连接（类似于系统的稳定性）
- 门控机制（类似于系统的可控性）

---

## 实践建议

### 系统分析步骤

1. **获取系统模型**：差分方程或状态空间
2. **计算冲激响应**：理解系统的时域特性
3. **计算频率响应**：理解系统的频域特性
4. **检查稳定性**：极点位置或特征值
5. **设计控制器**（如需要）

### Python 实现

```python
import numpy as np
from scipy import signal

# 定义系统：H(z) = 1 / (1 - 0.5*z^-1)
b = [1]
a = [1, -0.5]

# 计算冲激响应
h = signal.impulse((b, a, 1))[1]

# 计算频率响应
w, H = signal.freqz(b, a)

# 检查稳定性
poles = np.roots(a)
print(f"极点：{poles}")
print(f"稳定：{np.all(np.abs(poles) < 1)}")
```

---

## 关键要点

1. **LTI 系统可以用冲激响应或频率响应完全描述**
2. **稳定性由极点位置决定**
3. **状态空间表示是现代控制理论的基础**
4. **可控性和可观测性决定了系统的可设计性**
5. **RNN 可以看作非线性动态系统**

---

## 进一步阅读

- Oppenheim & Schafer, "Discrete-Time Signal Processing"
- Kailath, "Linear Systems"
- Ljung, "System Identification: Theory for the User"
