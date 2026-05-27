# 1.7 信号估计

**核心问题：** 如何从噪声中估计信号的参数？如何评估估计器的性能？

---

## 参数估计问题

### 问题设置

**观测模型：**
$$y[n] = s(x[n]; \theta) + w[n], \quad n = 0, 1, \ldots, N-1$$

其中：
- $y[n]$：观测信号
- $s(x[n]; \theta)$：参数化信号模型
- $\theta$：待估计的参数
- $w[n]$：加性高斯白噪声

**估计任务：** 给定观测 $\mathbf{y} = [y[0], y[1], \ldots, y[N-1]]^T$，估计参数 $\theta$。

### 参数估计的例子

**例1：正弦波参数估计**
$$s(t; \theta) = A \sin(2\pi f t + \phi)$$

参数：$\theta = [A, f, \phi]^T$（幅度、频率、相位）

**例2：信道估计**
$$y[n] = h \cdot x[n] + w[n]$$

参数：$\theta = h$（信道增益）

**例3：时延估计**
$$y[n] = s[n - \tau] + w[n]$$

参数：$\theta = \tau$（时延）

---

## 最大似然估计（Maximum Likelihood Estimation, MLE）

### 原理

**似然函数：** 给定参数 $\theta$，观测 $\mathbf{y}$ 的概率。

$$L(\theta; \mathbf{y}) = p(\mathbf{y}|\theta)$$

**最大似然估计：** 选择使似然函数最大的参数。

$$\hat{\theta}_{\text{ML}} = \arg\max_{\theta} L(\theta; \mathbf{y})$$

**对数似然函数：** 为了计算方便，通常最大化对数似然。

$$\hat{\theta}_{\text{ML}} = \arg\max_{\theta} \ln L(\theta; \mathbf{y})$$

### 高斯噪声下的MLE

**假设：** 噪声是AWGN，方差为 $\sigma^2$。

**似然函数：**
$$L(\theta; \mathbf{y}) = \prod_{n=0}^{N-1} \frac{1}{\sqrt{2\pi\sigma^2}} \exp\left(-\frac{(y[n] - s(x[n];\theta))^2}{2\sigma^2}\right)$$

**对数似然函数：**
$$\ln L(\theta; \mathbf{y}) = -\frac{1}{2\sigma^2} \sum_{n=0}^{N-1} (y[n] - s(x[n];\theta))^2 + \text{常数}$$

**最大化对数似然等价于最小化误差平方和：**
$$\hat{\theta}_{\text{ML}} = \arg\min_{\theta} \sum_{n=0}^{N-1} (y[n] - s(x[n];\theta))^2$$

### MLE的性质

**一致性：** 当 $N \to \infty$ 时，$\hat{\theta}_{\text{ML}} \to \theta_0$（真实参数）

**渐近正态性：** 当 $N$ 很大时，$\hat{\theta}_{\text{ML}}$ 近似高斯分布

**渐近有效性：** MLE达到Cramér-Rao界（最优性能）

---

## 最小二乘估计（Least Squares Estimation, LSE）

### 原理

**最小二乘准则：** 最小化观测与模型的误差平方和。

$$\hat{\theta}_{\text{LS}} = \arg\min_{\theta} \sum_{n=0}^{N-1} (y[n] - s(x[n];\theta))^2$$

**与MLE的关系：** 在高斯噪声假设下，LSE等价于MLE。

### 线性最小二乘

**线性模型：**
$$y[n] = \mathbf{h}^T \mathbf{x}[n] + w[n]$$

其中 $\mathbf{h}$ 是参数向量，$\mathbf{x}[n]$ 是观测向量。

**矩阵形式：**
$$\mathbf{y} = \mathbf{X} \mathbf{h} + \mathbf{w}$$

其中 $\mathbf{X}$ 是设计矩阵。

**闭式解：**
$$\hat{\mathbf{h}}_{\text{LS}} = (\mathbf{X}^T \mathbf{X})^{-1} \mathbf{X}^T \mathbf{y}$$

**优点：** 有闭式解，计算快速。

### 非线性最小二乘

**非线性模型：** $s(x[n];\theta)$ 对 $\theta$ 非线性。

**求解方法：**
- 梯度下降法
- 牛顿法
- Levenberg-Marquardt算法

**优点：** 适用于复杂模型。

**缺点：** 需要迭代，可能陷入局部最优。

---

## 贝叶斯估计

### 原理

**贝叶斯定理：**
$$p(\theta|\mathbf{y}) = \frac{p(\mathbf{y}|\theta) p(\theta)}{p(\mathbf{y})}$$

其中：
- $p(\theta|\mathbf{y})$：后验分布
- $p(\mathbf{y}|\theta)$：似然函数
- $p(\theta)$：先验分布
- $p(\mathbf{y})$：证据

**贝叶斯估计：** 利用后验分布进行估计。

### 常见的贝叶斯估计器

**1. 最大后验估计（MAP）**
$$\hat{\theta}_{\text{MAP}} = \arg\max_{\theta} p(\theta|\mathbf{y})$$

**2. 最小均方误差估计（MMSE）**
$$\hat{\theta}_{\text{MMSE}} = E[\theta|\mathbf{y}] = \int \theta \cdot p(\theta|\mathbf{y}) d\theta$$

**3. 最小绝对误差估计（MAE）**
$$\hat{\theta}_{\text{MAE}} = \text{median}(\theta|\mathbf{y})$$

### 贝叶斯 vs 频率学派

**频率学派（MLE）：**
- 参数是固定但未知的常数
- 估计器的性能由重复实验的分布决定

**贝叶斯学派：**
- 参数是随机变量
- 利用先验知识改进估计
- 结果是后验分布，不仅是点估计

**实际应用：** 当有先验知识时，贝叶斯方法通常更好。

---

## 克拉美罗界（Cramér-Rao Bound, CRB）

### 定义

**CRB：** 任何无偏估计器的方差的下界。

$$\text{Var}(\hat{\theta}) \geq \text{CRB}(\theta) = \frac{1}{I(\theta)}$$

其中 $I(\theta)$ 是Fisher信息矩阵。

### Fisher信息矩阵

**定义：**
$$I(\theta) = E\left[\left(\frac{\partial \ln L(\theta; \mathbf{y})}{\partial \theta}\right)^2\right]$$

**性质：**
- $I(\theta) > 0$（对于充分的观测）
- $I(\theta)$ 越大，CRB越小（估计越精确）
- 观测数 $N$ 越多，$I(\theta)$ 越大

### 高斯噪声下的CRB

**对于高斯噪声，Fisher信息矩阵为：**
$$I(\theta) = \frac{1}{\sigma^2} \sum_{n=0}^{N-1} \left(\frac{\partial s(x[n];\theta)}{\partial \theta}\right)^2$$

**CRB：**
$$\text{CRB}(\theta) = \frac{\sigma^2}{\sum_{n=0}^{N-1} \left(\frac{\partial s(x[n];\theta)}{\partial \theta}\right)^2}$$

### 达到CRB的条件

**MLE在以下条件下达到CRB：**
1. 似然函数是指数族分布
2. 观测数 $N$ 足够大
3. 参数在内部（不在边界）

---

## 估计器性能指标

### 1. 偏差（Bias）

**定义：**
$$\text{Bias}(\hat{\theta}) = E[\hat{\theta}] - \theta$$

**无偏估计器：** $\text{Bias}(\hat{\theta}) = 0$

**有偏估计器：** $\text{Bias}(\hat{\theta}) \neq 0$

**例子：**
- MLE通常是渐近无偏的
- 某些贝叶斯估计器可能有偏

### 2. 方差（Variance）

**定义：**
$$\text{Var}(\hat{\theta}) = E[(\hat{\theta} - E[\hat{\theta}])^2]$$

**性质：**
- 方差越小，估计越稳定
- 方差受CRB限制

### 3. 均方误差（Mean Squared Error, MSE）

**定义：**
$$\text{MSE}(\hat{\theta}) = E[(\hat{\theta} - \theta)^2]$$

**分解：**
$$\text{MSE}(\hat{\theta}) = \text{Bias}^2(\hat{\theta}) + \text{Var}(\hat{\theta})$$

**权衡：** 有时接受小的偏差来换取更小的方差。

### 4. 一致性（Consistency）

**定义：** 当 $N \to \infty$ 时，$\hat{\theta} \xrightarrow{p} \theta$

**含义：** 观测越多，估计越接近真实值。

---

## 实际应用

### 1. 频率估计

**问题：** 从噪声中估计正弦波的频率。

$$y[n] = A \sin(2\pi f n / f_s + \phi) + w[n]$$

**方法：**
- FFT：快速但分辨率有限
- MLE：精确但计算复杂
- 子空间方法（MUSIC）：高分辨率

**应用：** 功率系统监测、音乐分析

### 2. 信道估计

**问题：** 估计通信信道的参数。

$$y[n] = h \cdot x[n] + w[n]$$

**方法：**
- 最小二乘：简单快速
- 贝叶斯：利用信道统计特性
- 自适应滤波：跟踪时变信道

**应用：** 无线通信、雷达

### 3. 时延估计

**问题：** 估计信号的传播时延。

$$y[n] = s[n - \tau] + w[n]$$

**方法：**
- 相关法：计算 $y[n]$ 与 $s[n]$ 的相关性
- MLE：精确估计
- 子空间方法：多信号时延估计

**应用：** 定位、测距、同步

### 4. 医学信号处理

**问题：** 从生物信号中估计参数。

**例子：**
- 心率估计：从ECG估计心跳频率
- 呼吸率估计：从呼吸信号估计呼吸频率
- 脑波频率估计：从EEG估计脑波频率

**方法：** 通常使用频率估计技术

---

## 本节小结

信号估计是信号处理的核心问题：

- **估计方法**：MLE、LSE、贝叶斯估计
- **性能界**：Cramér-Rao界
- **性能指标**：偏差、方差、MSE、一致性
- **实际应用**：频率估计、信道估计、时延估计、医学信号处理

选择合适的估计方法取决于问题的具体特性和可用的先验知识。

---

## 在LLM中的应用

### 参数估计在LLM中的应用

1. **信号参数估计 ≈ Token预测**
   - DSP中的参数估计：从观测数据估计信号参数
   - LLM中的Token预测：从上文估计下一个Token
   - 都是"参数估计"

2. **最大似然估计（MLE）**
   - DSP中的MLE：最大化似然函数
   - LLM中的训练：最大化正确Token的概率
   - 都是"最大似然"

### 最小二乘估计在LLM中的应用

1. **LSE的思想**
   - DSP中的LSE：最小化误差的平方和
   - LLM中的训练：最小化交叉熵损失
   - 都是"最小化误差"

2. **线性回归 → 神经网络**
   - DSP中的LSE：线性参数估计
   - LLM中的输出层：线性变换
   - 都是"线性映射"

### 贝叶斯估计在LLM中的应用

1. **先验知识**
   - DSP中的贝叶斯估计：利用先验分布
   - LLM中的预训练：利用大规模文本的先验知识
   - 都是"利用先验"

2. **后验分布**
   - DSP中的贝叶斯估计：计算后验分布
   - LLM中的Softmax：计算后验概率分布
   - 都是"概率分布"

### Cramér-Rao界在LLM中的应用

1. **性能界**
   - DSP中的Cramér-Rao界：估计器性能的下界
   - LLM中的理论极限：模型性能的上界
   - 都是"性能分析"

2. **最优性**
   - DSP中的Cramér-Rao界：判断估计器是否最优
   - LLM中的性能评估：判断模型是否最优
   - 都是"最优性判断"

### 频率估计在LLM中的应用

1. **频率估计**
   - DSP中的频率估计：从信号估计频率
   - LLM中的模式识别：从文本识别模式
   - 都是"特征识别"

2. **高分辨率估计**
   - DSP中的高分辨率频率估计：MUSIC、ESPRIT
   - LLM中的细粒度分析：多头注意力
   - 都是"高分辨率"

### 信道估计在LLM中的应用

1. **信道估计**
   - DSP中的信道估计：估计通信信道的特性
   - LLM中的上下文理解：理解文本的"信道"特性
   - 都是"环境估计"

2. **自适应**
   - DSP中的自适应信道估计：动态调整
   - LLM中的上下文适应：根据上文调整
   - 都是"自适应"

### 时延估计在LLM中的应用

1. **时延估计**
   - DSP中的时延估计：估计信号的延迟
   - LLM中的位置编码：编码Token的位置
   - 都是"时间信息"

2. **相对位置**
   - DSP中的相对时延：信号之间的延迟关系
   - LLM中的相对位置编码：Token之间的相对位置
   - 都是"相对关系"

### 估计误差分析在LLM中的应用

1. **偏差（Bias）**
   - DSP中的偏差：估计器的系统误差
   - LLM中的系统偏差：模型的系统性错误
   - 都是"系统误差"

2. **方差（Variance）**
   - DSP中的方差：估计器的随机波动
   - LLM中的不确定性：模型的预测不确定性
   - 都是"随机误差"

3. **均方误差（MSE）**
   - DSP中的MSE：$E[(x - \hat{x})^2]$
   - LLM中的损失函数：衡量预测误差
   - 都是"误差度量"

### 一致性在LLM中的应用

1. **估计的一致性**
   - DSP中的一致性：样本数增加时估计收敛到真值
   - LLM中的收敛性：训练过程中损失逐步降低
   - 都是"收敛性"

2. **渐近性质**
   - DSP中的渐近无偏性：大样本时无偏
   - LLM中的渐近性能：大数据时性能提升
   - 都是"渐近性质"

### 参数学习在LLM中的应用

1. **参数优化**
   - DSP中的参数估计：找到最优参数
   - LLM中的反向传播：优化模型参数
   - 都是"参数优化"

2. **梯度下降**
   - DSP中的梯度下降：优化估计器
   - LLM中的梯度下降：优化神经网络
   - 都是"梯度优化"

---

**下一节：** [1.8 矩阵分解应用](08_matrix_decomposition.md)
