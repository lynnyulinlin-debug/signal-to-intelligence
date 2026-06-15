# 2.4 数值方法基础

**核心问题：** 为什么算法在理论上正确，实践中却仍然会失败？如何保证数值计算稳定？

---

## 为什么需要理解数值方法

理论上正确的算法，放到真实实现里也可能失效。原因通常不是模型本身，而是**浮点精度、条件数和实现方式**一起把误差放大了。

### 实际例子
在训练神经网络时，常见现象包括：

- 损失函数前几步正常，但很快变成 `NaN`
- 梯度在深层网络里逐层变小，训练几乎不动
- 同一个公式在不同实现里，因为浮点误差得到不同结果
---

## 浮点数精度

### 浮点数的表示

计算机用**有限位数**表示实数：

$$x = (-1)^s \times m \times 2^e$$

其中：
- $s$：符号位（0或1）
- $m$：尾数（mantissa），通常是 [1, 2) 之间的数
- $e$：指数（exponent）

### 精度限制

**单精度（float32）：** 约7位十进制精度
**双精度（float64）：** 约15位十进制精度

在深度学习里，float32 常用于训练和推理的主流程，float16 / bfloat16 常用于混合精度加速，但都需要关注溢出、下溢和舍入误差。

### 舍入误差

每次浮点运算都会产生舍入误差：

$$\text{computed} = \text{true} \times (1 + \epsilon)$$

其中 $|\epsilon| \leq \text{machine epsilon}$（机器精度）。

---

## 数值稳定性

### 什么是数值稳定的算法

如果小的输入误差不会被放大成大的输出误差，这个算法就可以认为是数值稳定的。

### 条件数（Condition Number）

对于线性系统 $\mathbf{A}\mathbf{x} = \mathbf{b}$，条件数定义为：

$$\kappa(\mathbf{A}) = \|\mathbf{A}\| \cdot \|\mathbf{A}^{-1}\|$$

**解释：**
- $\kappa$ 小（接近1）：问题是**良态的**（well-conditioned）
- $\kappa$ 大：问题是**病态的**（ill-conditioned）

**影响：** 如果 $\kappa$ 很大，输入的小误差会被放大 $\kappa$ 倍。

### 例子：病态矩阵

在实践中，病态矩阵会导致数值计算不稳定。例如：

```
A = [[1.0, 1.0],
     [1.0, 1.0 + 1e-10]]
```

这个矩阵的条件数很大（约2e10），求解线性系统时小的误差会被放大。

---

## 梯度的数值计算

### 有限差分法

当无法直接计算梯度时，可以用**有限差分**近似：

**前向差分：**
$$\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x)}{h}$$

**中心差分（更精确）：**
$$\frac{\partial f}{\partial x} \approx \frac{f(x + h) - f(x - h)}{2h}$$

### 步长的选择

步长 $h$ 的选择很关键：

- $h$ 太大：截断误差大（近似不准确）
- $h$ 太小：舍入误差大（浮点精度问题）

**经验步长：** 对前向差分，常用 $h \approx \sqrt{\epsilon}$ 作为起点，其中 $\epsilon$ 是机器精度；实际还要结合变量尺度调整。

### 梯度检查（Gradient Checking）

在实现反向传播时，应该用有限差分验证梯度的正确性：

**验证方法：**
- 计算数值梯度：$\frac{f(x + \epsilon) - f(x - \epsilon)}{2\epsilon}$
- 计算解析梯度：通过反向传播
- 检查两者是否接近（相对误差 < 1e-5）

梯度检查不是替代反向传播，而是用来定位实现错误。
如果数值梯度和解析梯度差很多，优先检查代码实现，而不是先怀疑理论本身。

### 链式法则与反向传播

#### 链式法则的详细应用

对于复合函数 $L = f(g(h(x)))$，链式法则告诉我们如何计算梯度：

$$\frac{\partial L}{\partial x} = \frac{\partial L}{\partial f} \cdot \frac{\partial f}{\partial g} \cdot \frac{\partial g}{\partial h} \cdot \frac{\partial h}{\partial x}$$

**例子：** 神经网络的前向传播和反向传播

```
前向传播：x → h₁ → h₂ → ... → hₙ → L
反向传播：∂L/∂x = ∂L/∂hₙ · ∂hₙ/∂hₙ₋₁ · ... · ∂h₁/∂x
```

#### 反向传播的数学推导

反向传播就是利用链式法则，从输出层开始逐层计算梯度。

**关键思想：** 复用中间结果，避免重复计算

**时间复杂度对比：**
- 直接计算每个参数的梯度：$O(n^2)$（n是参数数量）
- 反向传播：一次前向传播 + 一次反向传播即可得到所有参数梯度，成本通常和前向计算同量级

**反向传播的步骤：**
1. 前向传播：计算所有中间值和最终输出
2. 计算输出层的梯度：$\frac{\partial L}{\partial y}$
3. 逐层反向传播：$\frac{\partial L}{\partial w_i} = \frac{\partial L}{\partial h_i} \cdot \frac{\partial h_i}{\partial w_i}$
4. 更新参数：$w_i \leftarrow w_i - \alpha \frac{\partial L}{\partial w_i}$

#### 为什么反向传播高效

**问题：** 如果用数值差分逐个参数检查梯度，需要对每个参数额外做前向计算

**解决方案：** 反向传播只需一次前向传播和一次反向传播，就能计算所有参数的梯度

**效率提升：** 从“按参数逐个估计梯度”变成“一次反向传播得到整组梯度”

---

## 矩阵分解的数值方法

### 为什么不直接求逆

**不要这样做：**
$$\mathbf{x} = \mathbf{A}^{-1} \mathbf{b}$$

**原因：**
1. 计算 $\mathbf{A}^{-1}$ 数值不稳定
2. 容易放大舍入误差
3. 计算量大（$O(n^3)$）

**应该这样做：**
$$\mathbf{A}\mathbf{x} = \mathbf{b}$$

用 LU 分解或 QR 分解求解。

### LU 分解

$$\mathbf{A} = \mathbf{L}\mathbf{U}$$

其中 $\mathbf{L}$ 是下三角矩阵，$\mathbf{U}$ 是上三角矩阵。

**优势：**
- 数值稳定（使用部分主元）
- 计算量小（$O(n^3)$，但常数小）
- 可以快速求解多个右端项

### QR 分解

$$\mathbf{A} = \mathbf{Q}\mathbf{R}$$

其中 $\mathbf{Q}$ 是正交矩阵，$\mathbf{R}$ 是上三角矩阵。

**优势：**
- 数值最稳定
- 适合最小二乘问题
- 条件数改善

---

## 优化算法中的数值问题

### 1. 梯度消失（Vanishing Gradient）

在深层网络中，梯度可能变得非常小：

$$\frac{\partial L}{\partial w_1} = \frac{\partial L}{\partial w_n} \times \frac{\partial w_n}{\partial w_{n-1}} \times \cdots \times \frac{\partial w_2}{\partial w_1}$$

如果每个偏导数都 < 1，乘积会指数衰减。

**解决方案：**
- 使用 ReLU 激活函数（导数为0或1）
- 批量归一化（Batch Normalization）
- 残差连接（Residual Connections）

### 2. 梯度爆炸（Exploding Gradient）

相反的问题：梯度变得非常大。

**解决方案：**
- 梯度裁剪（Gradient Clipping）
- 权重初始化（Xavier/He initialization）

### 3. 学习率的选择

学习率 $\alpha$ 太大会导致发散，太小会导致收敛慢。

**自适应学习率的优势：**
- Adam、RMSprop 等自动调整学习率
- 对不同参数使用不同的学习率

---

## 实践建议

### 检查清单

在实现优化算法时：

- [ ] 检查梯度（用有限差分验证）
- [ ] 监控损失函数（应该单调下降）
- [ ] 检查权重的范数（不应该爆炸或消失）
- [ ] 使用合理的初始化（Xavier/He）
- [ ] 考虑使用自适应优化器（Adam）
- [ ] 监控学习率（可视化训练曲线）

### 常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 损失函数 NaN | 梯度爆炸或学习率太大 | 降低学习率，使用梯度裁剪 |
| 损失函数不下降 | 学习率太小或梯度计算错误 | 增加学习率，检查梯度 |
| 权重不更新 | 梯度消失 | 使用 ReLU，批量归一化 |
| 训练不稳定 | 数据未归一化或学习率不稳定 | 归一化数据，使用学习率调度 |

---

## 与后续章节的连接

- **第3章（深度学习）**：反向传播、残差连接和归一化都依赖稳定的数值实现
- **第4-8章（Transformer / LLM）**：注意力分数缩放、LayerNorm、混合精度和学习率调度都遵循同样的数值原则

---

## 代码实验

虽然数值方法本身没有专门的代码实验，但以下实验展示了数值稳定性的重要性：

**相关实验：**
- [`code/ch02_optimization/lms_vs_adam.py`](../../code/ch02_optimization/lms_vs_adam.py) - 展示不同优化器的数值稳定性
- [`code/ch02_optimization/linear_logistic_regression.py`](../../code/ch02_optimization/linear_logistic_regression.py) - 展示梯度计算的数值问题

**运行方式：**
```bash
python code/ch02_optimization/lms_vs_adam.py
python code/ch02_optimization/linear_logistic_regression.py
```

**关键观察：**
- 不同优化器的数值稳定性差异
- 梯度计算中的舍入误差
- 学习率对数值稳定性的影响

---

## 与深度学习的联系

### 数值稳定性 → 深度学习训练

```
浮点数精度问题
    ↓ (在深层网络中放大)
梯度消失/爆炸
    ↓ (导致训练失败)
需要数值稳定的算法
    ↓ (使用批量归一化、残差连接等)
稳定的深度学习训练
```

### 深度学习中的数值问题

1. **梯度消失（Vanishing Gradient）**
   - 在深层网络中，梯度通过链式法则逐层相乘
   - 如果每层的梯度 < 1，最终梯度会指数衰减
   - 导致深层参数无法更新

2. **梯度爆炸（Exploding Gradient）**
   - 相反的问题：梯度变得非常大
   - 导致参数更新过大，训练不稳定
   - 可能导致损失函数变为 NaN

3. **解决方案**
   - **批量归一化（Batch Normalization）** - 保持激活值在合理范围
   - **残差连接（Residual Connections）** - 允许梯度直接流动
   - **梯度裁剪（Gradient Clipping）** - 限制梯度的大小
   - **权重初始化（Xavier/He Initialization）** - 合理初始化权重

### 为什么LLM需要数值稳定性

- **参数众多**：LLM有数十亿个参数，数值误差会累积
- **训练时间长**：训练过程中舍入误差会不断累积
- **精度要求高**：需要使用混合精度训练（float16 + float32）来平衡速度和精度

---

## 在LLM中的应用

### 数值稳定性在LLM训练中的关键作用

LLM的成功训练需要数值稳定性作为基础保障：

1. **梯度消失/爆炸在LLM中的影响**
   - 大型模型通常有数十到上百层 Transformer 块
   - 梯度需要通过所有层反向传播
   - 数值不稳定会导致训练失败

2. **LLM中的数值稳定性技术**
   - **层归一化（Layer Normalization）** - 每层的输入归一化
   - **残差连接（Residual Connections）** - 梯度直接流动
   - **梯度裁剪（Gradient Clipping）** - 限制梯度范数
   - **混合精度训练** - float16计算，float32存储

### 混合精度训练在LLM中的应用

混合精度训练是LLM训练的标准做法：

1. **为什么需要混合精度？**
   - float32训练太慢：LLM训练需要数周甚至数月
   - float16内存占用少：可以训练更大的模型
   - 精度损失可接受：通过float32梯度存储补偿

2. **混合精度的实现**
   ```
   前向传播：float16计算（快速）
   损失计算：float32（精度）
   反向传播：float16计算（快速）
   梯度存储：float32（精度）
   参数更新：float32（精度）
   ```

3. **性能提升**
   - 速度：快2-3倍
   - 内存：减少50%
   - 精度：基本不变

### 数值问题的实际案例

1. **梯度爆炸导致的NaN**
   - 训练过程中损失突然变为NaN
   - 通常是梯度爆炸导致
   - 解决方案：梯度裁剪

2. **梯度消失导致的收敛缓慢**
   - 深层参数无法更新
   - 导致模型性能不佳
   - 解决方案：残差连接、层归一化

3. **舍入误差的累积**
   - 长期训练中舍入误差累积
   - 可能导致模型性能下降
   - 解决方案：定期验证、使用float32存储

### LLM训练中的数值稳定性最佳实践

1. **梯度裁剪**
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

2. **混合精度训练**
   ```python
   from torch.cuda.amp import autocast, GradScaler
   scaler = GradScaler()
   with autocast():
       loss = model(input_ids)
   scaler.scale(loss).backward()
   ```

3. **监控数值稳定性**
   - 监控梯度范数
   - 监控激活值范围
   - 监控损失值变化

---

**下一步：** 阅读 [2.5 线性回归与逻辑回归](05_linear_logistic_regression.md)
