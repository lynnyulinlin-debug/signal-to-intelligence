# 扩展：多模态应用进阶

**所属章节：** [第7章：多模态LLM](../README.md)
**前置阅读：** [7.4 多模态应用](../04_applications.md)、[7.5 实战案例](../05_case_studies.md)

---

## 概览

7.4 和 7.5 介绍了图像描述、VQA、图像搜索和三个实战案例的基本原理。本节补充：多模态 RAG 的架构设计、文档理解的完整技术栈（OCR+LLM 混合方案）、视觉 Agent 的工具调用模式，以及选择模型的决策框架。

---

## 多模态 RAG

### 为什么需要多模态 RAG

纯文本 RAG 的限制：

```
用户问：这份合同第5页的表格中，金额是多少？
纯文本 RAG：
  1. 用文本 embedding 检索 → 找到第5页的文字提取结果
  2. 问题：PDF 文字提取常常丢失表格结构
  3. 结果：答案不准确或找不到

多模态 RAG：
  1. 将 PDF 页面截图存入图像向量库
  2. 用 CLIP 编码查询文本，检索相关页面图像
  3. 将检索到的图像 + 问题送入多模态 LLM
  4. 结果：LLM 直接"看"表格，准确提取
```

### 架构

```
离线索引阶段：
  文档页面 → 截图 → CLIP 编码 → 图像向量库（Faiss/Weaviate）
                  → OCR 提取文字 → 文本向量库（可选双路检索）

在线查询阶段：
  用户问题 → CLIP 文本编码 → 向量检索 → Top-K 图像
                                              ↓
  用户问题 + Top-K 图像 → 多模态 LLM → 最终答案
```

### 双路检索策略

| 检索路径 | 优势 | 劣势 | 适合场景 |
|---------|------|------|---------|
| 图像向量（CLIP） | 捕获视觉布局 | 对纯文字内容不如文本检索 | 表格、图表、图文混排 |
| 文本向量（文字提取后） | 精确文字匹配 | 依赖 OCR 质量 | 纯文字段落 |
| 双路融合（RRF 重排序） | 互补优势 | 实现复杂 | 生产系统常见选择 |

**RRF（Reciprocal Rank Fusion）：** 将两路检索的排名倒数相加重排序，公式：

$$\text{score}(d) = \sum_{r \in \text{rankers}} \frac{1}{k + r(d)}$$

k=60 是经验值，防止 top-1 结果权重过大。

---

## 文档理解：OCR+LLM 混合方案

### 何时用纯多模态 LLM，何时需要 OCR

```
纯多模态 LLM 足够的场景：
  ✅ 清晰扫描件（300dpi+）
  ✅ 标准印刷字体
  ✅ 简单表格（行列对齐）
  ✅ 字段数量少（< 20 个）

需要 OCR 预处理的场景：
  ⚠️ 手写体或特殊字体
  ⚠️ 低分辨率扫描（<150dpi）
  ⚠️ 复杂嵌套表格
  ⚠️ 需要精确到字符的提取（合同条款）
  ⚠️ 大批量处理（OCR 比 LLM 便宜 10-100×）
```

### 混合方案架构

```
文档图像
    ↓
预处理（倾斜校正、增强对比度）
    ↓
OCR 引擎（PaddleOCR / Tesseract / Azure OCR）
    ↓
结构化文字 + 边界框坐标
    ↓
提示构造：
  "以下是文档中识别到的文字及其位置：
   [坐标(10,20)]: '发票号码'
   [坐标(150,20)]: 'INV-2024-001'
   ...
   请提取：发票号码、日期、总金额"
    ↓
LLM 理解结构关系（不需要识别文字，只需推理）
    ↓
结构化 JSON 输出
```

**优势：** LLM 只需做空间关系推理，文字识别由专用 OCR 完成；处理速度快 5-10×；成本低。

### 工具选型

| 工具 | 优势 | 适用场景 |
|------|------|---------|
| PaddleOCR | 开源，中英文强 | 中文文档，本地部署 |
| Azure Document Intelligence | 表格结构识别强 | 生产级文档处理 |
| Tesseract | 完全开源 | 简单场景，无预算 |
| Surya | 现代架构，多语言 | 复杂版式文档 |

---

## 视觉 Agent

### 多模态 LLM 作为 Agent 的感知模块

第6章介绍了文本 Agent（ReAct 模式）。多模态 LLM 可以扩展 Agent 的感知能力：

```
传统文本 Agent：
  工具调用 → 文字结果 → LLM 推理 → 下一步

视觉 Agent：
  工具调用 → 截图/图像结果 → 多模态 LLM 感知 → LLM 推理 → 下一步
```

### 实际应用：网页操作 Agent

```
任务："在电商网站上找到最便宜的红色耳机"

Step 1:
  Action: screenshot()
  Observe: [网页截图]
  Reason: "看到搜索框在右上角，输入关键词"

Step 2:
  Action: click(x=850, y=45)  # 搜索框坐标
  Action: type("红色耳机")

Step 3:
  Action: screenshot()
  Observe: [搜索结果页截图]
  Reason: "看到4个结果，需要按价格排序"

Step 4:
  Action: click_button("价格从低到高")
  Observe: [排序后截图]
  Reason: "第一个结果 ¥89，点击查看详情"
```

### 关键挑战

**坐标幻觉：** 模型看到截图后给出的点击坐标可能不准确，因为分辨率缩放后像素位置改变。

解决方案：
```python
# 传入缩放比例让模型做坐标换算
prompt = f"""
截图分辨率已从 {original_w}×{original_h} 缩放到 {scaled_w}×{scaled_h}
如需点击，请给出原始分辨率下的坐标
"""
```

**多步累积误差：** 每一步的视觉理解误差会在多步中累积。缓解方法：在每步操作后截图确认，而不是一次性规划所有步骤。

---

## 模型选型决策框架

下面是选型维度示例，不是长期固定推荐。多模态模型迭代很快，生产选型需要重新确认当前模型能力、价格、许可证和部署约束。

### 按任务选择

```
任务需求
    │
    ├─ 主要是文字提取（发票/合同）？
    │       → 高分辨率多模态模型 + OCR 混合方案
    │
    ├─ 需要中文理解？
    │       → 优先评估中文多模态模型
    │
    ├─ 图文检索/相似度搜索？
    │       → CLIP / OpenCLIP（轻量，不需要 LLM）
    │
    ├─ 开源本地部署，通用场景？
    │       → LLaVA-NeXT / InternVL2
    │
    ├─ 需要高能力 API 调用？
    │       → 评估当前主流闭源多模态 API
    │
    └─ 端侧设备，内存 < 8GB？
            → MiniCPM-o / MobileVLM
```

### 按资源选择

| 显存 | 示例候选 | 分辨率上限 | 推理速度 |
|------|---------|-----------|---------|
| 4GB | MiniCPM-o-3B | 448×448 | 快 |
| 8GB | LLaVA-1.5-7B | 336×336 | 中 |
| 12GB | Qwen2.5-VL-7B | 1024×1024 | 中 |
| 24GB | Qwen2.5-VL-32B / InternVL2-26B | 动态 | 慢 |
| API | 主流闭源多模态 API | 取决于服务商限制 | 取决于网络 |

---

**返回：** [第7章：多模态LLM](../README.md)

---

## 推荐论文

### 基础论文

| 论文 | 作者 | 贡献 |
|------|------|------|
| **"Learning Transferable Visual Models From Natural Language Supervision"** | Radford et al. (2021) | CLIP：对比学习实现零样本图文对齐 |
| **"An Image is Worth 16x16 Words"** | Dosovitskiy et al. (2020) | ViT：用 Transformer 处理图像 |
| **"Visual Instruction Tuning"** | Liu et al. (2023) | LLaVA：多模态指令微调的开创性工作 |
| **"Blip-2: Bootstrapping Language-Image Pre-training"** | Li et al. (2023) | Q-Former：参数高效的多模态预训练 |
| **"Flamingo: a Visual Language Model for Few-Shot Learning"** | Alayrac et al. (2022) | Gated cross-attention，few-shot 多模态 |

### 进阶论文

| 论文 | 贡献 |
|------|------|
| InstructBLIP (2023) | 指令感知的视觉特征提取 |
| LLaVA-1.5 (2023) | MLP 投影层替代线性层，显著提升性能 |
| Qwen-VL (2023) | 中文多模态，位置感知输入 |

---

## 常见坑

### 坑1：CLIP 文本模板影响零样本分类准确率

| 模板 | 准确率 | 原因 |
|------|--------|------|
| `"{class}"` | ~60% | 单词无上下文，编码器特征弱 |
| `"a photo of a {class}"` | ~85% | 描述性上下文，特征更强 |
| 多模板投票 | ~92% | 多角度描述，降低单一模板的偏差 |

```python
# 最佳实践：多模板投票
templates = [
    "a photo of a {}",
    "a picture of a {}",
    "an image of a {}",
]
text_features = [encode(t.format(class_name)) for t in templates]
final_feature = torch.stack(text_features).mean(0)
final_feature /= final_feature.norm()
```

### 坑2：Qwen2.5-VL 需要特定 transformers 版本

```bash
# 需要 4.40.0+
pip install "transformers>=4.40.0"
python -c "import transformers; print(transformers.__version__)"
```

### 坑3：高分辨率导致显存溢出

| 场景 | 推荐方案 |
|------|---------|
| 显存 < 16GB，一般任务 | 动态分辨率（自动限制上限） |
| 显存 < 16GB，文档任务 | 图像分块 + 量化（int8） |
| 显存充足 | 图像分块，保留最多细节 |

### 坑4：多图像处理时图像顺序混淆

```python
# ❌ 容易混淆
prompt = "比较这两张图片的区别"

# ✅ 明确指代
prompt = "[图1] 是原始状态，[图2] 是处理后状态。请比较 [图1] 和 [图2] 的变化。"
```
