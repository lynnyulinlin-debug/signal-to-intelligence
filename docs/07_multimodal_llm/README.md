# 第7章：多模态LLM

**版本：** v3.2  
**最后更新：** 2026-06-02

## 章节概览

本章介绍多模态 LLM，回答一个核心问题：**如何把视觉信息接入语言模型，并让它可靠地完成任务？**

本章主线是跨模态系统设计：**视觉编码 → 图文对齐 → 融合方式 → 高分辨率处理 → 任务约束 → 模型选型**。具体模型用于说明设计取舍，不作为固定推荐列表。

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 7.1 视觉-语言对齐 | [01_vision_language.md](01_vision_language.md) | ⭐⭐⭐ | 15分钟 |
| 7.2 Qwen2.5-VL 架构 | [02_qwen_vl.md](02_qwen_vl.md) | ⭐⭐⭐ | 15分钟 |
| 7.3 高分辨率图像处理 | [03_high_resolution.md](03_high_resolution.md) | ⭐⭐⭐ | 10分钟 |
| 7.4 多模态应用 | [04_applications.md](04_applications.md) | ⭐⭐ | 10分钟 |
| 7.5 实战案例 | [05_case_studies.md](05_case_studies.md) | ⭐⭐⭐ | 10分钟 |

## 小节目录

### 对齐与融合（7.1-7.2）

**7.1 视觉-语言对齐** — [📖 阅读](01_vision_language.md)
- ViT：图像分块 + Transformer 编码
- CLIP：对比学习实现图像-文本对齐
- 对齐的几何直观：相似对靠近，不相似对远离
- 对齐失败：细粒度文字、空间关系和长尾视觉概念

**7.2 Qwen2.5-VL 架构** — [📖 阅读](02_qwen_vl.md)
- LLaVA 架构（单编码器 + 投影层 + LLM）
- 以 Qwen2.5-VL 为例理解高分辨率、多图像、中文优化等设计
- 混合融合策略 vs 早期/晚期融合
- 模型案例背后的通用问题：视觉 token 如何进入语言模型

### 技术细节（7.3）

**7.3 高分辨率图像处理** — [📖 阅读](03_high_resolution.md)
- 为什么标准 224×224 不够用
- 三种方案：动态分辨率、图像分块、自适应采样
- 方案权衡：细节保留、token 成本、延迟和上下文占用

### 应用（7.4-7.5）

**7.4 多模态应用** — [📖 阅读](04_applications.md)
- 图像问答、文档理解、图表分析
- 模型选择框架
- 任务边界、输入质量和失败模式

**7.5 实战案例** — [📖 阅读](05_case_studies.md)
- 文档理解、图表分析、多语言应用的完整示例
- 从任务约束反推模型、分辨率和后处理策略

## 学习时间

- **快速版**（仅阅读正文）：25分钟
- **标准版**（包含代码实验）：60分钟
- **完整版**（包含扩展内容）：90分钟

## 核心问题

完成本章后，你应该能回答：

1. ViT 如何将图像转换为 Transformer 可处理的序列？
2. CLIP 的对比学习如何实现视觉-语言对齐？
3. 多模态模型如何把视觉 token 接入语言模型？
4. 三种高分辨率处理方案各有什么权衡？
5. 如何根据任务、语言、分辨率、成本和部署方式选择多模态模型？

## 代码实验

本章共有 **9 个代码脚本**，覆盖架构说明图、ViT、CLIP、高分辨率处理、多模态应用和实战案例（含7.2/7.3说明图）。

| 小节 | 脚本 | 生成图表 | 内容 |
|------|------|---------|------|
| 7.1 | [`architecture_diagrams.py`](../../code/ch07_multimodal_llm/architecture_diagrams.py) | `ch07_vit_cnn_comparison.png`<br>`ch07_temperature_effect.png` | CNN vs ViT 感受野对比、温度参数效果 |
| 7.2/7.3 | [`explainer_diagrams.py`](../../code/ch07_multimodal_llm/explainer_diagrams.py) | `ch07_architecture_comparison.png`<br>`ch07_fusion_strategies.png`<br>`ch07_dynamic_resolution.png`<br>`ch07_image_tiling.png` | LLaVA vs Qwen2.5-VL 对比、融合策略、动态分辨率、分块示意 |
| 7.1 | [`vit_patches.py`](../../code/ch07_multimodal_llm/vit_patches.py) | `ch07_vit_patches.png` | ViT 图像分块可视化 |
| 7.1 | [`clip_similarity.py`](../../code/ch07_multimodal_llm/clip_similarity.py) | `ch07_clip_similarity.png` | CLIP 图像-文本相似度矩阵 |
| 7.1 | [`clip_alignment_demo.py`](../../code/ch07_multimodal_llm/clip_alignment_demo.py) | `ch07_clip_alignment.png` | CLIP 对齐演示（需 `openai-clip`） |
| 7.2 | [`qwen_vl_analysis.py`](../../code/ch07_multimodal_llm/qwen_vl_analysis.py) | `ch07_qwen_vl_analysis.png` | Qwen2.5-VL vs LLaVA 架构对比 |
| 7.3 | [`high_resolution_processing.py`](../../code/ch07_multimodal_llm/high_resolution_processing.py) | `ch07_high_resolution_processing.png` | 三种高分辨率方案对比 |
| 7.4 | [`multimodal_applications.py`](../../code/ch07_multimodal_llm/multimodal_applications.py) | `ch07_multimodal_applications.png` | 多模态应用性能对比 |
| 7.5 | [`case_studies.py`](../../code/ch07_multimodal_llm/case_studies.py) | `ch07_case_studies.png` | 三个实战案例性能对比 |

**运行方式：**
```bash
python code/ch07_multimodal_llm/architecture_diagrams.py
python code/ch07_multimodal_llm/explainer_diagrams.py
python code/ch07_multimodal_llm/vit_patches.py
python code/ch07_multimodal_llm/clip_similarity.py
python code/ch07_multimodal_llm/qwen_vl_analysis.py
python code/ch07_multimodal_llm/high_resolution_processing.py
python code/ch07_multimodal_llm/multimodal_applications.py
python code/ch07_multimodal_llm/case_studies.py
```

> `clip_alignment_demo.py` 需额外安装 `openai-clip` 包（见 [附录B：环境配置](../appendix/B_environment_setup.md)）。


## 推荐学习路径

### 路径1：快速入门（25分钟）
- 阅读 7.1-7.2 的正文
- 理解 CLIP 对齐和 ViT 分块的核心思想
- 重点：对比学习的几何直观

### 路径2：标准学习（60分钟）
- 阅读所有正文（7.1-7.5）
- 运行 ViT 和 CLIP 实验
- 回答"核心问题"中的 5 个问题

### 路径3：深度学习（90分钟）
- 阅读所有正文和扩展内容
- 运行所有代码实验
- 深入理解高分辨率处理的权衡
- 阅读 CLIP、ViT 原始论文

## 关键概念速查

| 概念 | 核心思想 | 章节 |
|------|---------|------|
| ViT | 图像分块 → Patch 嵌入 → Transformer 编码 | 7.1 |
| CLIP | 对比学习：匹配对相似度高，不匹配对低 | 7.1 |
| 投影层 | 将视觉特征映射到 LLM 的语义空间 | 7.2 |
| 动态分辨率 | 根据图像大小自动调整处理分辨率 | 7.3 |
| 图像分块 | 将大图分成多个小块分别处理 | 7.3 |

## 常见问题

**Q: ViT 和 CNN 有什么区别？**  
A: CNN 用局部卷积核提取特征，感受野逐层扩大；ViT 把图像分成 patches，用 Transformer 的全局注意力直接建模 patch 之间的关系。ViT 在大规模训练下常有很强表现，但计算成本和数据需求也更高。

**Q: CLIP 为什么用对比学习而不是直接分类？**  
A: 对比学习不需要固定的类别标签，可以用互联网上海量的图像-文本对（图片+描述）训练，泛化能力更强。

**Q: 为什么高分辨率对文档理解帮助大？**  
A: 文档中的文字、表格、图表需要高分辨率才能识别细节。224×224 的标准分辨率下，A4 文档上的文字几乎无法辨认。

**Q: 如何选择多模态模型？**  
A: 需要结合任务、语言、预算、部署方式、分辨率需求和失败成本重新评估。本章以 Qwen2.5-VL、LLaVA 和 CLIP 为代表案例，重点学习选型维度，而不是给出永久固定的模型推荐。

## 扩展内容

### 多模态对齐训练深度细节 — [📖 阅读](extensions/multimodal_training_details.md)
- InfoNCE 损失完整推导与温度参数 τ 的分析
- 批大小对负样本数量的影响（256 vs 4096 vs 32768）
- 投影层设计：MLP vs Q-Former 对比
- 三阶段训练策略（冻结 ViT → 指令微调 → 全参数微调）

### 多模态应用进阶 — [📖 阅读](extensions/multimodal_applications_advanced.md)
- 多模态 RAG：双路检索（CLIP 图像向量 + 文本向量）与 RRF 融合
- OCR + LLM 混合文档理解技术栈
- 视觉 Agent 工具调用模式
- 模型选择决策树与常见坑（CLIP 模板、OOM、多图混乱）

## 关键连接点

### 第4章 Transformer → 第7章 ViT

```
第4章：Transformer 处理文本序列（token 序列）
    ↓ 同样的架构
第7章：ViT 处理图像（patch 序列）
    图像 → 分块 → Patch 嵌入 → Transformer
```

### 第5章 Embedding → 第7章 CLIP

```
第5章：token embedding（离散符号 → 连续向量）
    ↓ 同样的思想，扩展到跨模态
第7章：CLIP（图像 → 向量，文本 → 向量，对齐到同一空间）
```

---

**下一步：** 阅读 [7.1 视觉-语言对齐](01_vision_language.md)
