# 第7章：多模态LLM

**版本：** v3.1  
**最后更新：** 2026-05-30

## 章节概览

本章介绍多模态 LLM，回答一个核心问题：**如何让语言模型"看懂"图像？**

从 CLIP 的对比学习对齐，到 ViT 的图像分块，再到 Qwen2.5-VL 的高分辨率处理——这条演化链展示了视觉-语言融合的技术进步。

## 快速导航

| 章节 | 文件 | 难度 | 时间 |
|------|------|------|------|
| 7.1 视觉-语言对齐 | [01_vision_language.md](01_vision_language.md) | ⭐⭐⭐ | 15分钟 |
| 7.2 Qwen2.5-VL 架构 | [02_qwen_vl.md](02_qwen_vl.md) | ⭐⭐⭐ | 15分钟 |
| 7.3 高分辨率图像处理 | [03_high_resolution.md](03_high_resolution.md) | ⭐⭐⭐ | 10分钟 |
| 7.4 多模态应用 | [04_applications.md](04_applications.md) | ⭐⭐ | 10分钟 |
| 7.5 实战案例 | [05_case_studies.md](05_case_studies.md) | ⭐⭐⭐ | 10分钟 |

## 小节目录

### 基础与架构（7.1-7.2）

**7.1 视觉-语言对齐** — [📖 阅读](01_vision_language.md)
- ViT：图像分块 + Transformer 编码
- CLIP：对比学习实现图像-文本对齐
- 对齐的几何直观：相似对靠近，不相似对远离

**7.2 Qwen2.5-VL 架构** — [📖 阅读](02_qwen_vl.md)
- LLaVA 架构（单编码器 + 投影层 + LLM）
- Qwen2.5-VL 的改进：高分辨率、多图像、中文优化
- 混合融合策略 vs 早期/晚期融合

### 技术细节（7.3）

**7.3 高分辨率图像处理** — [📖 阅读](03_high_resolution.md)
- 为什么标准 224×224 不够用
- 三种方案：动态分辨率、图像分块、自适应采样
- 方案权衡：信息量 vs 计算成本

### 应用（7.4-7.5）

**7.4 多模态应用** — [📖 阅读](04_applications.md)
- 图像问答、文档理解、图表分析
- 模型选择框架

**7.5 实战案例** — [📖 阅读](05_case_studies.md)
- 文档理解、图表分析、多语言应用的完整示例

## 学习时间

- **快速版**（仅阅读正文）：25分钟
- **标准版**（包含代码实验）：60分钟
- **完整版**（包含扩展内容）：90分钟

## 核心问题

完成本章后，你应该能回答：

1. ViT 如何将图像转换为 Transformer 可处理的序列？
2. CLIP 的对比学习如何实现视觉-语言对齐？
3. 为什么 Qwen2.5-VL 比 LLaVA 支持更高分辨率？
4. 三种高分辨率处理方案各有什么权衡？
5. 如何根据任务需求选择合适的多模态模型？

## 代码实验

本章共有 **4 个代码脚本**，覆盖 ViT、CLIP、高分辨率处理和架构分析。

| 小节 | 脚本 | 内容 | 文档位置 |
|------|------|------|---------|
| 7.1 | [`vit_patches.py`](../../code/ch07_multimodal_llm/vit_patches.py) | ViT 图像分块可视化 | [01_vision_language.md](01_vision_language.md) |
| 7.1 | [`clip_similarity.py`](../../code/ch07_multimodal_llm/clip_similarity.py) | CLIP 图像-文本相似度 | [01_vision_language.md](01_vision_language.md) |
| 7.1 | [`clip_alignment_demo.py`](../../code/ch07_multimodal_llm/clip_alignment_demo.py) | CLIP 对齐演示（需 openai-clip） | [01_vision_language.md](01_vision_language.md) |
| 7.2 | [`qwen_vl_analysis.py`](../../code/ch07_multimodal_llm/qwen_vl_analysis.py) | Qwen2.5-VL 架构分析 | [02_qwen_vl.md](02_qwen_vl.md) |
| 7.3 | [`high_resolution_processing.py`](../../code/ch07_multimodal_llm/high_resolution_processing.py) | 高分辨率处理方案对比 | [03_high_resolution.md](03_high_resolution.md) |
| 7.4 | [`multimodal_applications.py`](../../code/ch07_multimodal_llm/multimodal_applications.py) | 多模态应用演示 | [04_applications.md](04_applications.md) |
| 7.5 | [`case_studies.py`](../../code/ch07_multimodal_llm/case_studies.py) | 实战案例演示 | [05_case_studies.md](05_case_studies.md) |

### 实验1：ViT Patches 可视化

- **文件：** [`code/ch07_multimodal_llm/vit_patches.py`](../../code/ch07_multimodal_llm/vit_patches.py)
- **内容：** 展示 Vision Transformer 如何将图像分割成 patches，理解 ViT 的基本原理
- **运行：** `python code/ch07_multimodal_llm/vit_patches.py`
- **输出：** Patches 可视化、Patch 嵌入、ViT 处理流程

**代码文件：** `code/ch07_multimodal_llm/vit_patches.py`  
**运行方式：** `python code/ch07_multimodal_llm/vit_patches.py`

![ViT Patches](../../assets/ch07_vit_patches.png)

*图7.1：Vision Transformer 的 Patches 可视化。展示 ViT 如何将图像分割成小块并进行处理。*

### 实验2：CLIP 相似度计算

- **文件：** [`code/ch07_multimodal_llm/clip_similarity.py`](../../code/ch07_multimodal_llm/clip_similarity.py)
- **内容：** 计算图像和文本的相似度，理解视觉-语言对齐的原理
- **运行：** `python code/ch07_multimodal_llm/clip_similarity.py`
- **输出：** 相似度矩阵、检索结果、对齐效果演示

**代码文件：** `code/ch07_multimodal_llm/clip_similarity.py`  
**运行方式：** `python code/ch07_multimodal_llm/clip_similarity.py`

![CLIP Similarity](../../assets/ch07_clip_similarity.png)

*图7.2：CLIP 的图像-文本相似度。展示如何通过对比学习实现视觉-语言对齐。*

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
A: CNN 用局部卷积核提取特征，感受野逐层扩大；ViT 把图像分成 patches，用 Transformer 的全局注意力直接建模所有 patch 之间的关系。ViT 在大数据集上通常更强，但需要更多数据训练。

**Q: CLIP 为什么用对比学习而不是直接分类？**  
A: 对比学习不需要固定的类别标签，可以用互联网上海量的图像-文本对（图片+描述）训练，泛化能力更强。

**Q: 为什么高分辨率对文档理解帮助大？**  
A: 文档中的文字、表格、图表需要高分辨率才能识别细节。224×224 的标准分辨率下，A4 文档上的文字几乎无法辨认。

**Q: 如何选择多模态模型？**  
A: 开源首选 Qwen2.5-VL（高分辨率、中文优化）；闭源首选 GPT-4V/Claude；资源有限用 LLaVA-1.5。

## 扩展内容

### 推荐论文与进阶资源 — [📖 阅读](extensions/resources.md)
- 模型选型决策树（CLIP vs LLaVA vs GPT-4V 的适用场景）
- 推荐论文（CLIP、ViT、LLaVA、BLIP-2、Flamingo、InstructBLIP）
- 常见坑与解决方案（分辨率、中文支持、幻觉问题）
- 进阶学习路径（视频理解、音频-视觉、具身智能）

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
