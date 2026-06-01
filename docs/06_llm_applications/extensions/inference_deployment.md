# 扩展：推理部署优化

**所属章节：** [第6章：LLM应用](../README.md)  
**原理层内容：** 见 [第5章 E5.x 推理机制优化](../../05_llm_basics/extensions/inference_optimization.md)

---

## 概览

第5章扩展介绍了 KV Cache、Flash Attention、量化的**原理**。本节聚焦**工程选型**：在实际部署中，如何选择量化方案、推理框架和批处理策略。

---

## 量化方案选型

### 精度 vs 速度 vs 显存

```
FP16（半精度）：
  显存：~14GB（7B 模型）
  速度：基准
  精度：无损失
  适用：有充足 GPU 显存，追求最高精度

INT8（8-bit 量化）：
  显存：~7GB（7B 模型）
  速度：接近 FP16（部分操作更快）
  精度：损失极小（<1%）
  适用：显存有限，对精度要求高

INT4（4-bit 量化）：
  显存：~3.5GB（7B 模型）
  速度：比 FP16 快 1.5-2 倍
  精度：有一定损失（1-3%）
  适用：消费级 GPU，可接受轻微精度损失
```

### 主流量化方案对比

| 方案 | 精度 | 速度 | 工具 | 适用场景 |
|------|------|------|------|---------|
| FP16 | 无损 | 基准 | 原生 | 服务器部署，精度优先 |
| LLM.int8() | 极小损失 | 接近 FP16 | bitsandbytes | 显存有限的服务器 |
| GPTQ | 小损失 | 快 1.5x | AutoGPTQ | GPU 推理，精度与速度平衡 |
| AWQ | 小损失 | 快 1.5x | AutoAWQ | GPU 推理，比 GPTQ 精度略好 |
| GGUF | 可配置 | CPU 可用 | llama.cpp | CPU 推理，本地部署 |

**推荐选择：**
- 服务器 GPU 部署 → AWQ INT4 或 GPTQ INT4
- 消费级 GPU（RTX 3090/4090）→ GGUF Q4_K_M 或 AWQ
- CPU 推理（无 GPU）→ GGUF Q4_K_M（llama.cpp）
- 精度优先 → FP16 或 LLM.int8()

---

## 推理框架选型

### vLLM

**核心技术：** PagedAttention——将 KV Cache 分页管理，类似操作系统的虚拟内存。

```
传统 KV Cache：
  为每个请求预分配最大序列长度的连续显存
  → 大量显存碎片和浪费

PagedAttention：
  KV Cache 分成固定大小的"页"
  按需分配，不同请求可以共享物理页
  → 显存利用率提升 2-4 倍
  → 吞吐量提升 2-4 倍
```

**适用场景：** 高并发在线服务，追求最大吞吐量。

**快速启动：**
```bash
pip install vllm
python -m vllm.entrypoints.openai.api_server \
    --model Qwen/Qwen2.5-7B-Instruct \
    --quantization awq
```

### llama.cpp

**特点：** 纯 C++ 实现，支持 CPU 推理，跨平台（Mac/Linux/Windows）。

**适用场景：** 本地部署、无 GPU 环境、边缘设备。

```bash
# 下载 GGUF 格式模型
# 运行推理
./llama-cli -m model.gguf -p "你好" -n 100
```

### Ollama

**特点：** 封装 llama.cpp，提供简单的 CLI 和 REST API，一键运行本地模型。

```bash
ollama run qwen2.5:7b
```

**适用场景：** 开发测试、个人使用，不需要高并发。

### TGI（Text Generation Inference）

**特点：** Hugging Face 官方推理服务，支持 Flash Attention、连续批处理。

**适用场景：** 中等规模服务，与 Hugging Face 生态集成。

---

## 批处理策略

### 静态批处理 vs 连续批处理

```
静态批处理（Static Batching）：
  等待一批请求凑齐后一起处理
  问题：不同请求生成长度不同，短请求要等长请求完成
  → GPU 利用率低

连续批处理（Continuous Batching）：
  请求完成后立即加入新请求，不等待整批完成
  → GPU 利用率提升 2-3 倍
  → vLLM、TGI 均支持
```

### 推测解码（Speculative Decoding）

**核心思想：** 用小模型（草稿模型）快速生成多个候选 token，用大模型（目标模型）并行验证。

```
草稿模型（小，快）：生成 [token₁, token₂, token₃, token₄]
目标模型（大，慢）：并行验证这 4 个 token
  → 全部接受：一次前向传播生成 4 个 token
  → 部分接受：接受前 k 个，重新生成第 k+1 个

平均加速：1.5-3 倍（取决于草稿模型的接受率）
```

**适用场景：** 延迟敏感的在线服务，草稿模型和目标模型来自同一家族（如 LLaMA-3-8B 草稿 + LLaMA-3-70B 目标）。

---

## 部署架构选型

| 场景 | 推荐方案 | 说明 |
|------|---------|------|
| 个人开发测试 | Ollama | 一键启动，简单易用 |
| 小团队内部服务 | vLLM + AWQ | 高吞吐，成本可控 |
| 生产高并发服务 | vLLM + 多卡 | 水平扩展，负载均衡 |
| 无 GPU 环境 | llama.cpp GGUF | CPU 推理，性能有限 |
| 边缘/移动设备 | MLC-LLM | 专为移动端优化 |
| 不想自运维 | 托管 API | OpenAI/Anthropic/Together |

---

## 成本估算

```
自部署成本（示例：Qwen2.5-7B，AWQ INT4）：
  显存需求：~5GB → RTX 3090（24GB）可运行
  吞吐量：~500 tokens/s（单卡）
  云服务器成本：~$0.5-1/小时（A10G）

API 成本（示例：GPT-4o-mini）：
  输入：$0.15/1M tokens
  输出：$0.6/1M tokens
  
  日均 100 万 tokens → ~$0.75/天 → ~$22/月

自部署 vs API 的盈亏平衡点：
  月均 token 量 > ~50M → 自部署更划算
  月均 token 量 < ~50M → API 更划算（无运维成本）
```

---

**返回：** [第6章：LLM应用](../README.md)
