# 如何使用本教程

本教程有四种使用方式。默认入口是在线阅读 VitePress；代码、Notebook 和云端环境用于补充实验和复现。

---

## 推荐入口

| 使用方式 | 适合场景 | 主要入口 |
|---------|----------|----------|
| 在线阅读 VitePress | 系统学习正文、查阅章节结构 | `docs/` 发布后的文档站 |
| 本地运行代码 | 复现实验、生成图片、修改参数 | `code/`、`tests/` |
| Notebook 交互实验 | 教学演示、边改参数边观察结果 | `notebooks/` |
| CNB 云端环境 | 不想配置本地环境、需要云端算力 | 云端配置说明 |

正文以 VitePress 文档为准；核心代码以 `code/` 为准；Notebook 是交互补充。

---

## 方式一：在线阅读 VitePress

适合：

- 只想系统学习内容
- 不急着运行代码
- 用浏览器、手机或平板阅读

在线文档由 GitHub Pages workflow 构建。正文来自 `docs/`，图片来自 `assets/` 同步到 `docs/public/assets/` 后的发布产物。

本地预览：

```bash
npm run docs:dev
```

构建检查：

```bash
npm run docs:build
```

---

## 方式二：本地运行代码

适合：

- 复现实验结果
- 重新生成图表
- 修改脚本参数
- 检查教程中的代码是否仍然可运行

基本命令：

```bash
pip install -e .
python code/ch01_dsp/fft_spectrum.py
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/ -q
```

更完整的代码运行说明见 [附录 C：代码运行指南](../appendix/C_code_guide.md)。

---

## 方式三：Notebook 交互实验

Notebook 的定位：

- 是交互实验补充
- 不作为教程正文的第二份源头
- 不复制完整章节解释
- 核心算法优先调用 `code/` 中的函数

适合：

- 教学演示
- 边改参数边看图
- 在本地 Jupyter、Colab 或 CNB 中逐步执行

启动：

```bash
jupyter notebook
```

Notebook 通过 `notebooks/project.py` 加载 `code/` 下的脚本，避免一份算法同时维护在 `.py`、测试和 `.ipynb` 里。

---

## 方式四：CNB 云端环境

适合：

- 不想配置本地 Python 环境
- 临时运行代码或 Notebook
- 没有本地 GPU
- 想快速复现实验

当前默认原则：

- 第 1-6 章默认不依赖 GPU
- 第 7 章多模态实验以轻量 NumPy 演示为主
- 第 8 章偏工程配置和成本分析，不默认依赖 GPU

云端环境配置后续统一放在附录 B 或独立部署说明中，避免 README、Notebook 和正文重复维护。

---

## 如何选择

| 目标 | 推荐方式 |
|------|----------|
| 先理解整体脉络 | 在线阅读 VitePress |
| 验证公式和图表 | 本地运行 `code/` |
| 课堂演示或调参观察 | Notebook |
| 免配置体验 | CNB 云端环境 |
| 维护项目内容 | 阅读 [附录 E：项目维护说明](../appendix/E_project_maintenance.md) |

如果不确定从哪里开始，先阅读 [学习路径](02_learning_paths.md)，再按章节顺序阅读正文。
