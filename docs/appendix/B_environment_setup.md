# 环境配置指南

**版本：** v1.0  
**最后更新：** 2026-05-13

本指南详细说明如何配置《Signals to Intelligence》教程的开发环境。

---

## 目录

1. [系统要求](#系统要求)
2. [Python 环境配置](#python-环境配置)
3. [依赖安装](#依赖安装)
4. [验证安装](#验证安装)
5. [常见问题](#常见问题)
6. [IDE 配置（可选）](#ide-配置可选)

---

## 系统要求

### 操作系统

- **Linux** — Ubuntu 20.04 LTS 或更新版本
- **macOS** — 10.14 或更新版本
- **Windows** — Windows 10 或更新版本（推荐使用 WSL2）

### Python 版本

- **Python 3.10+**（推荐 3.11 或 3.12）

检查 Python 版本：
```bash
python3 --version
```

如果系统中有多个 Python 版本，使用特定版本创建虚拟环境：
```bash
python3.10 -m venv venv  # 或 python3.11, python3.12
```

### 磁盘空间

- 最小：500 MB（不含数据集）
- 推荐：2 GB（含可选依赖）

---

## Python 环境配置

### 步骤1：克隆仓库

```bash
git clone <repo-url>
cd signals-to-intelligence
```

### 步骤2：创建虚拟环境

使用 Python 内置的 `venv` 模块创建虚拟环境：

```bash
python3 -m venv venv
```

### 步骤3：激活虚拟环境

**Linux / macOS：**
```bash
source venv/bin/activate
```

**Windows (PowerShell)：**
```powershell
venv\Scripts\Activate.ps1
```

**Windows (CMD)：**
```cmd
venv\Scripts\activate.bat
```

激活成功后，命令行提示符会显示 `(venv)` 前缀。

### 步骤4：升级 pip

```bash
pip install --upgrade pip setuptools wheel
```

---

## 依赖安装

### 安装核心依赖

```bash
pip install -r requirements.txt
```

这会安装：
- **numpy** — 数值计算
- **matplotlib** — 绘图
- **pytest** — 测试框架
- **pytest-cov** — 代码覆盖率

### 安装可选依赖（第6-8章）

如果你计划学习深度学习章节（第6-8章），还需要安装 PyTorch：

```bash
# CPU 版本（推荐用于学习）
pip install torch torchvision

# GPU 版本（CUDA 11.8）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# GPU 版本（CUDA 12.1）
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**选择 CPU 还是 GPU？**
- **CPU 版本** — 更轻量，适合学习和实验
- **GPU 版本** — 更快，适合大规模训练（本教程不需要）

### 安装开发工具（可选）

```bash
# Jupyter 笔记本（可选）
pip install jupyter ipython

# 代码检查工具（可选）
pip install flake8 black
```

---

## 验证安装

### 验证 Python 和虚拟环境

```bash
which python3  # 应该显示 venv 目录下的 python
python3 --version
```

### 验证核心依赖

```bash
python3 -c "import numpy; print(f'NumPy {numpy.__version__}')"
python3 -c "import matplotlib; print(f'Matplotlib {matplotlib.__version__}')"
python3 -c "import pytest; print(f'pytest {pytest.__version__}')"
```

### 运行第一个实验

```bash
python3 code/ch01_dsp/fft_spectrum.py
```

预期输出：
```
==================================================
FFT 频谱分析
==================================================
信号长度: 1000
频率分量: [5.0, 10.0]
==================================================
```

### 运行测试

```bash
pytest tests/ -v
```

预期输出：
```
tests/conftest.py::test_seed PASSED
tests/conftest.py::test_sample_signal PASSED
```

---

## 常见问题

### Q1: 如何退出虚拟环境？

```bash
deactivate
```

### Q2: 如何删除虚拟环境？

```bash
rm -rf venv  # Linux/macOS
rmdir /s venv  # Windows
```

### Q3: 如何重新安装依赖？

```bash
pip install --force-reinstall -r requirements.txt
```

### Q4: 如何检查已安装的包？

```bash
pip list
```

### Q5: 如何升级某个包？

```bash
pip install --upgrade numpy
```

### Q6: 在 Windows 上遇到权限错误？

如果遇到 `Permission denied` 错误，尝试以管理员身份运行 PowerShell 或 CMD。

### Q7: PyTorch 安装失败？

检查网络连接，或使用清华大学镜像：

```bash
pip install torch torchvision -i https://pypi.tsinghua.edu.cn/simple
```

### Q8: 如何使用 Jupyter 笔记本？

```bash
# 安装 Jupyter
pip install jupyter

# 启动 Jupyter
jupyter notebook

# 在浏览器中打开 http://localhost:8888
```

---

## IDE 配置（可选）

### VS Code

1. 安装 Python 扩展
2. 打开命令面板（Ctrl+Shift+P）
3. 搜索 "Python: Select Interpreter"
4. 选择 `./venv/bin/python`

### PyCharm

1. 打开 Settings → Project → Python Interpreter
2. 点击齿轮图标 → Add
3. 选择 "Existing Environment"
4. 选择 `./venv/bin/python`

### Jupyter Lab（可选）

```bash
pip install jupyterlab
jupyter lab
```

---

## 环境变量（可选）

如果需要调试，可以设置以下环境变量：

```bash
# 启用 NumPy 调试模式
export NUMPY_EXPERIMENTAL_ARRAY_FUNCTION=1

# 启用 PyTorch 调试模式
export TORCH_DISTRIBUTED_DEBUG=INFO
```

---

## 下一步

1. 阅读 [README.md](../README.md) 了解项目概览
2. 查看 [C_code_guide.md](C_code_guide.md) 了解如何运行代码实验
3. 开始阅读 [第0章：导论](../00_introduction.md)

---

## 获取帮助

- 遇到问题？查看 [常见问题](#常见问题) 部分
- 需要更多帮助？提交 [GitHub Issue](https://github.com/your-repo/issues)
