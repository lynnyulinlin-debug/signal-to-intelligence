# deploy/

本目录存放容器化部署配置文件。

## 文件说明

| 文件 | 用途 |
|------|------|
| `docker-compose.yml` | CPU 版 Jupyter 环境，适合第 1-5 章离线实验 |
| `docker-compose.gpu.yml` | GPU 版 Jupyter 环境，适合第 5-6 章开源模型实验 |

## 快速启动

```bash
# CPU 版（第 1-5 章，不需要 GPU）
make docker-up

# GPU 版（第 5-6 章，需要 NVIDIA GPU + nvidia-container-toolkit）
make docker-up-gpu

# 停止
make docker-down
```

访问 http://localhost:8888

## 前置条件

**CPU 版：** 安装 [Docker Desktop](https://www.docker.com/products/docker-desktop/)

**GPU 版：** 额外需要：
1. NVIDIA 驱动（`nvidia-smi` 验证）
2. [nvidia-container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

## 环境变量

容器启动时会自动读取根目录的 `.env` 文件。首次使用：

```bash
cp .env.example .env
# 编辑 .env，填入 API Key
```
