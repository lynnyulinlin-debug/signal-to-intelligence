"""Pytest configuration and shared fixtures."""
import importlib.util
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def seed():
    """固定随机种子以确保可复现性"""
    np.random.seed(42)
    return 42


@pytest.fixture
def sample_signal():
    """生成示例信号用于测试"""
    t = np.linspace(0, 1, 100)
    signal = np.sin(2 * np.pi * 5 * t) + 0.1 * np.random.randn(100)
    return signal


@pytest.fixture
def load_code_module():
    """按文件路径加载 code/ 下的实验脚本，避免与标准库 code 模块重名。"""

    def _load(relative_path: str):
        module_path = ROOT / relative_path
        module_name = relative_path.replace("/", "_").replace(".py", "")
        spec = importlib.util.spec_from_file_location(module_name, module_path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    return _load
