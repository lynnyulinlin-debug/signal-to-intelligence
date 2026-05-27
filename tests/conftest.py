"""Pytest configuration and shared fixtures."""
import numpy as np
import pytest


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
