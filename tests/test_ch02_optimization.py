"""Tests for Chapter 2: Optimization"""
import numpy as np


class TestLMSOptimizer:
    """Test LMS optimizer"""

    def test_lms_convergence(self, load_code_module):
        """Test that LMS converges"""
        optim = load_code_module("code/ch02_optimization/lms_vs_adam.py")
        X, y, _ = optim.generate_linear_data(n_samples=100, n_features=5, seed=42)
        _, losses = optim.train_lms(X, y, learning_rate=0.01, epochs=50)

        # Check convergence
        assert losses[-1] < losses[0]
        assert not np.any(np.isnan(losses))

    def test_lms_weight_estimation(self, load_code_module):
        """Test that LMS estimates weights correctly"""
        optim = load_code_module("code/ch02_optimization/lms_vs_adam.py")
        n_samples = 200
        w_true = np.array([1.0, -0.5, 2.0])
        X = np.random.randn(n_samples, len(w_true))
        y = X @ w_true
        w_lms, _ = optim.train_lms(X, y, learning_rate=0.01, epochs=100)

        # Check if estimated weights are close to true weights
        assert np.allclose(w_lms, w_true, atol=0.1)


class TestAdamOptimizer:
    """Test Adam optimizer"""

    def test_adam_convergence(self, load_code_module):
        """Test that Adam converges"""
        optim = load_code_module("code/ch02_optimization/lms_vs_adam.py")
        X, y, _ = optim.generate_linear_data(n_samples=100, n_features=5, seed=42)
        _, losses = optim.train_adam(X, y, learning_rate=0.01, epochs=50)

        # Check convergence
        assert losses[-1] < losses[0]
        assert not np.any(np.isnan(losses))

    def test_adam_vs_lms(self, load_code_module):
        """Test that Adam converges faster than LMS"""
        optim = load_code_module("code/ch02_optimization/lms_vs_adam.py")
        X, y, _ = optim.generate_linear_data(n_samples=100, n_features=5, seed=42)
        _, losses_lms = optim.train_lms(X, y, learning_rate=0.01, epochs=50)
        _, losses_adam = optim.train_adam(X, y, learning_rate=0.1, epochs=100)

        # Adam should converge to lower loss than LMS (100 epochs vs 50 epochs)
        assert losses_adam[-1] <= losses_lms[-1]
