"""Tests for Chapter 2: Optimization"""
import numpy as np
import pytest


class TestLMSOptimizer:
    """Test LMS optimizer"""

    def test_lms_convergence(self, seed):
        """Test that LMS converges"""
        n_samples = 100
        n_features = 5
        learning_rate = 0.01
        epochs = 50

        # Generate data
        w_true = np.random.randn(n_features)
        X = np.random.randn(n_samples, n_features)
        y = X @ w_true + 0.1 * np.random.randn(n_samples)

        # LMS algorithm
        w_lms = np.zeros(n_features)
        losses = []

        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                y_pred = x_i @ w_lms
                error = y_i - y_pred
                w_lms += 2 * learning_rate * error * x_i

            y_pred_lms = X @ w_lms
            loss = np.mean((y - y_pred_lms) ** 2)
            losses.append(loss)

        # Check convergence
        assert losses[-1] < losses[0]
        assert not np.any(np.isnan(losses))

    def test_lms_weight_estimation(self, seed):
        """Test that LMS estimates weights correctly"""
        n_samples = 200
        n_features = 3
        learning_rate = 0.01
        epochs = 100

        w_true = np.array([1.0, -0.5, 2.0])
        X = np.random.randn(n_samples, n_features)
        y = X @ w_true

        w_lms = np.zeros(n_features)

        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                y_pred = x_i @ w_lms
                error = y_i - y_pred
                w_lms += 2 * learning_rate * error * x_i

        # Check if estimated weights are close to true weights
        assert np.allclose(w_lms, w_true, atol=0.1)


class TestAdamOptimizer:
    """Test Adam optimizer"""

    def test_adam_convergence(self, seed):
        """Test that Adam converges"""
        n_samples = 100
        n_features = 5
        learning_rate = 0.01
        epochs = 50

        # Generate data
        w_true = np.random.randn(n_features)
        X = np.random.randn(n_samples, n_features)
        y = X @ w_true + 0.1 * np.random.randn(n_samples)

        # Adam optimizer
        w_adam = np.zeros(n_features)
        m = np.zeros(n_features)
        v = np.zeros(n_features)
        beta1, beta2 = 0.9, 0.999
        epsilon = 1e-8
        losses = []

        for epoch in range(epochs):
            y_pred = X @ w_adam
            error = y_pred - y
            grad = X.T @ error / n_samples

            m = beta1 * m + (1 - beta1) * grad
            v = beta2 * v + (1 - beta2) * (grad ** 2)

            m_hat = m / (1 - beta1 ** (epoch + 1))
            v_hat = v / (1 - beta2 ** (epoch + 1))

            w_adam -= learning_rate * m_hat / (np.sqrt(v_hat) + epsilon)

            y_pred = X @ w_adam
            loss = np.mean((y - y_pred) ** 2)
            losses.append(loss)

        # Check convergence
        assert losses[-1] < losses[0]
        assert not np.any(np.isnan(losses))

    def test_adam_vs_lms(self, seed):
        """Test that Adam converges faster than LMS"""
        n_samples = 100
        n_features = 5
        epochs = 50

        w_true = np.random.randn(n_features)
        X = np.random.randn(n_samples, n_features)
        y = X @ w_true + 0.1 * np.random.randn(n_samples)

        # LMS
        w_lms = np.zeros(n_features)
        losses_lms = []
        for epoch in range(epochs):
            for i in range(n_samples):
                x_i = X[i]
                y_i = y[i]
                y_pred = x_i @ w_lms
                error = y_i - y_pred
                w_lms += 2 * 0.01 * error * x_i
            y_pred_lms = X @ w_lms
            loss = np.mean((y - y_pred_lms) ** 2)
            losses_lms.append(loss)

        # Adam
        w_adam = np.zeros(n_features)
        m = np.zeros(n_features)
        v = np.zeros(n_features)
        losses_adam = []
        for epoch in range(epochs):
            y_pred = X @ w_adam
            error = y_pred - y
            grad = X.T @ error / n_samples
            m = 0.9 * m + 0.1 * grad
            v = 0.999 * v + 0.001 * (grad ** 2)
            m_hat = m / (1 - 0.9 ** (epoch + 1))
            v_hat = v / (1 - 0.999 ** (epoch + 1))
            w_adam -= 0.01 * m_hat / (np.sqrt(v_hat) + 1e-8)
            y_pred = X @ w_adam
            loss = np.mean((y - y_pred) ** 2)
            losses_adam.append(loss)

        # Adam should converge faster (lower loss at the end)
        assert losses_adam[-1] <= losses_lms[-1]
