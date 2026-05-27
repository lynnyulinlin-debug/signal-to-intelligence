"""Tests for Chapter 2: Optimization and Machine Learning"""
import numpy as np
import pytest


class TestLinearRegression:
    """Test linear regression"""

    def test_linear_regression_convergence(self, seed):
        """Test that linear regression converges"""
        n_samples = 100
        X = np.random.randn(n_samples, 1)
        y = 2 * X + 1 + 0.1 * np.random.randn(n_samples, 1)

        w = np.random.randn(1, 1) * 0.1
        b = 0.0
        learning_rate = 0.01
        epochs = 50

        losses = []
        for epoch in range(epochs):
            y_pred = X @ w + b
            loss = np.mean((y_pred - y) ** 2)
            losses.append(loss)

            dw = 2 * X.T @ (y_pred - y) / n_samples
            db = 2 * np.mean(y_pred - y)

            w -= learning_rate * dw
            b -= learning_rate * db

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_linear_regression_parameter_estimation(self, seed):
        """Test that linear regression estimates parameters correctly"""
        n_samples = 200
        X = np.random.randn(n_samples, 1)
        w_true = 2.0
        b_true = 1.0
        y = w_true * X + b_true

        w = np.random.randn(1, 1) * 0.1
        b = 0.0
        learning_rate = 0.01
        epochs = 100

        for epoch in range(epochs):
            y_pred = X @ w + b
            dw = 2 * X.T @ (y_pred - y) / n_samples
            db = 2 * np.mean(y_pred - y)
            w -= learning_rate * dw
            b -= learning_rate * db

        # Check if estimated parameters are close to true parameters
        assert np.abs(w[0, 0] - w_true) < 0.5
        assert np.abs(b - b_true) < 0.5


class TestLogisticRegression:
    """Test logistic regression"""

    def test_logistic_regression_convergence(self, seed):
        """Test that logistic regression converges"""
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

        w = np.random.randn(2, 1) * 0.1
        b = 0.0
        learning_rate = 0.01
        epochs = 50

        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        losses = []
        for epoch in range(epochs):
            z = X @ w + b
            y_pred = sigmoid(z)
            loss = -np.mean(y * np.log(y_pred + 1e-8) +
                           (1 - y) * np.log(1 - y_pred + 1e-8))
            losses.append(loss)

            dw = X.T @ (y_pred - y) / n_samples
            db = np.mean(y_pred - y)

            w -= learning_rate * dw
            b -= learning_rate * db

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_logistic_regression_classification(self, seed):
        """Test logistic regression classification"""
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        y = (X[:, 0] + X[:, 1] > 0).astype(int).reshape(-1, 1)

        w = np.random.randn(2, 1) * 0.1
        b = 0.0
        learning_rate = 0.01
        epochs = 100

        def sigmoid(z):
            return 1 / (1 + np.exp(-np.clip(z, -500, 500)))

        for epoch in range(epochs):
            z = X @ w + b
            y_pred = sigmoid(z)
            dw = X.T @ (y_pred - y) / n_samples
            db = np.mean(y_pred - y)
            w -= learning_rate * dw
            b -= learning_rate * db

        # Check accuracy
        z_final = X @ w + b
        y_pred_final = sigmoid(z_final)
        y_pred_class = (y_pred_final > 0.5).astype(int)
        accuracy = np.mean(y_pred_class == y)

        assert accuracy > 0.7


class TestSVM:
    """Test Support Vector Machine"""

    def test_svm_convergence(self, seed):
        """Test that SVM converges"""
        n_samples = 100
        X_class0 = np.random.randn(n_samples // 2, 2) + np.array([2, 2])
        X_class1 = np.random.randn(n_samples // 2, 2) + np.array([-2, -2])
        X = np.vstack([X_class0, X_class1])
        y = np.hstack([np.ones(n_samples // 2), -np.ones(n_samples // 2)])

        w = np.random.randn(2) * 0.1
        b = 0.0
        learning_rate = 0.01
        C = 1.0
        epochs = 50

        losses = []
        for epoch in range(epochs):
            z = X @ w + b
            margins = 1 - y * z
            hinge_loss = np.maximum(0, margins)
            loss = np.mean(hinge_loss) + C * np.sum(w ** 2) / 2
            losses.append(loss)

            mask = margins > 0
            grad_w = -X[mask].T @ y[mask] / n_samples + C * w
            grad_b = -np.sum(y[mask]) / n_samples

            w -= learning_rate * grad_w
            b -= learning_rate * grad_b

        # Loss should decrease
        assert losses[-1] < losses[0]


class TestDecisionTree:
    """Test decision tree"""

    def test_decision_tree_gini(self, seed):
        """Test Gini coefficient calculation"""
        y = np.array([0, 0, 1, 1, 1])

        # Calculate Gini
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)

        # Gini should be between 0 and 0.5 for binary classification
        assert 0 <= gini <= 0.5

    def test_decision_tree_pure_node(self, seed):
        """Test Gini for pure node"""
        y = np.array([1, 1, 1, 1])

        # Calculate Gini
        classes, counts = np.unique(y, return_counts=True)
        probabilities = counts / len(y)
        gini = 1 - np.sum(probabilities ** 2)

        # Pure node should have Gini = 0
        assert gini == 0


class TestRandomForest:
    """Test random forest"""

    def test_bootstrap_sampling(self, seed):
        """Test bootstrap sampling"""
        n_samples = 100
        X = np.random.randn(n_samples, 2)

        # Bootstrap sample
        indices = np.random.choice(n_samples, n_samples, replace=True)
        X_boot = X[indices]

        # Should have same size
        assert X_boot.shape == X.shape

        # Should have some repeated samples
        unique_indices = len(np.unique(indices))
        assert unique_indices < n_samples
