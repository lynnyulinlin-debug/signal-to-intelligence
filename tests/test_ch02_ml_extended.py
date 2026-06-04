"""Tests for Chapter 2: Optimization and Machine Learning"""
import numpy as np


class TestLinearRegression:
    """Test linear regression"""

    def test_linear_regression_convergence(self, load_code_module):
        """Test that linear regression converges"""
        model = load_code_module("code/ch02_optimization/linear_logistic_regression.py")
        X, y = model.generate_linear_regression_data(n_samples=100, noise_std=0.1, seed=42)
        _, _, losses = model.train_linear_regression(X, y, learning_rate=0.01, epochs=50)

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_linear_regression_parameter_estimation(self, load_code_module):
        """Test that linear regression estimates parameters correctly"""
        model = load_code_module("code/ch02_optimization/linear_logistic_regression.py")
        w_true = 2.0
        b_true = 1.0
        X, y = model.generate_linear_regression_data(n_samples=200, noise_std=0.0, seed=42)
        w, b, _ = model.train_linear_regression(X, y, learning_rate=0.01, epochs=100)

        # Check if estimated parameters are close to true parameters
        assert np.abs(w[0, 0] - w_true) < 0.5
        assert np.abs(b - b_true) < 0.5


class TestMMSEvsNN:
    """Test MMSE vs neural network estimation"""

    def test_mmse_run_experiment(self, load_code_module):
        mmse_nn = load_code_module("code/ch02_optimization/mmse_vs_nn.py")
        result = mmse_nn.run_experiment(seed=42)

        assert result["s_true"].shape == (500,)
        assert result["s_mmse"].shape == (500,)
        assert result["s_nn"].shape == (500,)
        assert result["losses_nn"]
        assert np.isfinite(result["mse_mmse"])
        assert np.isfinite(result["mse_nn"])
        assert np.isfinite(result["snr_out_mmse"])
        assert np.isfinite(result["snr_out_nn"])


class TestLogisticRegression:
    """Test logistic regression"""

    def test_logistic_regression_convergence(self, load_code_module):
        """Test that logistic regression converges"""
        model = load_code_module("code/ch02_optimization/linear_logistic_regression.py")
        X, y = model.generate_logistic_regression_data(n_samples=100, seed=42)
        _, _, losses = model.train_logistic_regression(X, y, learning_rate=0.01, epochs=50)

        # Loss should decrease
        assert losses[-1] < losses[0]

    def test_logistic_regression_classification(self, load_code_module):
        """Test logistic regression classification"""
        model = load_code_module("code/ch02_optimization/linear_logistic_regression.py")
        X, y = model.generate_logistic_regression_data(n_samples=100, seed=42)
        w, b, _ = model.train_logistic_regression(X, y, learning_rate=0.01, epochs=100)
        accuracy = model.classification_accuracy(X, y, w, b)

        assert accuracy > 0.7


class TestSVM:
    """Test Support Vector Machine"""

    def test_svm_convergence(self, load_code_module):
        """Test that SVM converges"""
        svm = load_code_module("code/ch02_optimization/svm_kernel.py")
        X, y = svm.generate_linear_svm_data(n_samples=100, seed=42)
        _, _, losses = svm.train_linear_svm(X, y, learning_rate=0.01, c=1.0, epochs=50)

        # Loss should decrease
        assert losses[-1] < losses[0]


class TestDecisionTree:
    """Test decision tree"""

    def test_decision_tree_gini(self, load_code_module):
        """Test Gini coefficient calculation"""
        tree_module = load_code_module("code/ch02_optimization/decision_tree_random_forest.py")
        y = np.array([0, 0, 1, 1, 1])
        gini = tree_module.SimpleDecisionTree()._gini(y)

        # Gini should be between 0 and 0.5 for binary classification
        assert 0 <= gini <= 0.5

    def test_decision_tree_pure_node(self, load_code_module):
        """Test Gini for pure node"""
        tree_module = load_code_module("code/ch02_optimization/decision_tree_random_forest.py")
        y = np.array([1, 1, 1, 1])
        gini = tree_module.SimpleDecisionTree()._gini(y)

        # Pure node should have Gini = 0
        assert gini == 0


class TestRandomForest:
    """Test random forest"""

    def test_bootstrap_sampling(self, load_code_module):
        """Test bootstrap sampling"""
        tree_module = load_code_module("code/ch02_optimization/decision_tree_random_forest.py")
        n_samples = 100
        X = np.random.randn(n_samples, 2)
        X_boot, indices = tree_module.bootstrap_sample(X, seed=42)

        # Should have same size
        assert X_boot.shape == X.shape

        # Should have some repeated samples
        unique_indices = len(np.unique(indices))
        assert unique_indices < n_samples
