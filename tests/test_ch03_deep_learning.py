"""Tests for Chapter 3: Deep Learning"""
import numpy as np


class TestPolynomialVsMLP:
    """Test polynomial fitting vs MLP"""

    def test_mlp_output_shape(self, load_code_module):
        """Test MLP output shape"""
        poly = load_code_module("code/ch03_deep_learning_fast/polynomial_vs_mlp.py")
        n_samples = 100
        input_size = 1
        hidden_size = 16
        output_size = 1

        X = np.random.randn(n_samples, input_size)
        model = poly.SimpleNN(input_size, hidden_size, output_size, seed=42)
        output = model.forward(X)

        assert output.shape == (n_samples, output_size)

    def test_mlp_nonlinearity(self, load_code_module):
        """Test that MLP can learn nonlinear functions"""
        poly = load_code_module("code/ch03_deep_learning_fast/polynomial_vs_mlp.py")
        n_samples = 200
        X = np.linspace(-1, 1, n_samples).reshape(-1, 1)
        y = X ** 2  # Nonlinear function

        model = poly.SimpleNN(
            input_dim=1,
            hidden_dim=16,
            output_dim=1,
            learning_rate=0.01,
            seed=42,
        )
        losses = model.train(X, y, epochs=100)

        assert losses[-1] < losses[0]
        assert losses[-1] < 0.5  # Should learn something


class TestCNNStructure:
    """Test CNN structure"""

    def test_cnn_output_shape(self, load_code_module):
        cnn = load_code_module("code/ch03_deep_learning_fast/mnist_cnn.py")
        model = cnn.SimpleCNN(num_filters=16, kernel_size=3, seed=42)
        X = np.random.randn(8, 28, 28)
        output = model.forward(X)

        assert output.shape == (8, 10)

    def test_cnn_run_experiment(self, load_code_module):
        cnn = load_code_module("code/ch03_deep_learning_fast/mnist_cnn.py")
        result = cnn.run_experiment(
            seed=42,
            train_samples=64,
            test_samples=16,
            epochs=2,
            batch_size=16,
        )

        assert result["X_train"].shape == (64, 28, 28)
        assert result["X_test"].shape == (16, 28, 28)
        assert len(result["losses"]) == result["epochs"]
        assert np.isfinite(result["train_acc"])
        assert np.isfinite(result["test_acc"])


class TestRNNStructure:
    """Test RNN structure"""

    def test_rnn_hidden_state_shape(self, load_code_module):
        """Test RNN hidden state shape"""
        rnn = load_code_module("code/ch03_deep_learning_fast/rnn_structure.py")
        seq_length = 50
        input_size = 3
        hidden_size = 8

        X = np.random.randn(seq_length, input_size)
        weights = rnn.initialize_rnn_weights(input_size, hidden_size, seed=42)
        _, h_states = rnn.run_rnn_forward(X, weights)

        assert h_states.shape == (seq_length, hidden_size)

    def test_rnn_hidden_state_evolution(self, load_code_module):
        """Test that RNN hidden states evolve over time"""
        rnn = load_code_module("code/ch03_deep_learning_fast/rnn_structure.py")
        seq_length = 50
        input_size = 3
        hidden_size = 8

        X = np.random.randn(seq_length, input_size)
        weights = rnn.initialize_rnn_weights(input_size, hidden_size, seed=42)
        _, h_states = rnn.run_rnn_forward(X, weights)

        # Check that hidden states are not all the same
        assert not np.allclose(h_states[0], h_states[-1])

    def test_rnn_bounded_activation(self, load_code_module):
        """Test that RNN activations are bounded"""
        rnn = load_code_module("code/ch03_deep_learning_fast/rnn_structure.py")
        seq_length = 50
        input_size = 3
        hidden_size = 8

        X = np.random.randn(seq_length, input_size)
        weights = rnn.initialize_rnn_weights(input_size, hidden_size, seed=42)
        _, h_states = rnn.run_rnn_forward(X, weights)

        # tanh output should be in [-1, 1]
        assert np.all(h_states >= -1.0)
        assert np.all(h_states <= 1.0)


class TestSequenceModels1DSignal:
    """Test sequence models on 1D signals"""

    def test_sequence_model_shapes_and_values(self, load_code_module):
        sequence_models = load_code_module(
            "code/ch03_deep_learning_fast/sequence_models_1d_signal.py"
        )
        result = sequence_models.run_experiment(seed=42)

        assert result["timesteps"] == 200
        assert result["signal"].shape == (200,)
        assert result["cnn_edge"].shape == (200,)
        assert result["cnn_smooth"].shape == (200,)
        assert result["rnn_state"].shape == (200,)
        assert result["attention_summary"].shape == (200,)
        assert result["transformer_like"].shape == (200,)
        assert np.isfinite(result["global_context"])
        assert np.isfinite(result["signal"]).all()
        assert np.isfinite(result["cnn_edge"]).all()
        assert np.isfinite(result["cnn_smooth"]).all()
        assert np.isfinite(result["rnn_state"]).all()
        assert np.isfinite(result["attention_summary"]).all()
        assert np.isfinite(result["transformer_like"]).all()
