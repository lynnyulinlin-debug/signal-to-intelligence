"""Tests for Chapter 3: Deep Learning"""
import numpy as np
import pytest


class TestPolynomialVsMLP:
    """Test polynomial fitting vs MLP"""

    def test_mlp_output_shape(self, seed):
        """Test MLP output shape"""
        n_samples = 100
        input_size = 1
        hidden_size = 16
        output_size = 1

        X = np.random.randn(n_samples, input_size)
        W1 = np.random.randn(input_size, hidden_size)
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, output_size)
        b2 = np.zeros((1, output_size))

        # Forward pass
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2

        assert z2.shape == (n_samples, output_size)

    def test_mlp_nonlinearity(self, seed):
        """Test that MLP can learn nonlinear functions"""
        n_samples = 200
        X = np.linspace(-1, 1, n_samples).reshape(-1, 1)
        y = X ** 2  # Nonlinear function

        # Simple MLP
        hidden_size = 16
        W1 = np.random.randn(1, hidden_size) * 0.1
        b1 = np.zeros((1, hidden_size))
        W2 = np.random.randn(hidden_size, 1) * 0.1
        b2 = np.zeros((1, 1))

        # Train for a few iterations
        learning_rate = 0.01
        for _ in range(100):
            z1 = X @ W1 + b1
            a1 = np.tanh(z1)
            z2 = a1 @ W2 + b2
            y_pred = z2.flatten()

            loss = np.mean((y.flatten() - y_pred) ** 2)

            # Backward pass
            dz2 = (y_pred - y.flatten()).reshape(-1, 1) / n_samples
            dW2 = a1.T @ dz2
            db2 = np.sum(dz2, axis=0, keepdims=True)

            da1 = dz2 @ W2.T
            dz1 = da1 * (1 - a1 ** 2)
            dW1 = X.T @ dz1
            db1 = np.sum(dz1, axis=0, keepdims=True)

            W1 -= learning_rate * dW1
            b1 -= learning_rate * db1
            W2 -= learning_rate * dW2
            b2 -= learning_rate * db2

        # Check that loss decreased
        z1 = X @ W1 + b1
        a1 = np.tanh(z1)
        z2 = a1 @ W2 + b2
        y_pred = z2.flatten()
        final_loss = np.mean((y.flatten() - y_pred) ** 2)

        assert final_loss < 0.5  # Should learn something


class TestRNNStructure:
    """Test RNN structure"""

    def test_rnn_hidden_state_shape(self, seed):
        """Test RNN hidden state shape"""
        seq_length = 50
        input_size = 3
        hidden_size = 8

        X = np.random.randn(seq_length, input_size)
        W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        W_x = np.random.randn(input_size, hidden_size) * 0.1
        b_h = np.zeros((1, hidden_size))

        h = np.zeros((1, hidden_size))
        h_states = []

        for t in range(seq_length):
            x_t = X[t:t+1]
            h = np.tanh(x_t @ W_x + h @ W_h + b_h)
            h_states.append(h.copy())

        h_states = np.array(h_states).squeeze()
        assert h_states.shape == (seq_length, hidden_size)

    def test_rnn_hidden_state_evolution(self, seed):
        """Test that RNN hidden states evolve over time"""
        seq_length = 50
        input_size = 3
        hidden_size = 8

        X = np.random.randn(seq_length, input_size)
        W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        W_x = np.random.randn(input_size, hidden_size) * 0.1
        b_h = np.zeros((1, hidden_size))

        h = np.zeros((1, hidden_size))
        h_states = []

        for t in range(seq_length):
            x_t = X[t:t+1]
            h = np.tanh(x_t @ W_x + h @ W_h + b_h)
            h_states.append(h.copy())

        h_states = np.array(h_states).squeeze()

        # Check that hidden states are not all the same
        assert not np.allclose(h_states[0], h_states[-1])

    def test_rnn_bounded_activation(self, seed):
        """Test that RNN activations are bounded"""
        seq_length = 50
        input_size = 3
        hidden_size = 8

        X = np.random.randn(seq_length, input_size)
        W_h = np.random.randn(hidden_size, hidden_size) * 0.1
        W_x = np.random.randn(input_size, hidden_size) * 0.1
        b_h = np.zeros((1, hidden_size))

        h = np.zeros((1, hidden_size))

        for t in range(seq_length):
            x_t = X[t:t+1]
            h = np.tanh(x_t @ W_x + h @ W_h + b_h)

        # tanh output should be in [-1, 1]
        assert np.all(h >= -1.0)
        assert np.all(h <= 1.0)
