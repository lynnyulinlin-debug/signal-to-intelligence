"""Tests for Chapter 4: Transformer"""
import numpy as np
import pytest


class TestSelfAttention:
    """Test self-attention mechanism"""

    def test_attention_output_shape(self, seed):
        """Test attention output shape"""
        seq_len = 10
        d_model = 64

        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        # Scaled dot-product attention
        scores = Q @ K.T / np.sqrt(d_model)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)
        output = attention_weights @ V

        assert output.shape == (seq_len, d_model)

    def test_attention_weights_sum_to_one(self, seed):
        """Test that attention weights sum to 1"""
        seq_len = 10
        d_model = 64

        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)

        scores = Q @ K.T / np.sqrt(d_model)
        attention_weights = np.exp(scores) / np.sum(np.exp(scores), axis=-1, keepdims=True)

        # Each row should sum to 1
        assert np.allclose(np.sum(attention_weights, axis=1), 1.0)

    def test_multihead_attention(self, seed):
        """Test multi-head attention"""
        seq_len = 10
        d_model = 64
        num_heads = 8
        d_k = d_model // num_heads

        Q = np.random.randn(seq_len, d_model)
        K = np.random.randn(seq_len, d_model)
        V = np.random.randn(seq_len, d_model)

        # Split into heads
        Q_heads = Q.reshape(seq_len, num_heads, d_k)
        K_heads = K.reshape(seq_len, num_heads, d_k)
        V_heads = V.reshape(seq_len, num_heads, d_k)

        assert Q_heads.shape == (seq_len, num_heads, d_k)
        assert K_heads.shape == (seq_len, num_heads, d_k)
        assert V_heads.shape == (seq_len, num_heads, d_k)


class TestTransformerBlock:
    """Test Transformer block"""

    def test_layer_norm_stability(self, seed):
        """Test layer normalization stability"""
        batch_size = 32
        seq_len = 10
        d_model = 64

        x = np.random.randn(batch_size, seq_len, d_model)

        # Layer normalization
        mean = np.mean(x, axis=-1, keepdims=True)
        std = np.std(x, axis=-1, keepdims=True)
        x_norm = (x - mean) / (std + 1e-6)

        # Check that normalized values have mean ~0 and std ~1
        assert np.allclose(np.mean(x_norm, axis=-1), 0, atol=1e-5)
        assert np.allclose(np.std(x_norm, axis=-1), 1, atol=1e-5)

    def test_feed_forward_network(self, seed):
        """Test feed-forward network in Transformer"""
        seq_len = 10
        d_model = 64
        d_ff = 256

        x = np.random.randn(seq_len, d_model)

        # Feed-forward: Linear -> ReLU -> Linear
        W1 = np.random.randn(d_model, d_ff)
        b1 = np.zeros(d_ff)
        W2 = np.random.randn(d_ff, d_model)
        b2 = np.zeros(d_model)

        hidden = np.maximum(0, x @ W1 + b1)  # ReLU
        output = hidden @ W2 + b2

        assert output.shape == (seq_len, d_model)
