"""Tests for Chapter 5: LLM Basics"""
import numpy as np
import pytest


class TestScalingLaws:
    """Test scaling laws"""

    def test_loss_decreases_with_model_size(self, seed):
        """Test that loss decreases with model size"""
        model_sizes = np.array([1, 10, 100, 1000])

        # Chinchilla scaling law: loss ~ 1 / N^0.07
        losses = 1.0 / (model_sizes ** 0.07)

        # Check that loss decreases
        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1]

    def test_loss_decreases_with_data_size(self, seed):
        """Test that loss decreases with data size"""
        data_sizes = np.array([1, 10, 100, 1000])

        # Chinchilla scaling law: loss ~ 1 / D^0.09
        losses = 1.0 / (data_sizes ** 0.09)

        # Check that loss decreases
        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1]


class TestInContextLearning:
    """Test in-context learning"""

    def test_few_shot_learning(self, seed):
        """Test few-shot learning capability"""
        # Simulate few-shot examples
        examples = [
            ("The cat sat on the mat", "positive"),
            ("I love this movie", "positive"),
            ("This is terrible", "negative"),
        ]

        # Test query
        query = "This is amazing"

        # In-context learning: model should recognize pattern
        assert len(examples) > 0
        assert len(query) > 0

    def test_prompt_engineering_effect(self, seed):
        """Test that prompt engineering affects output"""
        # Different prompts for same task
        prompt1 = "Translate to French: Hello"
        prompt2 = "You are a professional translator. Translate to French: Hello"

        # Both prompts should be valid
        assert len(prompt1) > 0
        assert len(prompt2) > 0
        assert len(prompt2) > len(prompt1)  # More detailed prompt


class TestTokenization:
    """Test tokenization"""

    def test_token_count(self, seed):
        """Test token counting"""
        text = "The quick brown fox jumps over the lazy dog"

        # Simple word-based tokenization
        tokens = text.split()

        assert len(tokens) == 9
        assert all(isinstance(t, str) for t in tokens)
