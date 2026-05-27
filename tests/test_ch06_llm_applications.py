"""Tests for Chapter 6: LLM Applications"""
import numpy as np
import pytest


class TestRAGSystem:
    """Test RAG (Retrieval-Augmented Generation) system"""

    def test_document_retrieval(self, seed):
        """Test document retrieval"""
        documents = [
            "Transformer is a neural network architecture",
            "Self-attention mechanism allows parallel processing",
            "BERT is a pre-trained language model",
        ]

        query = "What is Transformer?"

        # Simple similarity-based retrieval
        query_tokens = set(query.lower().split())
        scores = []
        for doc in documents:
            doc_tokens = set(doc.lower().split())
            similarity = len(query_tokens & doc_tokens) / len(query_tokens | doc_tokens)
            scores.append(similarity)

        # Should retrieve at least one document
        assert max(scores) > 0

    def test_context_augmentation(self, seed):
        """Test context augmentation"""
        query = "What is attention?"
        retrieved_docs = [
            "Attention mechanism computes weighted sum of values",
            "Self-attention allows each position to attend to all positions",
        ]

        # Augmented context
        augmented_context = query + "\n" + "\n".join(retrieved_docs)

        assert query in augmented_context
        assert all(doc in augmented_context for doc in retrieved_docs)


class TestFineTuning:
    """Test fine-tuning"""

    def test_fine_tuning_convergence(self, seed):
        """Test that fine-tuning converges"""
        n_samples = 50
        n_features = 10
        learning_rate = 0.01
        epochs = 20

        # Generate data
        X = np.random.randn(n_samples, n_features)
        y = np.random.randint(0, 2, n_samples)

        # Simple fine-tuning
        w = np.random.randn(n_features)
        losses = []

        for epoch in range(epochs):
            logits = X @ w
            predictions = (logits > 0).astype(int)
            loss = np.mean((predictions - y) ** 2)
            losses.append(loss)

            # Update weights
            grad = X.T @ (predictions - y) / n_samples
            w -= learning_rate * grad

        # Loss should decrease
        assert losses[-1] <= losses[0]


class TestAgentFramework:
    """Test agent framework"""

    def test_agent_decision_making(self, seed):
        """Test agent decision making"""
        tools = ["calculator", "search", "database"]

        # Agent should choose appropriate tool
        query = "What is 2 + 2?"

        # Simple tool selection
        if "+" in query or "-" in query or "*" in query or "/" in query:
            selected_tool = "calculator"
        else:
            selected_tool = "search"

        assert selected_tool in tools
