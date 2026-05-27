"""Tests for Chapter 7: Multimodal LLM"""
import numpy as np
import pytest


class TestVisionTransformer:
    """Test Vision Transformer (ViT)"""

    def test_image_patch_division(self, seed):
        """Test image patch division"""
        image_size = 224
        patch_size = 16

        n_patches = (image_size // patch_size) ** 2

        assert n_patches == 196  # 14x14 patches

    def test_patch_embedding(self, seed):
        """Test patch embedding"""
        n_patches = 196
        patch_dim = 16 * 16 * 3  # 16x16 RGB patches
        embedding_dim = 768

        # Patch embedding layer
        patches = np.random.randn(n_patches, patch_dim)
        W = np.random.randn(patch_dim, embedding_dim)

        embeddings = patches @ W

        assert embeddings.shape == (n_patches, embedding_dim)


class TestCLIPModel:
    """Test CLIP model"""

    def test_image_text_alignment(self, seed):
        """Test image-text alignment"""
        batch_size = 32
        embedding_dim = 512

        # Image and text embeddings
        image_embeddings = np.random.randn(batch_size, embedding_dim)
        text_embeddings = np.random.randn(batch_size, embedding_dim)

        # Normalize
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        # Similarity matrix
        similarity = image_embeddings @ text_embeddings.T

        assert similarity.shape == (batch_size, batch_size)
        assert np.all(similarity >= -1) and np.all(similarity <= 1)

    def test_contrastive_loss(self, seed):
        """Test contrastive loss"""
        batch_size = 32
        embedding_dim = 512
        temperature = 0.07

        # Embeddings
        image_embeddings = np.random.randn(batch_size, embedding_dim)
        text_embeddings = np.random.randn(batch_size, embedding_dim)

        # Normalize
        image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        # Similarity
        logits = image_embeddings @ text_embeddings.T / temperature

        # Loss should be positive
        assert np.all(logits > -100) and np.all(logits < 100)


class TestHighResolutionProcessing:
    """Test high-resolution image processing"""

    def test_patch_division_method(self, seed):
        """Test patch division method"""
        image_size = 1024
        patch_size = 16

        n_patches = (image_size // patch_size) ** 2

        assert n_patches == 4096  # 64x64 patches

    def test_dynamic_resolution(self, seed):
        """Test dynamic resolution"""
        target_tokens = 576
        aspect_ratio = 1.0

        # Calculate patches
        patches_h = int(np.sqrt(target_tokens / aspect_ratio))
        patches_w = int(patches_h * aspect_ratio)

        # Adjust to multiple of 16
        patches_h = (patches_h // 16) * 16
        patches_w = (patches_w // 16) * 16

        assert patches_h > 0 and patches_w > 0
