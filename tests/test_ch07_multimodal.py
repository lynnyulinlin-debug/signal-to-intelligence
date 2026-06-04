"""Tests for Chapter 7: Multimodal LLM"""
import numpy as np


class TestVisionTransformer:
    """Test Vision Transformer (ViT)"""

    def test_image_patch_division(self, seed, load_code_module):
        """Test image patch division"""
        vit = load_code_module("code/ch07_multimodal_llm/vit_patches.py")
        image_size = 224
        patch_size = 16

        image = vit.create_gradient_image(image_size)
        patches = vit.create_patches(image, patch_size)

        assert patches.shape == (196, 16 * 16 * 3)

    def test_patch_embedding(self, seed, load_code_module):
        """Test patch embedding"""
        vit = load_code_module("code/ch07_multimodal_llm/vit_patches.py")
        n_patches = 196
        patch_dim = 16 * 16 * 3  # 16x16 RGB patches
        embedding_dim = 768

        patches = np.random.randn(n_patches, patch_dim)
        embeddings = vit.patch_embedding(patches, embedding_dim, seed=42)

        assert embeddings.shape == (n_patches, embedding_dim)


class TestCLIPModel:
    """Test CLIP model"""

    def test_image_text_alignment(self, seed, load_code_module):
        """Test image-text alignment"""
        clip = load_code_module("code/ch07_multimodal_llm/clip_similarity.py")
        batch_size = 32
        embedding_dim = 512

        image_embeddings = np.random.randn(batch_size, embedding_dim)
        text_embeddings = np.random.randn(batch_size, embedding_dim)

        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings,
            axis=1,
            keepdims=True,
        )
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        similarity = clip.compute_similarity_matrix(image_embeddings, text_embeddings)

        assert similarity.shape == (batch_size, batch_size)
        assert np.all(similarity >= -1) and np.all(similarity <= 1)

    def test_contrastive_loss(self, seed, load_code_module):
        """Test contrastive loss"""
        clip = load_code_module("code/ch07_multimodal_llm/clip_similarity.py")
        batch_size = 32
        embedding_dim = 512
        temperature = 0.07

        image_embeddings = np.random.randn(batch_size, embedding_dim)
        text_embeddings = np.random.randn(batch_size, embedding_dim)

        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings,
            axis=1,
            keepdims=True,
        )
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)

        logits = clip.compute_similarity_matrix(image_embeddings, text_embeddings) / temperature

        # Loss should be positive
        assert np.all(logits > -100) and np.all(logits < 100)


class TestHighResolutionProcessing:
    """Test high-resolution image processing"""

    def test_patch_division_method(self, seed, load_code_module):
        """Test patch division method"""
        high_res = load_code_module("code/ch07_multimodal_llm/high_resolution_processing.py")
        image_size = 1024
        patch_size = 16

        image = high_res.create_demo_image(image_size, seed=42)
        patches, positions, grid = high_res.patch_division_method(image, patch_size, 224)

        assert len(patches) == 4096
        assert len(positions) == 4096
        assert grid == (64, 64)

    def test_dynamic_resolution(self, seed, load_code_module):
        """Test dynamic resolution"""
        high_res = load_code_module("code/ch07_multimodal_llm/high_resolution_processing.py")
        target_tokens = 576
        image = high_res.create_demo_image(1024, seed=42)

        image_resized, resolution, grid = high_res.dynamic_resolution_method(
            image, target_tokens=target_tokens
        )

        assert image_resized.size > 0
        assert resolution[0] > 0 and resolution[1] > 0
        assert grid[0] > 0 and grid[1] > 0


class TestChapter7Demos:
    """Test chapter 7 demo scripts"""

    def test_architecture_diagrams(self, load_code_module, tmp_path):
        demo = load_code_module("code/ch07_multimodal_llm/architecture_diagrams.py")
        result = demo.run_experiment()
        paths = demo.plot_results(
            result,
            output_vit_cnn=tmp_path / "vit_cnn.png",
            output_temperature=tmp_path / "temperature.png",
        )

        assert paths["vit_cnn_path"].exists()
        assert paths["temperature_path"].exists()

    def test_qwen_vl_analysis(self, load_code_module):
        demo = load_code_module("code/ch07_multimodal_llm/qwen_vl_analysis.py")
        result = demo.run_experiment()

        assert len(result["model_names"]) == 5
        assert result["resolutions"].shape == (5,)
        assert result["accuracies"].shape == (5,)
        assert result["efficiency"].shape == (5,)
        assert result["models_data"]["Qwen2.5-VL"]["chinese_optimized"] is True
        assert result["accuracies"][result["model_names"].index("Qwen2.5-VL")] > result["accuracies"][0]

    def test_case_studies_static_data(self, load_code_module, tmp_path):
        demo = load_code_module("code/ch07_multimodal_llm/case_studies.py")
        result = demo.run_experiment()
        chart_path = demo._save_performance_chart(
            datasets=result["datasets"],
            output_path=tmp_path / "case_studies.png",
        )

        assert len(result["datasets"]) == 3
        assert result["datasets"][2]["llava"] is None
        assert result["datasets"][0]["qwen"].shape == (3,)
        assert chart_path.exists()

    def test_multimodal_applications_static_data(self, load_code_module, tmp_path):
        demo = load_code_module("code/ch07_multimodal_llm/multimodal_applications.py")
        result = demo.run_experiment()
        chart_path = demo._save_performance_chart(
            datasets=result["datasets"],
            output_path=tmp_path / "multimodal_applications.png",
        )

        assert len(result["datasets"]) == 3
        assert result["datasets"][1]["ylabel"] == "Accuracy (%)"
        assert result["datasets"][0]["qwen"].shape == (3,)
        assert chart_path.exists()

    def test_explainer_diagrams(self, load_code_module, tmp_path):
        demo = load_code_module("code/ch07_multimodal_llm/explainer_diagrams.py")
        result = demo.run_experiment()
        outputs = demo.plot_results(
            result,
            output_arch=tmp_path / "architecture.png",
            output_fusion=tmp_path / "fusion.png",
            output_resolution=tmp_path / "resolution.png",
            output_tiling=tmp_path / "tiling.png",
        )

        assert len(result["outputs"]) == 4
        assert outputs["architecture"].exists()
        assert outputs["fusion"].exists()
        assert outputs["resolution"].exists()
        assert outputs["tiling"].exists()
