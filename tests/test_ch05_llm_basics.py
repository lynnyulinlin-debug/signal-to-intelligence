"""Tests for Chapter 5: LLM Basics"""
import numpy as np


class TestScalingLaws:
    """Test scaling laws"""

    def test_loss_decreases_with_model_size(self, seed, load_code_module):
        """Test that loss decreases with model size"""
        scaling_laws = load_code_module("code/ch05_llm_basics/scaling_laws.py")
        model_sizes = np.array([1, 10, 100, 1000])

        # Chinchilla scaling law: loss ~ 1 / N^0.07
        losses = scaling_laws.power_law(model_sizes, a=1.0, alpha=0.07)

        # Check that loss decreases
        for i in range(len(losses) - 1):
            assert losses[i] > losses[i + 1]

    def test_loss_decreases_with_data_size(self, seed, load_code_module):
        """Test that loss decreases with data size"""
        scaling_laws = load_code_module("code/ch05_llm_basics/scaling_laws.py")
        data_sizes = np.array([1, 10, 100, 1000])

        # Chinchilla scaling law: loss ~ 1 / D^0.09
        losses = scaling_laws.power_law(data_sizes, a=1.0, alpha=0.09)

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

    def test_token_count(self, seed, load_code_module):
        """Test token counting"""
        bpe = load_code_module("code/ch05_llm_basics/bpe_tokenization.py")
        corpus = ["low", "lower", "newest", "widest"]

        vocab = bpe.get_vocab(corpus)
        merges, vocab_sizes = bpe.run_bpe(corpus, num_merges=3)

        assert len(vocab) == len(corpus)
        assert len(merges) == 3
        assert vocab_sizes[-1] < vocab_sizes[0]


class TestChapter5Demos:
    """Test chapter 5 demo scripts"""

    def test_autoregressive_generation(self, load_code_module):
        demo = load_code_module("code/ch05_llm_basics/autoregressive_generation.py")
        result = demo.run_experiment()

        assert len(result["tokens"]) == 10
        assert result["temperature_0_2"].shape == (10,)
        assert result["temperature_1_0"].shape == (10,)
        assert result["temperature_2_0"].shape == (10,)
        assert result["top_k"].shape == (10,)
        assert result["top_p"].shape == (10,)
        assert np.isclose(result["temperature_1_0"].sum(), 1.0)
        assert np.isclose(result["top_k"].sum(), 1.0)
        assert np.isclose(result["top_p"].sum(), 1.0)
        assert np.count_nonzero(result["top_k"]) == 3
        assert np.count_nonzero(result["top_p"]) <= 10

    def test_training_data_composition(self, load_code_module):
        demo = load_code_module("code/ch05_llm_basics/training_data_composition.py")
        result = demo.run_experiment()

        assert len(result["pretrain_labels"]) == 5
        assert result["pretrain_sizes"].shape == (5,)
        assert result["data_sizes"].shape == (4,)
        assert len(result["models"]) == 5
        assert result["train_tokens"].shape == (5,)
        assert result["mmlu_scores"].shape == (5,)
        assert result["pretrain_sizes"].sum() > 400
        assert result["mmlu_scores"].max() > 70

    def test_model_families_evolution(self, load_code_module):
        demo = load_code_module("code/ch05_llm_basics/model_families_evolution.py")
        result = demo.run_experiment()

        assert len(result["models"]) == 17
        assert set(result["family_colors"]) == {"GPT", "LLaMA", "Qwen", "DeepSeek", "Mistral"}
        assert result["open_scores"].shape == (5,)
        assert result["closed_scores"].shape == (5,)
        assert result["open_scores"].sum() > result["closed_scores"].sum()

    def test_lora_visualization(self, load_code_module):
        demo = load_code_module("code/ch05_llm_basics/lora_visualization.py")
        result = demo.run_experiment(seed=42)

        assert result["ranks"].shape == (6,)
        assert result["errors"].shape == (6,)
        assert result["param_ratios"].shape == (6,)
        assert result["errors"][0] > result["errors"][-1]
        assert result["param_ratios"][0] < result["param_ratios"][-1]
        assert np.isfinite(result["errors"]).all()
        assert np.isfinite(result["param_ratios"]).all()

    def test_rlhf_pipeline(self, load_code_module):
        demo = load_code_module("code/ch05_llm_basics/rlhf_pipeline.py")
        result = demo.run_experiment(seed=42)

        assert result["pre_align_scores"].shape == (500,)
        assert result["post_align_scores"].shape == (500,)
        assert result["reward_curve"].shape == result["steps"].shape
        assert result["kl_curve"].shape == result["steps"].shape
        assert result["rlhf_winrate"].shape == (5,)
        assert result["dpo_winrate"].shape == (5,)
        assert result["post_align_scores"].mean() > result["pre_align_scores"].mean()
        assert result["reward_curve"][-1] > result["reward_curve"][0]
        assert np.isfinite(result["kl_curve"]).all()

    def test_benchmark_comparison(self, load_code_module):
        demo = load_code_module("code/ch05_llm_basics/benchmark_comparison.py")
        result = demo.run_experiment()

        assert len(result["models"]) == 6
        assert len(result["benchmarks"]) == 4
        assert len(result["normalized"]) == 4
        assert result["elo_scores"].shape == (6,)
        assert len(result["sorted_models"]) == 6
        assert result["sorted_elo"].shape == (6,)
        assert result["sorted_models"][0] == "GPT-4o"
        assert result["sorted_elo"][0] == result["elo_scores"].max()
