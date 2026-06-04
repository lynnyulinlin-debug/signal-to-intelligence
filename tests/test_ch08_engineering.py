"""Tests for Chapter 8: LLM Engineering."""


class TestModelSelection:
    """Test model selection utilities."""

    def test_model_selector_filters_by_latency_and_cost(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        selector = engineering.ModelSelector()

        candidates = selector.select_by_criteria(
            max_latency_ms=400,
            max_cost_per_1m_tokens=10,
            min_context_window=16_000,
        )

        candidate_names = {model.name for model in candidates}
        assert "GPT-3.5 Turbo" in candidate_names
        assert "Claude 3 Sonnet" in candidate_names
        assert "GPT-4" not in candidate_names

    def test_model_selector_requires_finetuning(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        selector = engineering.ModelSelector()

        candidates = selector.select_by_criteria(requires_finetuning=True)

        assert candidates
        assert all(model.supports_finetuning for model in candidates)


class TestCostOptimization:
    """Test cost analysis utilities."""

    def test_cost_estimation_scales_with_requests(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        analyzer = engineering.CostAnalyzer(engineering.ModelSelector())

        small = analyzer.estimate_cost("gpt-3.5-turbo", 100, 1_000, 200)
        large = analyzer.estimate_cost("gpt-3.5-turbo", 200, 1_000, 200)

        assert small.daily_cost > 0
        assert large.daily_cost == 2 * small.daily_cost
        assert small.monthly_cost == small.daily_cost * 30
        assert small.yearly_cost == small.daily_cost * 365

    def test_unknown_model_raises_error(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        analyzer = engineering.CostAnalyzer(engineering.ModelSelector())

        try:
            analyzer.estimate_cost("missing-model", 100, 1_000, 200)
        except ValueError as exc:
            assert "Unknown model" in str(exc)
        else:
            raise AssertionError("Expected ValueError for unknown model")


class TestRetryStrategy:
    """Test retry strategy."""

    def test_exponential_backoff_is_capped(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        strategy = engineering.RetryStrategy(initial_delay_ms=100, max_delay_ms=500)

        assert strategy.get_delay_ms(0) == 100
        assert strategy.get_delay_ms(1) == 200
        assert strategy.get_delay_ms(10) == 500

    def test_retryable_error_codes(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        strategy = engineering.RetryStrategy()

        assert strategy.should_retry(429)
        assert strategy.should_retry(503)
        assert not strategy.should_retry(400)


class TestMonitoring:
    """Test monitoring utilities."""

    def test_metrics_collector_aggregates_requests(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        collector = engineering.MetricsCollector()

        collector.record_request(latency_ms=100, tokens=50)
        collector.record_request(latency_ms=300, tokens=150, error=True)
        metrics = collector.get_current_metrics()

        assert metrics.avg_latency_ms == 200
        assert metrics.error_rate == 0.5
        assert metrics.tokens_per_request == 100

    def test_metrics_collector_alerts(self, load_code_module):
        engineering = load_code_module("code/ch08_llm_engineering/llm_engineering_demo.py")
        collector = engineering.MetricsCollector()
        metrics = engineering.MetricSnapshot(
            timestamp=0,
            requests_per_sec=120,
            avg_latency_ms=1500,
            error_rate=0.1,
            cost_per_hour=0.0,
            tokens_per_request=100,
        )

        alerts = collector.check_alerts(metrics)

        assert "HIGH_ERROR_RATE: 10.0%" in alerts
        assert "HIGH_LATENCY: 1500ms" in alerts
        assert "HIGH_THROUGHPUT: 120.0 req/s" in alerts
