#!/usr/bin/env python3
"""
Chapter 8: LLM Engineering Practices
Demonstrates model selection, deployment, cost optimization, and monitoring.

Key concepts:
- Model selection criteria (performance, cost, availability)
- Deployment options (API vs self-hosted)
- Cost analysis and optimization
- Error handling and retry mechanisms
- Monitoring and alerting
"""

import time
import json
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from enum import Enum


# ============================================================================
# 1. Model Selection and Comparison
# ============================================================================

class DeploymentMode(Enum):
    """Deployment options for LLM applications."""
    API = "api"
    SELF_HOSTED = "self_hosted"
    HYBRID = "hybrid"


@dataclass
class ModelSpec:
    """Specification of an LLM model."""
    name: str
    provider: str
    input_cost_per_1m_tokens: float  # Cost in USD
    output_cost_per_1m_tokens: float
    latency_ms: float  # Average latency in milliseconds
    throughput_tokens_per_sec: float
    context_window: int
    supports_finetuning: bool
    deployment_mode: DeploymentMode


class ModelSelector:
    """Select the best model based on requirements."""

    def __init__(self):
        self.models = {
            "gpt-4": ModelSpec(
                name="GPT-4",
                provider="OpenAI",
                input_cost_per_1m_tokens=30.0,
                output_cost_per_1m_tokens=60.0,
                latency_ms=500,
                throughput_tokens_per_sec=20,
                context_window=128000,
                supports_finetuning=False,
                deployment_mode=DeploymentMode.API
            ),
            "gpt-3.5-turbo": ModelSpec(
                name="GPT-3.5 Turbo",
                provider="OpenAI",
                input_cost_per_1m_tokens=0.5,
                output_cost_per_1m_tokens=1.5,
                latency_ms=200,
                throughput_tokens_per_sec=100,
                context_window=16384,
                supports_finetuning=True,
                deployment_mode=DeploymentMode.API
            ),
            "claude-3-opus": ModelSpec(
                name="Claude 3 Opus",
                provider="Anthropic",
                input_cost_per_1m_tokens=15.0,
                output_cost_per_1m_tokens=75.0,
                latency_ms=400,
                throughput_tokens_per_sec=30,
                context_window=200000,
                supports_finetuning=False,
                deployment_mode=DeploymentMode.API
            ),
            "claude-3-sonnet": ModelSpec(
                name="Claude 3 Sonnet",
                provider="Anthropic",
                input_cost_per_1m_tokens=3.0,
                output_cost_per_1m_tokens=15.0,
                latency_ms=300,
                throughput_tokens_per_sec=50,
                context_window=200000,
                supports_finetuning=False,
                deployment_mode=DeploymentMode.API
            ),
            "llama-2-70b": ModelSpec(
                name="Llama 2 70B",
                provider="Meta",
                input_cost_per_1m_tokens=0.0,  # Self-hosted, no API cost
                output_cost_per_1m_tokens=0.0,
                latency_ms=800,
                throughput_tokens_per_sec=10,
                context_window=4096,
                supports_finetuning=True,
                deployment_mode=DeploymentMode.SELF_HOSTED
            ),
        }

    def select_by_criteria(
        self,
        max_latency_ms: float = float('inf'),
        max_cost_per_1m_tokens: float = float('inf'),
        min_context_window: int = 0,
        requires_finetuning: bool = False
    ) -> List[ModelSpec]:
        """Select models matching criteria."""
        candidates = []
        for model in self.models.values():
            if (model.latency_ms <= max_latency_ms and
                (model.input_cost_per_1m_tokens + model.output_cost_per_1m_tokens) / 2 <= max_cost_per_1m_tokens and
                model.context_window >= min_context_window and
                (not requires_finetuning or model.supports_finetuning)):
                candidates.append(model)
        return sorted(candidates, key=lambda m: m.latency_ms)

    def compare_models(self, model_names: List[str]) -> Dict:
        """Compare multiple models."""
        comparison = {}
        for name in model_names:
            if name in self.models:
                model = self.models[name]
                comparison[name] = {
                    "name": model.name,
                    "provider": model.provider,
                    "input_cost": f"${model.input_cost_per_1m_tokens}/1M tokens",
                    "output_cost": f"${model.output_cost_per_1m_tokens}/1M tokens",
                    "latency": f"{model.latency_ms}ms",
                    "throughput": f"{model.throughput_tokens_per_sec} tokens/sec",
                    "context_window": f"{model.context_window:,} tokens",
                    "finetuning": "Yes" if model.supports_finetuning else "No",
                    "deployment": model.deployment_mode.value
                }
        return comparison


# ============================================================================
# 2. Cost Analysis and Optimization
# ============================================================================

@dataclass
class CostEstimate:
    """Cost estimation for LLM usage."""
    daily_requests: int
    avg_input_tokens: int
    avg_output_tokens: int
    model_name: str
    daily_cost: float
    monthly_cost: float
    yearly_cost: float


class CostAnalyzer:
    """Analyze and optimize LLM costs."""

    def __init__(self, model_selector: ModelSelector):
        self.selector = model_selector

    def estimate_cost(
        self,
        model_name: str,
        daily_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int
    ) -> CostEstimate:
        """Estimate daily/monthly/yearly costs."""
        if model_name not in self.selector.models:
            raise ValueError(f"Unknown model: {model_name}")

        model = self.selector.models[model_name]

        # Calculate daily cost
        daily_input_cost = (daily_requests * avg_input_tokens *
                           model.input_cost_per_1m_tokens / 1_000_000)
        daily_output_cost = (daily_requests * avg_output_tokens *
                            model.output_cost_per_1m_tokens / 1_000_000)
        daily_cost = daily_input_cost + daily_output_cost

        return CostEstimate(
            daily_requests=daily_requests,
            avg_input_tokens=avg_input_tokens,
            avg_output_tokens=avg_output_tokens,
            model_name=model_name,
            daily_cost=daily_cost,
            monthly_cost=daily_cost * 30,
            yearly_cost=daily_cost * 365
        )

    def compare_costs(
        self,
        model_names: List[str],
        daily_requests: int,
        avg_input_tokens: int,
        avg_output_tokens: int
    ) -> Dict[str, Dict]:
        """Compare costs across models."""
        comparison = {}
        for model_name in model_names:
            estimate = self.estimate_cost(
                model_name, daily_requests, avg_input_tokens, avg_output_tokens
            )
            comparison[model_name] = {
                "daily": f"${estimate.daily_cost:.2f}",
                "monthly": f"${estimate.monthly_cost:.2f}",
                "yearly": f"${estimate.yearly_cost:.2f}"
            }
        return comparison

    def optimization_strategies(self) -> Dict[str, str]:
        """List cost optimization strategies."""
        return {
            "1_cheaper_model": "Use cheaper models (GPT-3.5 instead of GPT-4)",
            "2_prompt_optimization": "Reduce prompt size (fewer examples, shorter context)",
            "3_caching": "Cache repeated queries and responses",
            "4_batching": "Process multiple requests in batch",
            "5_finetuning": "Fine-tune cheaper model instead of using expensive one",
            "6_self_hosting": "Self-host open-source models for high volume",
            "7_rate_limiting": "Implement rate limiting to control usage",
            "8_monitoring": "Monitor and alert on cost anomalies"
        }


# ============================================================================
# 3. Error Handling and Retry Mechanisms
# ============================================================================

class RetryStrategy:
    """Retry strategy with exponential backoff."""

    def __init__(
        self,
        max_retries: int = 3,
        initial_delay_ms: float = 100,
        max_delay_ms: float = 10000,
        exponential_base: float = 2.0
    ):
        self.max_retries = max_retries
        self.initial_delay_ms = initial_delay_ms
        self.max_delay_ms = max_delay_ms
        self.exponential_base = exponential_base

    def get_delay_ms(self, attempt: int) -> float:
        """Calculate delay for given attempt number."""
        delay = self.initial_delay_ms * (self.exponential_base ** attempt)
        return min(delay, self.max_delay_ms)

    def should_retry(self, error_code: int) -> bool:
        """Determine if error is retryable."""
        # Retryable errors: rate limit (429), server error (5xx), timeout
        retryable_codes = {429, 500, 502, 503, 504}
        return error_code in retryable_codes

    def execute_with_retry(self, func, *args, **kwargs):
        """Execute function with retry logic."""
        last_error = None

        for attempt in range(self.max_retries + 1):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < self.max_retries:
                    delay_ms = self.get_delay_ms(attempt)
                    print(f"Attempt {attempt + 1} failed: {e}")
                    print(f"Retrying in {delay_ms}ms...")
                    time.sleep(delay_ms / 1000)
                else:
                    print(f"All {self.max_retries + 1} attempts failed")

        raise last_error


# ============================================================================
# 4. Monitoring and Alerting
# ============================================================================

@dataclass
class MetricSnapshot:
    """Snapshot of system metrics."""
    timestamp: float
    requests_per_sec: float
    avg_latency_ms: float
    error_rate: float
    cost_per_hour: float
    tokens_per_request: float


class MetricsCollector:
    """Collect and track LLM application metrics."""

    def __init__(self):
        self.metrics: List[MetricSnapshot] = []
        self.request_count = 0
        self.error_count = 0
        self.total_latency_ms = 0
        self.total_tokens = 0
        self.start_time = time.time()

    def record_request(self, latency_ms: float, tokens: int, error: bool = False):
        """Record a single request."""
        self.request_count += 1
        self.total_latency_ms += latency_ms
        self.total_tokens += tokens
        if error:
            self.error_count += 1

    def get_current_metrics(self) -> MetricSnapshot:
        """Get current metrics snapshot."""
        elapsed_sec = time.time() - self.start_time

        return MetricSnapshot(
            timestamp=time.time(),
            requests_per_sec=self.request_count / elapsed_sec if elapsed_sec > 0 else 0,
            avg_latency_ms=self.total_latency_ms / self.request_count if self.request_count > 0 else 0,
            error_rate=self.error_count / self.request_count if self.request_count > 0 else 0,
            cost_per_hour=0.0,  # Would be calculated based on actual model
            tokens_per_request=self.total_tokens / self.request_count if self.request_count > 0 else 0
        )

    def check_alerts(self, metrics: MetricSnapshot) -> List[str]:
        """Check if any alerts should be triggered."""
        alerts = []

        if metrics.error_rate > 0.05:  # 5% error rate
            alerts.append(f"HIGH_ERROR_RATE: {metrics.error_rate:.1%}")

        if metrics.avg_latency_ms > 1000:  # 1 second
            alerts.append(f"HIGH_LATENCY: {metrics.avg_latency_ms:.0f}ms")

        if metrics.requests_per_sec > 100:  # Rate limit warning
            alerts.append(f"HIGH_THROUGHPUT: {metrics.requests_per_sec:.1f} req/s")

        return alerts


# ============================================================================
# 5. Demonstrations
# ============================================================================

def demo_model_selection():
    """Demonstrate model selection."""
    print("\n" + "="*70)
    print("DEMO 1: Model Selection")
    print("="*70)

    selector = ModelSelector()

    # Compare popular models
    print("\n1. Comparing popular models:")
    comparison = selector.compare_models([
        "gpt-4",
        "gpt-3.5-turbo",
        "claude-3-opus",
        "claude-3-sonnet",
        "llama-2-70b"
    ])

    for model_name, specs in comparison.items():
        print(f"\n{specs['name']} ({specs['provider']}):")
        for key, value in specs.items():
            if key not in ['name', 'provider']:
                print(f"  {key}: {value}")

    # Select by criteria
    print("\n2. Selecting models by criteria (latency < 400ms, cost < $10/1M tokens):")
    candidates = selector.select_by_criteria(
        max_latency_ms=400,
        max_cost_per_1m_tokens=10
    )
    for model in candidates:
        print(f"  ✓ {model.name}: {model.latency_ms}ms, ${model.input_cost_per_1m_tokens}/1M tokens")


def demo_cost_analysis():
    """Demonstrate cost analysis."""
    print("\n" + "="*70)
    print("DEMO 2: Cost Analysis and Optimization")
    print("="*70)

    selector = ModelSelector()
    analyzer = CostAnalyzer(selector)

    # Scenario: 10,000 daily requests
    print("\nScenario: 10,000 daily requests, 500 input tokens, 200 output tokens")

    cost_comparison = analyzer.compare_costs(
        ["gpt-4", "gpt-3.5-turbo", "claude-3-sonnet"],
        daily_requests=10000,
        avg_input_tokens=500,
        avg_output_tokens=200
    )

    print("\nCost comparison:")
    for model, costs in cost_comparison.items():
        print(f"  {model}:")
        print(f"    Daily:   {costs['daily']}")
        print(f"    Monthly: {costs['monthly']}")
        print(f"    Yearly:  {costs['yearly']}")

    # Optimization strategies
    print("\nCost optimization strategies:")
    strategies = analyzer.optimization_strategies()
    for key, strategy in strategies.items():
        print(f"  {key}: {strategy}")

    # Savings calculation
    print("\nPotential savings by switching from GPT-4 to GPT-3.5:")
    gpt4_estimate = analyzer.estimate_cost("gpt-4", 10000, 500, 200)
    gpt35_estimate = analyzer.estimate_cost("gpt-3.5-turbo", 10000, 500, 200)
    savings = gpt4_estimate.yearly_cost - gpt35_estimate.yearly_cost
    print(f"  Yearly savings: ${savings:,.2f} ({savings/gpt4_estimate.yearly_cost:.1%})")


def demo_error_handling():
    """Demonstrate error handling and retry."""
    print("\n" + "="*70)
    print("DEMO 3: Error Handling and Retry Mechanisms")
    print("="*70)

    strategy = RetryStrategy(max_retries=3, initial_delay_ms=100)

    print("\nRetry strategy configuration:")
    print(f"  Max retries: {strategy.max_retries}")
    print(f"  Initial delay: {strategy.initial_delay_ms}ms")
    print(f"  Max delay: {strategy.max_delay_ms}ms")
    print(f"  Exponential base: {strategy.exponential_base}")

    print("\nRetry delays for each attempt:")
    for attempt in range(strategy.max_retries + 1):
        delay = strategy.get_delay_ms(attempt)
        print(f"  Attempt {attempt + 1}: {delay:.0f}ms")

    print("\nRetryable error codes:")
    test_codes = [429, 500, 502, 503, 504, 400, 401, 404]
    for code in test_codes:
        retryable = strategy.should_retry(code)
        status = "✓ Retryable" if retryable else "✗ Not retryable"
        print(f"  {code}: {status}")

    # Simulate retry execution
    print("\nSimulating function execution with retries:")
    attempt_count = [0]

    def flaky_function():
        attempt_count[0] += 1
        if attempt_count[0] < 3:
            raise Exception(f"Simulated error on attempt {attempt_count[0]}")
        return "Success!"

    try:
        result = strategy.execute_with_retry(flaky_function)
        print(f"  Result: {result}")
    except Exception as e:
        print(f"  Failed: {e}")


def demo_monitoring():
    """Demonstrate monitoring and alerting."""
    print("\n" + "="*70)
    print("DEMO 4: Monitoring and Alerting")
    print("="*70)

    collector = MetricsCollector()

    # Simulate requests
    print("\nSimulating 100 requests...")
    import random

    for i in range(100):
        latency = random.gauss(300, 50)  # Normal distribution
        tokens = random.randint(100, 500)
        error = random.random() < 0.02  # 2% error rate
        collector.record_request(latency, tokens, error)

    # Get metrics
    metrics = collector.get_current_metrics()

    print(f"\nMetrics snapshot:")
    print(f"  Requests: {collector.request_count}")
    print(f"  Requests/sec: {metrics.requests_per_sec:.2f}")
    print(f"  Avg latency: {metrics.avg_latency_ms:.1f}ms")
    print(f"  Error rate: {metrics.error_rate:.1%}")
    print(f"  Avg tokens/request: {metrics.tokens_per_request:.0f}")

    # Check alerts
    alerts = collector.check_alerts(metrics)
    if alerts:
        print(f"\nAlerts triggered:")
        for alert in alerts:
            print(f"  ⚠️  {alert}")
    else:
        print(f"\n✓ No alerts triggered")


def demo_deployment_decision():
    """Demonstrate deployment decision making."""
    print("\n" + "="*70)
    print("DEMO 5: Deployment Decision Making")
    print("="*70)

    scenarios = [
        {
            "name": "Startup MVP",
            "requirements": {
                "latency_ms": 1000,
                "cost_per_1m": 50,
                "context_window": 4096,
                "finetuning": False
            }
        },
        {
            "name": "Production SaaS",
            "requirements": {
                "latency_ms": 500,
                "cost_per_1m": 10,
                "context_window": 16384,
                "finetuning": True
            }
        },
        {
            "name": "High-volume Enterprise",
            "requirements": {
                "latency_ms": 200,
                "cost_per_1m": 5,
                "context_window": 8192,
                "finetuning": True
            }
        }
    ]

    selector = ModelSelector()

    for scenario in scenarios:
        print(f"\nScenario: {scenario['name']}")
        print(f"  Requirements: {scenario['requirements']}")

        candidates = selector.select_by_criteria(
            max_latency_ms=scenario['requirements']['latency_ms'],
            max_cost_per_1m_tokens=scenario['requirements']['cost_per_1m'],
            min_context_window=scenario['requirements']['context_window'],
            requires_finetuning=scenario['requirements']['finetuning']
        )

        if candidates:
            print(f"  Recommended models:")
            for model in candidates[:3]:
                print(f"    • {model.name} ({model.provider})")
        else:
            print(f"  No models match all requirements")


# ============================================================================
# Main
# ============================================================================

def main():
    """Run all demonstrations."""
    print("\n" + "="*70)
    print("Chapter 8: LLM Engineering Practices")
    print("="*70)

    demo_model_selection()
    demo_cost_analysis()
    demo_error_handling()
    demo_monitoring()
    demo_deployment_decision()

    print("\n" + "="*70)
    print("Summary")
    print("="*70)
    print("""
Key takeaways:
1. Model selection depends on performance, cost, and availability
2. Cost optimization requires monitoring and strategic choices
3. Error handling with exponential backoff improves reliability
4. Monitoring metrics help identify issues early
5. Deployment decisions should be scenario-specific

Next steps:
- Implement monitoring in your LLM application
- Set up cost alerts and budgets
- Test error handling with real API calls
- Document your deployment decisions
    """)


if __name__ == "__main__":
    main()
