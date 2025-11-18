"""
OpenAI Embeddings Examples

This example demonstrates using OpenAI's embedding models:
- Different OpenAI embedding models
- API configuration and authentication
- Cost optimization strategies
- Error handling and rate limiting

Prerequisites:
- PDF Vector System installed
- OpenAI API key (set OPENAI_API_KEY environment variable)

Usage:
    python openai_examples.py

Expected Output:
    - OpenAI model demonstrations
    - Cost analysis and optimization
    - Rate limiting and error handling examples
    - Best practices for production use

Learning Objectives:
- Understand OpenAI embedding models
- Learn cost optimization techniques
- Master API error handling
- See production deployment patterns
"""

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from examples.utils.example_helpers import (
    check_api_key,
    example_context,
    print_section,
    print_subsection,
)

from vectorflow import Config, PDFVectorPipeline
from vectorflow.core.config.settings import EmbeddingModelType

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class OpenAIModelInfo:
    """Information about OpenAI embedding models."""

    name: str
    description: str
    dimensions: int
    max_tokens: int
    cost_per_1k_tokens: float
    use_case: str
    performance_tier: str


def get_openai_models() -> dict[str, OpenAIModelInfo]:
    """Get information about available OpenAI embedding models."""
    return {
        "text-embedding-3-small": OpenAIModelInfo(
            name="text-embedding-3-small",
            description="Most efficient OpenAI embedding model",
            dimensions=1536,
            max_tokens=8191,
            cost_per_1k_tokens=0.00002,  # $0.00002 per 1K tokens
            use_case="General purpose, cost-sensitive applications",
            performance_tier="Efficient",
        ),
        "text-embedding-3-large": OpenAIModelInfo(
            name="text-embedding-3-large",
            description="Highest quality OpenAI embedding model",
            dimensions=3072,
            max_tokens=8191,
            cost_per_1k_tokens=0.00013,  # $0.00013 per 1K tokens
            use_case="High-quality applications, research",
            performance_tier="Premium",
        ),
        "text-embedding-ada-002": OpenAIModelInfo(
            name="text-embedding-ada-002",
            description="Legacy OpenAI embedding model (deprecated)",
            dimensions=1536,
            max_tokens=8191,
            cost_per_1k_tokens=0.0001,  # $0.0001 per 1K tokens
            use_case="Legacy applications (use text-embedding-3-small instead)",
            performance_tier="Legacy",
        ),
    }


def check_openai_availability() -> bool:
    """Check if OpenAI API is available and configured."""
    print_subsection("OpenAI API Configuration")

    api_key = os.environ.get("OPENAI_API_KEY")

    if not api_key:
        print("OPENAI_API_KEY is not set. Live API calls will be skipped.")
        return False

    # Mask the API key for display
    masked = f"{api_key[:8]}...{api_key[-4:]}" if len(api_key) > 12 else "***"
    print("OpenAI API key detected:", masked)

    return True


def demonstrate_model_characteristics() -> None:
    """Demonstrate characteristics of OpenAI embedding models."""
    print_subsection("OpenAI Model Characteristics")

    models = get_openai_models()

    for model_name, info in models.items():
        legacy_note = (
            " (LEGACY - consider migrating to text-embedding-3-small)"
            if info.performance_tier == "Legacy"
            else ""
        )
        print(f"\nModel: {info.name}{legacy_note}")
        print(f"  Description: {info.description}")
        print(f"  Dimensions: {info.dimensions}")
        print(f"  Max tokens: {info.max_tokens}")
        print(f"  Cost per 1K tokens: ${info.cost_per_1k_tokens:.5f}")
        print(f"  Recommended use: {info.use_case}")
        print(f"  Tier: {info.performance_tier}")


def benchmark_openai_model(
    model_name: str, model_info: OpenAIModelInfo
) -> Optional[dict[str, Any]]:
    """Benchmark a specific OpenAI model."""

    try:
        # Create configuration for this model
        config = Config()
        config.embedding.model_type = EmbeddingModelType.OPENAI
        config.embedding.model_name = model_name
        config.embedding.batch_size = 50  # OpenAI supports larger batches
        config.chroma_db.collection_name = f"test_openai_{model_name.replace('-', '_')}"
        config.chroma_db.persist_directory = Path(
            f"./test_openai_{model_name.replace('-', '_')}_db"
        )
        config.debug = False

        # Test texts
        test_texts = [
            "OpenAI's embedding models provide high-quality vector representations of text.",
            "These models are trained on diverse internet text and can understand context.",
            "The text-embedding-3-small model offers the best balance of cost and performance.",
            "For applications requiring the highest quality, text-embedding-3-large is recommended.",
            "Proper batch processing can significantly reduce API costs and improve throughput.",
        ]

        # Create pipeline and measure performance
        start_time = time.time()
        pipeline = PDFVectorPipeline(config)

        # Measure embedding generation
        embed_start = time.time()
        result = pipeline.batch_processor.process_texts(test_texts, show_progress=False)
        embed_time = time.time() - embed_start

        if result.success:
            total_time = time.time() - start_time
            embeddings_per_second = len(test_texts) / embed_time

            # Estimate cost (rough approximation)
            total_chars = sum(len(text) for text in test_texts)
            estimated_tokens = total_chars // 4  # Rough token estimation
            estimated_cost = (estimated_tokens / 1000) * model_info.cost_per_1k_tokens

            return {
                "model_name": model_name,
                "total_time": total_time,
                "embed_time": embed_time,
                "embeddings_per_second": embeddings_per_second,
                "dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                "estimated_cost": estimated_cost,
                "estimated_tokens": estimated_tokens,
                "success": True,
            }
        return None

    except Exception:
        return None


def run_openai_benchmarks() -> list[dict[str, Any]]:
    """Run benchmarks for available OpenAI models."""
    print_subsection("OpenAI Performance Benchmarks")

    # Use helper to check provider availability
    if not check_api_key("openai"):
        print("OpenAI provider not available. Skipping benchmarks.")
        return []

    models = get_openai_models()
    benchmarks = []

    # Test current models (skip legacy)
    test_models = ["text-embedding-3-small", "text-embedding-3-large"]

    for model_name in test_models:
        if model_name in models:
            model_info = models[model_name]
            print(f"\nRunning benchmark for {model_name}...")
            benchmark = benchmark_openai_model(model_name, model_info)

            if benchmark:
                benchmarks.append(benchmark)
            else:
                print(f"  Benchmark for {model_name} failed or returned no data.")

    if not benchmarks:
        print("\nNo successful benchmark results were collected.")
        return benchmarks

    print("\nBenchmark results:")
    for result in benchmarks:
        print(
            f"  {result['model_name']}: "
            f"{result['embeddings_per_second']:.1f} embeddings/s, "
            f"dim={result['dimensions']}, "
            f"est_tokens={result['estimated_tokens']}, "
            f"est_cost=${result['estimated_cost']:.4f}"
        )

    fastest = max(benchmarks, key=lambda b: b["embeddings_per_second"])
    slowest = min(benchmarks, key=lambda b: b["embeddings_per_second"])
    if fastest["embeddings_per_second"] > 0 and slowest["embeddings_per_second"] > 0:
        speedup = fastest["embeddings_per_second"] / slowest["embeddings_per_second"]
        print(
            f"\nFastest model: {fastest['model_name']} "
            f"({fastest['embeddings_per_second']:.1f} emb/s)"
        )
        print(
            f"Slowest model: {slowest['model_name']} "
            f"({slowest['embeddings_per_second']:.1f} emb/s)"
        )
        print(f"Speed difference: {speedup:.2f}x")

    return benchmarks


def demonstrate_cost_optimization() -> None:
    """Demonstrate cost optimization strategies for OpenAI."""
    print_subsection("Cost Optimization Strategies")

    # Batch processing optimization
    batch_strategies = [
        "Use larger batch sizes (up to 2048 inputs per request)",
        "Combine multiple small requests into batches",
        "Process documents in chunks to maximize batch efficiency",
        "Monitor rate limits to avoid throttling",
        "Use async processing for large volumes",
    ]

    print("\nBatching strategies:")
    for strategy in batch_strategies:
        print(f"  - {strategy}")

    # Model selection optimization
    model_guidance = {
        "text-embedding-3-small": {
            "cost_factor": "1x",
            "quality": "High",
            "recommendation": "Default choice for most applications",
        },
        "text-embedding-3-large": {
            "cost_factor": "6.5x",
            "quality": "Highest",
            "recommendation": "Use only when highest quality is required",
        },
    }

    print("\nModel selection guidance:")
    for model, info in model_guidance.items():
        print(f"  {model}:")
        print(f"    Cost factor: {info['cost_factor']}")
        print(f"    Quality: {info['quality']}")
        print(f"    Recommendation: {info['recommendation']}")

    # Cost estimation example

    example_scenarios = [
        {
            "scenario": "Small application (10K documents, 500 tokens each)",
            "tokens": 5_000_000,
            "small_cost": 5_000_000 / 1000 * 0.00002,
            "large_cost": 5_000_000 / 1000 * 0.00013,
        },
        {
            "scenario": "Medium application (100K documents, 500 tokens each)",
            "tokens": 50_000_000,
            "small_cost": 50_000_000 / 1000 * 0.00002,
            "large_cost": 50_000_000 / 1000 * 0.00013,
        },
    ]

    print("\nCost estimation examples:")
    for scenario in example_scenarios:
        small_cost = scenario["small_cost"]
        large_cost = scenario["large_cost"]
        diff = large_cost - small_cost
        ratio = large_cost / small_cost if small_cost else 0
        print(f"  {scenario['scenario']}:")
        print(f"    text-embedding-3-small: ${small_cost:.2f}")
        print(f"    text-embedding-3-large: ${large_cost:.2f}")
        print(f"    Large vs small: ~{ratio:.1f}x cost (+${diff:.2f})")


def demonstrate_error_handling() -> None:
    """Demonstrate error handling and rate limiting for OpenAI API."""
    print_subsection("Error Handling and Rate Limiting")

    # Common error scenarios
    error_scenarios = [
        {
            "error": "Rate limit exceeded",
            "cause": "Too many requests per minute/day",
            "solution": "Implement exponential backoff, reduce batch size",
        },
        {
            "error": "Invalid API key",
            "cause": "Missing or incorrect API key",
            "solution": "Verify API key configuration",
        },
        {
            "error": "Token limit exceeded",
            "cause": "Input text too long",
            "solution": "Split text into smaller chunks",
        },
        {
            "error": "Network timeout",
            "cause": "Network connectivity issues",
            "solution": "Implement retry logic with timeouts",
        },
    ]

    print("\nCommon error scenarios:")
    for scenario in error_scenarios:
        print(f"  - {scenario['error']}:")
        print(f"      Cause: {scenario['cause']}")
        print(f"      Solution: {scenario['solution']}")

    # Rate limiting best practices
    rate_limit_tips = [
        "Monitor rate limit headers in API responses",
        "Implement exponential backoff for rate limit errors",
        "Use smaller batch sizes if hitting rate limits",
        "Consider upgrading API tier for higher limits",
        "Implement request queuing for high-volume applications",
    ]

    print("\nRate limiting best practices:")
    for tip in rate_limit_tips:
        print(f"  - {tip}")

    # Configuration example


def demonstrate_production_patterns() -> None:
    """Demonstrate production deployment patterns for OpenAI."""
    print_subsection("Production Deployment Patterns")

    # Security practices
    security_practices = [
        "Store API keys in environment variables or secret managers",
        "Never commit API keys to version control",
        "Use different API keys for different environments",
        "Implement API key rotation procedures",
        "Monitor API usage for anomalies",
    ]

    print("\nSecurity practices:")
    for practice in security_practices:
        print(f"  - {practice}")

    # Monitoring and observability
    monitoring_practices = [
        "Track API usage and costs",
        "Monitor response times and error rates",
        "Set up alerts for rate limit violations",
        "Log API errors with sufficient context",
        "Implement health checks for API connectivity",
    ]

    print("\nMonitoring and observability:")
    for practice in monitoring_practices:
        print(f"  - {practice}")

    # Scalability patterns
    scalability_patterns = [
        "Use connection pooling for API requests",
        "Implement caching for frequently used embeddings",
        "Use async processing for large batches",
        "Consider embedding caching strategies",
        "Plan for API quota management",
    ]

    print("\nScalability patterns:")
    for pattern in scalability_patterns:
        print(f"  - {pattern}")


def main() -> None:
    """
    Demonstrate OpenAI embedding usage and best practices.

    This function shows how to effectively use OpenAI's embedding
    models with proper configuration, error handling, and optimization.
    """
    with example_context("OpenAI Embeddings Examples"):
        print_section("OpenAI Embeddings Overview")

        # Check API availability
        api_available = check_openai_availability()

        # Show model characteristics
        demonstrate_model_characteristics()

        if api_available:
            # Run benchmarks if API is available
            benchmarks = run_openai_benchmarks()
            if not benchmarks:
                print(
                    "No benchmark data was produced. This may be due to API errors "
                    "or connectivity issues."
                )
        else:
            print(
                "OpenAI API key not available - skipping live benchmarks. "
                "Static model information and best practices are still shown."
            )

        print_section("Optimization and Best Practices")

        # Show cost optimization
        demonstrate_cost_optimization()

        # Show error handling
        demonstrate_error_handling()

        # Show production patterns
        demonstrate_production_patterns()

        print_section("OpenAI Embeddings Summary")


if __name__ == "__main__":
    main()
