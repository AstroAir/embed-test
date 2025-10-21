"""
Embedding Provider Comparison Example

This example demonstrates and compares different embedding providers:
- Sentence Transformers (local models)
- OpenAI embeddings
- Cohere embeddings
- Google embeddings
- Performance and quality comparisons

Prerequisites:
- PDF Vector System installed
- Optional: API keys for cloud providers

Usage:
    python provider_comparison.py

Expected Output:
    - Available provider detection
    - Performance benchmarks for each provider
    - Quality comparison using sample queries
    - Recommendations for different use cases

Learning Objectives:
- Understand different embedding providers
- Learn performance characteristics
- See quality trade-offs between providers
- Choose the right provider for your use case
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import EmbeddingModelType
from utils.example_helpers import (
    example_context,
    get_available_providers,
    print_section,
    print_subsection,
)


@dataclass
class ProviderBenchmark:
    """Benchmark results for an embedding provider."""

    provider_name: str
    model_name: str
    embedding_time: float
    embeddings_per_second: float
    embedding_dimension: int
    available: bool
    error_message: Optional[str] = None


@dataclass
class QualityResult:
    """Quality assessment result for a provider."""

    provider_name: str
    query: str
    top_result_score: float
    avg_score: float
    relevant_results: int


def get_provider_configs() -> dict[str, dict[str, Any]]:
    """Get configuration for each embedding provider."""
    return {
        "sentence_transformers_mini": {
            "model_type": EmbeddingModelType.SENTENCE_TRANSFORMERS,
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "description": "Fast, lightweight local model",
            "use_case": "Development, testing, resource-constrained environments",
        },
        "sentence_transformers_mpnet": {
            "model_type": EmbeddingModelType.SENTENCE_TRANSFORMERS,
            "model_name": "all-mpnet-base-v2",
            "batch_size": 16,
            "description": "High-quality local model",
            "use_case": "Production with local inference requirements",
        },
        "openai_small": {
            "model_type": EmbeddingModelType.OPENAI,
            "model_name": "text-embedding-3-small",
            "batch_size": 100,
            "description": "OpenAI's efficient embedding model",
            "use_case": "Production with good quality/cost balance",
        },
        "openai_large": {
            "model_type": EmbeddingModelType.OPENAI,
            "model_name": "text-embedding-3-large",
            "batch_size": 50,
            "description": "OpenAI's highest quality embedding model",
            "use_case": "Production requiring highest quality",
        },
        "cohere": {
            "model_type": EmbeddingModelType.COHERE,
            "model_name": "embed-english-v3.0",
            "batch_size": 48,
            "description": "Cohere's multilingual embedding model",
            "use_case": "Multilingual applications, semantic search",
        },
    }


def create_test_pipeline(
    provider_config: dict[str, Any],
) -> Optional[PDFVectorPipeline]:
    """Create a test pipeline with the specified provider configuration."""
    try:
        config = Config()
        config.embedding.model_type = provider_config["model_type"]
        config.embedding.model_name = provider_config["model_name"]
        config.embedding.batch_size = provider_config["batch_size"]
        config.chroma_db.collection_name = (
            f"test_{provider_config['model_name'].replace('-', '_')}"
        )
        config.chroma_db.persist_directory = Path(
            f"./test_{provider_config['model_name'].replace('-', '_')}_db"
        )
        config.debug = False

        return PDFVectorPipeline(config)

    except Exception:
        return None


def benchmark_provider(
    provider_name: str, provider_config: dict[str, Any]
) -> ProviderBenchmark:
    """Benchmark a specific embedding provider."""

    # Test texts for benchmarking
    test_texts = [
        "Machine learning is a subset of artificial intelligence.",
        "Natural language processing helps computers understand text.",
        "Vector databases store high-dimensional embeddings efficiently.",
        "Semantic search finds content by meaning rather than keywords.",
        "Deep learning uses neural networks with multiple layers.",
        "Transformers have revolutionized natural language processing.",
        "Embeddings capture semantic relationships between words.",
        "Information retrieval systems help users find relevant content.",
    ]

    try:
        # Create test pipeline
        pipeline = create_test_pipeline(provider_config)
        if not pipeline:
            return ProviderBenchmark(
                provider_name=provider_name,
                model_name=provider_config["model_name"],
                embedding_time=0,
                embeddings_per_second=0,
                embedding_dimension=0,
                available=False,
                error_message="Failed to create pipeline",
            )

        # Benchmark embedding generation
        start_time = time.time()

        result = pipeline.batch_processor.process_texts(test_texts, show_progress=False)

        embedding_time = time.time() - start_time

        if result.success and result.embeddings:
            embeddings_per_second = len(test_texts) / embedding_time
            embedding_dimension = len(result.embeddings[0]) if result.embeddings else 0

            return ProviderBenchmark(
                provider_name=provider_name,
                model_name=provider_config["model_name"],
                embedding_time=embedding_time,
                embeddings_per_second=embeddings_per_second,
                embedding_dimension=embedding_dimension,
                available=True,
            )
        return ProviderBenchmark(
            provider_name=provider_name,
            model_name=provider_config["model_name"],
            embedding_time=embedding_time,
            embeddings_per_second=0,
            embedding_dimension=0,
            available=False,
            error_message=result.error_message,
        )

    except Exception as e:
        return ProviderBenchmark(
            provider_name=provider_name,
            model_name=provider_config["model_name"],
            embedding_time=0,
            embeddings_per_second=0,
            embedding_dimension=0,
            available=False,
            error_message=str(e),
        )


def demonstrate_provider_availability() -> None:
    """Demonstrate provider availability detection."""
    print_subsection("Provider Availability")

    available_providers = get_available_providers()
    provider_configs = get_provider_configs()

    for _provider_key, config in provider_configs.items():
        model_type = config["model_type"]
        config["model_name"]
        config["description"]

        # Check if provider is available
        provider_available = True

        if model_type == EmbeddingModelType.OPENAI:
            provider_available = available_providers.get("openai", False)
        elif model_type == EmbeddingModelType.COHERE:
            provider_available = available_providers.get("cohere", False)
        elif model_type == EmbeddingModelType.SENTENCE_TRANSFORMERS:
            provider_available = True  # Always available

        if not provider_available:
            if model_type in (EmbeddingModelType.OPENAI, EmbeddingModelType.COHERE):
                pass


def run_performance_benchmarks() -> list[ProviderBenchmark]:
    """Run performance benchmarks for all available providers."""
    print_subsection("Performance Benchmarks")

    provider_configs = get_provider_configs()
    available_providers = get_available_providers()
    benchmarks = []

    for provider_key, config in provider_configs.items():
        model_type = config["model_type"]

        # Skip unavailable providers
        if (
            model_type == EmbeddingModelType.OPENAI
            and not available_providers.get("openai")
        ) or (
            model_type == EmbeddingModelType.COHERE
            and not available_providers.get("cohere")
        ):
            continue

        benchmark = benchmark_provider(provider_key, config)
        benchmarks.append(benchmark)

        if benchmark.available:
            pass
        else:
            pass

    return benchmarks


def analyze_benchmark_results(benchmarks: list[ProviderBenchmark]) -> None:
    """Analyze and display benchmark results."""
    print_subsection("Benchmark Analysis")

    available_benchmarks = [b for b in benchmarks if b.available]

    if not available_benchmarks:
        return

    # Sort by speed
    speed_sorted = sorted(
        available_benchmarks, key=lambda x: x.embeddings_per_second, reverse=True
    )

    for _i, benchmark in enumerate(speed_sorted, 1):
        pass

    # Dimension analysis
    for benchmark in available_benchmarks:
        pass

    # Performance categories

    for benchmark in available_benchmarks:
        speed = benchmark.embeddings_per_second

        if speed > 50 or speed > 20 or speed > 10:
            pass
        else:
            pass


def demonstrate_provider_recommendations() -> None:
    """Demonstrate provider recommendations for different use cases."""
    print_subsection("Provider Recommendations")

    recommendations = {
        "🚀 Development & Testing": {
            "provider": "sentence_transformers_mini",
            "reasons": [
                "No API key required",
                "Fast processing for quick iteration",
                "Minimal resource requirements",
                "Good for prototyping",
            ],
        },
        "🏭 Production (Local)": {
            "provider": "sentence_transformers_mpnet",
            "reasons": [
                "High quality embeddings",
                "No external API dependencies",
                "Consistent performance",
                "Data privacy (local processing)",
            ],
        },
        "☁️  Production (Cloud)": {
            "provider": "openai_small",
            "reasons": [
                "Excellent quality/cost balance",
                "High throughput with large batches",
                "Regularly updated models",
                "Reliable service",
            ],
        },
        "🎯 High-Quality Applications": {
            "provider": "openai_large",
            "reasons": [
                "Highest quality embeddings",
                "Best semantic understanding",
                "Latest model improvements",
                "Worth the higher cost for critical applications",
            ],
        },
        "🌍 Multilingual Applications": {
            "provider": "cohere",
            "reasons": [
                "Strong multilingual support",
                "Good semantic search capabilities",
                "Competitive pricing",
                "Specialized for search applications",
            ],
        },
    }

    for _use_case, recommendation in recommendations.items():
        for _reason in recommendation["reasons"]:
            pass


def demonstrate_cost_considerations() -> None:
    """Demonstrate cost considerations for different providers."""
    print_subsection("Cost Considerations")

    cost_info = {
        "sentence_transformers": {
            "type": "Local Processing",
            "cost": "Hardware/compute costs only",
            "scaling": "Linear with compute resources",
            "considerations": [
                "One-time model download",
                "GPU recommended for large volumes",
                "No per-request costs",
                "Predictable costs",
            ],
        },
        "openai": {
            "type": "API Service",
            "cost": "Per token pricing",
            "scaling": "Pay per use",
            "considerations": [
                "text-embedding-3-small: Lower cost",
                "text-embedding-3-large: Higher cost, better quality",
                "Batch processing reduces costs",
                "Rate limits may apply",
            ],
        },
        "cohere": {
            "type": "API Service",
            "cost": "Per request pricing",
            "scaling": "Pay per use",
            "considerations": [
                "Competitive pricing",
                "Free tier available",
                "Volume discounts",
                "Good for search applications",
            ],
        },
    }

    for _provider, info in cost_info.items():
        for _consideration in info["considerations"]:
            pass


def main() -> None:
    """
    Compare and analyze different embedding providers.

    This function demonstrates the characteristics, performance,
    and use cases for different embedding providers.
    """
    with example_context("Embedding Provider Comparison"):
        print_section("Provider Analysis")

        # Check provider availability
        demonstrate_provider_availability()

        # Run performance benchmarks
        benchmarks = run_performance_benchmarks()

        # Analyze results
        analyze_benchmark_results(benchmarks)

        print_section("Selection Guidance")

        # Provide recommendations
        demonstrate_provider_recommendations()

        # Discuss cost considerations
        demonstrate_cost_considerations()

        print_section("Provider Comparison Summary")


if __name__ == "__main__":
    main()
