"""
Sentence Transformers Examples

This example demonstrates using Sentence Transformers models:
- Different model variants and their characteristics
- Local model management and optimization
- Performance tuning for different use cases
- Model selection guidance

Prerequisites:
- PDF Vector System installed
- sentence-transformers library

Usage:
    python sentence_transformers_examples.py

Expected Output:
    - Available model demonstrations
    - Performance comparisons
    - Model selection guidance
    - Optimization techniques

Learning Objectives:
- Understand Sentence Transformers ecosystem
- Learn model selection criteria
- See performance optimization techniques
- Master local embedding deployment
"""

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from utils.example_helpers import example_context, print_section, print_subsection

from vectorflow import Config, PDFVectorPipeline
from vectorflow.config.settings import EmbeddingModelType

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class ModelInfo:
    """Information about a Sentence Transformers model."""

    name: str
    description: str
    dimensions: int
    max_seq_length: int
    size_mb: int
    use_case: str
    performance_tier: str


def get_sentence_transformer_models() -> dict[str, ModelInfo]:
    """Get information about available Sentence Transformers models."""
    return {
        "all-MiniLM-L6-v2": ModelInfo(
            name="all-MiniLM-L6-v2",
            description="Fast and lightweight model, good for development",
            dimensions=384,
            max_seq_length=256,
            size_mb=90,
            use_case="Development, testing, resource-constrained environments",
            performance_tier="Fast",
        ),
        "all-MiniLM-L12-v2": ModelInfo(
            name="all-MiniLM-L12-v2",
            description="Balanced speed and quality",
            dimensions=384,
            max_seq_length=256,
            size_mb=120,
            use_case="General purpose applications",
            performance_tier="Balanced",
        ),
        "all-mpnet-base-v2": ModelInfo(
            name="all-mpnet-base-v2",
            description="High quality model, slower but better results",
            dimensions=768,
            max_seq_length=384,
            size_mb=420,
            use_case="Production applications requiring high quality",
            performance_tier="High Quality",
        ),
        "all-distilroberta-v1": ModelInfo(
            name="all-distilroberta-v1",
            description="Distilled RoBERTa model, good balance",
            dimensions=768,
            max_seq_length=512,
            size_mb=290,
            use_case="Applications needing longer context",
            performance_tier="Balanced",
        ),
        "multi-qa-mpnet-base-dot-v1": ModelInfo(
            name="multi-qa-mpnet-base-dot-v1",
            description="Optimized for question-answering tasks",
            dimensions=768,
            max_seq_length=512,
            size_mb=420,
            use_case="Question answering, FAQ systems",
            performance_tier="Specialized",
        ),
    }


def demonstrate_model_characteristics() -> None:
    """Demonstrate characteristics of different Sentence Transformers models."""
    print_subsection("Model Characteristics")

    models = get_sentence_transformer_models()

    print("\nAvailable Sentence Transformer Models:")
    for model_name, info in models.items():
        print(f"\n  {model_name}:")
        print(f"    Dimensions: {info.dimensions}")
        print(f"    Quality: {info.quality}")
        print(f"    Speed: {info.speed}")
        print(f"    Use case: {info.use_case}")


def benchmark_model(model_name: str, model_info: ModelInfo) -> Optional[dict[str, Any]]:
    """Benchmark a specific Sentence Transformers model."""

    try:
        # Create configuration for this model
        config = Config()
        config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        config.embedding.model_name = model_name
        config.embedding.batch_size = 16
        config.chroma_db.collection_name = f"test_{model_name.replace('-', '_')}"
        config.chroma_db.persist_directory = Path(
            f"./test_{model_name.replace('-', '_')}_db"
        )
        config.debug = False

        # Test texts
        test_texts = [
            "Machine learning algorithms can automatically learn patterns from data.",
            "Natural language processing enables computers to understand human language.",
            "Deep learning uses neural networks with multiple hidden layers.",
            "Artificial intelligence aims to create systems that can perform tasks requiring human intelligence.",
            "Vector databases efficiently store and search high-dimensional embeddings.",
        ]

        # Create pipeline and measure initialization time
        init_start = time.time()
        pipeline = PDFVectorPipeline(config)
        init_time = time.time() - init_start

        # Measure embedding generation time
        embed_start = time.time()
        result = pipeline.batch_processor.process_texts(test_texts, show_progress=False)
        embed_time = time.time() - embed_start

        if result.success:
            embeddings_per_second = len(test_texts) / embed_time

            return {
                "model_name": model_name,
                "init_time": init_time,
                "embed_time": embed_time,
                "embeddings_per_second": embeddings_per_second,
                "dimensions": len(result.embeddings[0]) if result.embeddings else 0,
                "success": True,
            }
        return None

    except Exception as e:
        print(f"\nBenchmark failed for {model_name}: {e}")
        return None


def run_model_benchmarks() -> list[dict[str, Any]]:
    """Run benchmarks for available models."""
    print_subsection("Performance Benchmarks")

    models = get_sentence_transformer_models()
    benchmarks = []

    # Test a subset of models to avoid long execution time
    test_models = ["all-MiniLM-L6-v2", "all-mpnet-base-v2"]

    print(f"\nTesting {len(test_models)} models...")
    print("Note: This may take a minute as models are loaded for the first time\n")

    for model_name in test_models:
        if model_name in models:
            model_info = models[model_name]
            print(f"Benchmarking {model_name}...")
            benchmark = benchmark_model(model_name, model_info)

            if benchmark:
                benchmarks.append(benchmark)
                print(f"\n  {benchmark['model_name']} Results:")
                print(f"    Init time: {benchmark['init_time']:.3f}s")
                print(f"    Embed time: {benchmark['embed_time']:.3f}s")
                print(
                    f"    Speed: {benchmark['embeddings_per_second']:.2f} embeddings/sec"
                )
                print(f"    Dimensions: {benchmark['dimensions']}")
            else:
                print(f"  Benchmark failed for {model_name}")
        else:
            print(f"  Model {model_name} not found in available models")

    # Display comparison if multiple benchmarks succeeded
    if len(benchmarks) >= 2:
        print("\n" + "=" * 50)
        print("Benchmark Comparison:")
        print("=" * 50)

        fastest = max(benchmarks, key=lambda x: x["embeddings_per_second"])
        slowest = min(benchmarks, key=lambda x: x["embeddings_per_second"])

        print(f"\nFastest model: {fastest['model_name']}")
        print(f"  Speed: {fastest['embeddings_per_second']:.2f} embeddings/sec")

        print(f"\nSlowest model: {slowest['model_name']}")
        print(f"  Speed: {slowest['embeddings_per_second']:.2f} embeddings/sec")

        speed_diff = fastest["embeddings_per_second"] / slowest["embeddings_per_second"]
        print(f"\nSpeed difference: {speed_diff:.2f}x")

        print("\nRecommendations:")
        print(f"  - Use {fastest['model_name']} for speed-critical applications")
        print(f"  - Use {slowest['model_name']} for quality-critical applications")

    return benchmarks


def demonstrate_optimization_techniques() -> None:
    """Demonstrate optimization techniques for Sentence Transformers."""
    print_subsection("Optimization Techniques")

    # Batch size optimization
    print("\nBatch Size Optimization:")
    print("  Larger batches = Better throughput, Higher memory usage")
    print("  Smaller batches = Lower latency, Lower memory usage")
    print("  Recommended: Start with 32 and adjust based on your needs")

    # Model selection optimization
    optimization_guide = {
        "Speed Priority": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 32,
            "use_case": "Real-time applications, development",
        },
        "Quality Priority": {
            "model": "all-mpnet-base-v2",
            "batch_size": 16,
            "use_case": "Production search, high-quality requirements",
        },
        "Balanced": {
            "model": "all-MiniLM-L12-v2",
            "batch_size": 24,
            "use_case": "General production applications",
        },
    }

    print("\nOptimization Guide:")
    for priority, config in optimization_guide.items():
        print(f"\n  {priority}:")
        print(f"    Model: {config['model']}")
        print(f"    Batch size: {config['batch_size']}")
        print(f"    Use case: {config['use_case']}")

    # Hardware optimization
    hardware_tips = [
        "Use GPU acceleration when available (CUDA/MPS)",
        "Increase batch size with more GPU memory",
        "Use CPU with multiple workers for CPU-only setups",
        "Consider model quantization for memory savings",
        "Use SSD storage for faster model loading",
    ]

    print("\nHardware Optimization Tips:")
    for tip in hardware_tips:
        print(f"  - {tip}")


def demonstrate_model_management() -> None:
    """Demonstrate model management and caching."""
    print_subsection("Model Management")

    print("\nSentence Transformers models are managed through HuggingFace Hub")
    print("Models are automatically downloaded and cached on first use\n")

    # Model caching
    caching_info = [
        "Models are cached locally after first download",
        "Cache location: ~/.cache/huggingface/transformers/",
        "Shared across applications using the same model",
        "Can be pre-downloaded for offline deployment",
        "Consider cache size for multiple models",
    ]

    print("\nModel Caching Information:")
    for info in caching_info:
        print(f"  - {info}")

    # Model versioning
    versioning_tips = [
        "Pin specific model versions in production",
        "Test new model versions before deployment",
        "Keep track of model performance changes",
        "Use model tags for reproducible deployments",
        "Document model selection rationale",
    ]

    print("\nModel Versioning Tips:")
    for tip in versioning_tips:
        print(f"  - {tip}")

    # Deployment considerations
    deployment_tips = [
        "Pre-download models in container images",
        "Use model warm-up for faster first requests",
        "Monitor model memory usage",
        "Implement model health checks",
        "Consider model serving frameworks for scale",
    ]

    print("\nDeployment Considerations:")
    for tip in deployment_tips:
        print(f"  - {tip}")


def demonstrate_configuration() -> None:
    """Demonstrate configuration options."""
    print_subsection("Configuration Options")

    # Custom configuration example

    # Environment-specific configurations

    env_configs = {
        "Development": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 16,
            "workers": 2,
            "rationale": "Fast iteration, minimal resources",
        },
        "Testing": {
            "model": "all-MiniLM-L6-v2",
            "batch_size": 8,
            "workers": 1,
            "rationale": "Deterministic, minimal resources",
        },
        "Production": {
            "model": "all-mpnet-base-v2",
            "batch_size": 32,
            "workers": 8,
            "rationale": "High quality, optimized performance",
        },
    }

    print("\nEnvironment-Specific Configurations:")
    for env, config in env_configs.items():
        print(f"\n  {env}:")
        print(f"    Model: {config['model']}")
        print(f"    Batch size: {config['batch_size']}")
        print(f"    Workers: {config['workers']}")
        print(f"    Rationale: {config['rationale']}")


def main() -> None:
    """
    Demonstrate Sentence Transformers usage and optimization.

    This function shows how to effectively use Sentence Transformers
    models for different use cases and optimize their performance.
    """
    with example_context("Sentence Transformers Examples"):
        print_section("Sentence Transformers Overview")

        # Show model characteristics
        demonstrate_model_characteristics()

        # Run performance benchmarks
        run_model_benchmarks()

        print_section("Optimization and Management")

        # Show optimization techniques
        demonstrate_optimization_techniques()

        # Show model management
        demonstrate_model_management()

        # Show configuration options
        demonstrate_configuration()

        print_section("Sentence Transformers Summary")


if __name__ == "__main__":
    main()
