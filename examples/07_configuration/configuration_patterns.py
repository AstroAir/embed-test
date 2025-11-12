"""
Configuration Patterns Example

This example demonstrates configuration patterns for the PDF Vector System:
- Multi-provider configuration with fallbacks
- Environment-specific configurations
- Dynamic configuration switching
- Configuration validation and optimization
- Performance tuning patterns

Prerequisites:
- PDF Vector System installed
- Understanding of configuration basics
- Optional: API keys for cloud providers

Usage:
    python configuration_patterns.py

Expected Output:
    - Configuration patterns demonstrated
    - Provider fallback mechanisms
    - Performance optimization examples
    - Configuration validation results

Learning Objectives:
- Master configuration patterns
- Learn provider fallback strategies
- Understand performance optimization
- See production-ready configuration examples
"""

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from utils.example_helpers import (
    example_context,
    get_available_providers,
    print_section,
    print_subsection,
)

from pdf_vector_system import Config
from pdf_vector_system.config.settings import EmbeddingModelType, LogLevel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


@dataclass
class ProviderConfig:
    """Configuration for a specific embedding provider."""

    model_type: EmbeddingModelType
    model_name: str
    batch_size: int
    description: str
    requires_api_key: bool = False
    api_key_env: Optional[str] = None


def get_provider_configurations() -> list[ProviderConfig]:
    """Get configurations for all supported providers."""
    return [
        ProviderConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            description="Fast, lightweight local model",
            requires_api_key=False,
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-mpnet-base-v2",
            batch_size=16,
            description="High-quality local model",
            requires_api_key=False,
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            batch_size=100,
            description="OpenAI's efficient embedding model",
            requires_api_key=True,
            api_key_env="OPENAI_API_KEY",
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-large",
            batch_size=50,
            description="OpenAI's high-quality embedding model",
            requires_api_key=True,
            api_key_env="OPENAI_API_KEY",
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.COHERE,
            model_name="embed-english-v3.0",
            batch_size=48,
            description="Cohere's multilingual embedding model",
            requires_api_key=True,
            api_key_env="COHERE_API_KEY",
        ),
    ]


def demonstrate_provider_fallback() -> None:
    """Demonstrate automatic provider fallback configuration."""
    print_subsection("Provider Fallback Configuration")

    provider_configs = get_provider_configurations()
    available_providers = get_available_providers()

    # Define preferred order (best to fallback)
    preferred_order = [
        EmbeddingModelType.OPENAI,
        EmbeddingModelType.COHERE,
        EmbeddingModelType.SENTENCE_TRANSFORMERS,
    ]

    selected_config = None

    for preferred_type in preferred_order:
        # Find configurations for this provider type
        type_configs = [c for c in provider_configs if c.model_type == preferred_type]

        for config in type_configs:
            provider_key = config.model_type.value.replace("-", "_")

            if config.requires_api_key:
                if available_providers.get(provider_key, False):
                    selected_config = config
                    break
            else:
                selected_config = config
                break

        if selected_config:
            break

    if selected_config:
        # Create configuration with selected provider
        print(f"\nSelected provider: {selected_config.model_type.value}")
        print(f"Model: {selected_config.model_name}")
        print(f"Description: {selected_config.description}")

        config = Config()
        config.embedding.model_type = selected_config.model_type
        config.embedding.model_name = selected_config.model_name
        config.embedding.batch_size = selected_config.batch_size

        return config
    else:
        print("\nNo suitable provider found with available API keys")
        return None


def demonstrate_environment_specific_configs() -> dict[str, Config]:
    """Demonstrate environment-specific configurations."""
    print_subsection("Environment-Specific Configurations")

    configs = {}

    # Development configuration
    dev_config = Config()
    dev_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    dev_config.embedding.model_name = "all-MiniLM-L6-v2"
    dev_config.embedding.batch_size = 16
    dev_config.text_processing.chunk_size = 800
    dev_config.text_processing.chunk_overlap = 100
    dev_config.chroma_db.collection_name = "dev_documents"
    dev_config.chroma_db.persist_directory = Path("./dev_chroma_db")
    dev_config.logging.level = LogLevel.DEBUG
    dev_config.debug = True
    dev_config.max_workers = 2

    configs["development"] = dev_config
    print("\nDevelopment Config:")
    print(f"  Model: {dev_config.embedding.model_name}")
    print(f"  Batch size: {dev_config.embedding.batch_size}")
    print(f"  Workers: {dev_config.max_workers}")

    # Testing configuration
    test_config = Config()
    test_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    test_config.embedding.model_name = "all-MiniLM-L6-v2"
    test_config.embedding.batch_size = 8
    test_config.text_processing.chunk_size = 500
    test_config.text_processing.chunk_overlap = 50
    test_config.chroma_db.collection_name = "test_documents"
    test_config.chroma_db.persist_directory = Path("./test_chroma_db")
    test_config.logging.level = LogLevel.INFO
    test_config.debug = True
    test_config.max_workers = 1

    configs["testing"] = test_config
    print("\nTesting Config:")
    print(f"  Model: {test_config.embedding.model_name}")
    print(f"  Batch size: {test_config.embedding.batch_size}")
    print(f"  Workers: {test_config.max_workers}")

    # Production configuration
    prod_config = Config()

    # Use best available provider for production
    available_providers = get_available_providers()
    if available_providers.get("openai"):
        prod_config.embedding.model_type = EmbeddingModelType.OPENAI
        prod_config.embedding.model_name = "text-embedding-3-small"
        prod_config.embedding.batch_size = 100
    else:
        prod_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        prod_config.embedding.model_name = "all-mpnet-base-v2"
        prod_config.embedding.batch_size = 32

    prod_config.text_processing.chunk_size = 1200
    prod_config.text_processing.chunk_overlap = 200
    prod_config.chroma_db.collection_name = "production_documents"
    prod_config.chroma_db.persist_directory = Path("/data/chroma_db")
    prod_config.logging.level = LogLevel.INFO
    prod_config.debug = False
    prod_config.max_workers = 8

    configs["production"] = prod_config
    print("\nProduction Config:")
    print(f"  Model type: {prod_config.embedding.model_type.value}")
    print(f"  Model: {prod_config.embedding.model_name}")
    print(f"  Batch size: {prod_config.embedding.batch_size}")
    print(f"  Workers: {prod_config.max_workers}")

    return configs


def demonstrate_performance_optimization() -> None:
    """Demonstrate performance optimization configurations."""
    print_subsection("Performance Optimization")

    # High-throughput configuration
    print("\nHigh-Throughput Configuration:")
    print("  Optimized for processing large volumes of documents")
    throughput_config = Config()
    throughput_config.embedding.batch_size = 200  # Large batches
    throughput_config.text_processing.chunk_size = 1500  # Larger chunks
    throughput_config.max_workers = 16  # Many workers
    print(f"    Batch size: {throughput_config.embedding.batch_size}")
    print(f"    Chunk size: {throughput_config.text_processing.chunk_size}")
    print(f"    Workers: {throughput_config.max_workers}")
    print("    Best for: Batch processing, data pipelines")

    # Low-latency configuration
    print("\nLow-Latency Configuration:")
    print("  Optimized for quick response times")
    latency_config = Config()
    latency_config.embedding.batch_size = 16  # Small batches
    latency_config.text_processing.chunk_size = 600  # Smaller chunks
    latency_config.max_workers = 4  # Moderate workers
    print(f"    Batch size: {latency_config.embedding.batch_size}")
    print(f"    Chunk size: {latency_config.text_processing.chunk_size}")
    print(f"    Workers: {latency_config.max_workers}")
    print("    Best for: Real-time search, interactive applications")

    # Memory-optimized configuration
    print("\nMemory-Optimized Configuration:")
    print("  Optimized for limited memory environments")
    memory_config = Config()
    memory_config.embedding.batch_size = 8  # Very small batches
    memory_config.text_processing.chunk_size = 400  # Small chunks
    memory_config.max_workers = 2  # Few workers
    print(f"    Batch size: {memory_config.embedding.batch_size}")
    print(f"    Chunk size: {memory_config.text_processing.chunk_size}")
    print(f"    Workers: {memory_config.max_workers}")
    print("    Best for: Resource-constrained environments, edge devices")


def demonstrate_dynamic_configuration() -> None:
    """Demonstrate dynamic configuration switching."""
    print_subsection("Dynamic Configuration Switching")

    # Simulate different conditions
    conditions = {
        "document_count": 100,
        "available_memory_gb": 8,
        "has_gpu": False,
        "is_production": False,
    }

    print("\nCurrent conditions:")
    for key, value in conditions.items():
        print(f"  {key}: {value}")

    # Create adaptive configuration
    config = Config()

    # Adjust based on document count
    if conditions["document_count"] > 1000:
        config.embedding.batch_size = 100
        config.max_workers = 8
    elif conditions["document_count"] > 100:
        config.embedding.batch_size = 50
        config.max_workers = 4
    else:
        config.embedding.batch_size = 16
        config.max_workers = 2

    # Adjust based on available memory
    if conditions["available_memory_gb"] < 4:
        config.text_processing.chunk_size = 400
        config.embedding.batch_size = min(config.embedding.batch_size, 8)
    elif conditions["available_memory_gb"] >= 16:
        # Can handle larger batches with more memory
        config.embedding.batch_size = min(config.embedding.batch_size * 2, 200)

    # Adjust based on GPU availability
    if conditions["has_gpu"]:
        # Can use larger batches with GPU
        config.embedding.batch_size = int(config.embedding.batch_size * 1.5)
        print("  Note: GPU detected, increasing batch size")

    # Adjust based on environment
    if conditions["is_production"]:
        config.logging.level = LogLevel.WARNING
        config.debug = False
    else:
        config.logging.level = LogLevel.DEBUG
        config.debug = True

    print("\nAdaptive configuration created:")
    print(f"  Batch size: {config.embedding.batch_size}")
    print(f"  Chunk size: {config.text_processing.chunk_size}")
    print(f"  Workers: {config.max_workers}")
    print(f"  Log level: {config.logging.level.value}")
    print(f"  Debug mode: {config.debug}")

    # Provide reasoning
    print("\nConfiguration reasoning:")
    if conditions["document_count"] > 1000:
        print("  - High document count: Using large batches and many workers")
    elif conditions["document_count"] > 100:
        print("  - Medium document count: Using moderate settings")
    else:
        print("  - Low document count: Using small batches for efficiency")

    if conditions["available_memory_gb"] < 4:
        print("  - Low memory: Reduced batch size and chunk size")
    elif conditions["available_memory_gb"] >= 16:
        print("  - High memory: Increased batch size for better throughput")


def demonstrate_configuration_validation() -> None:
    """Demonstrate configuration validation."""
    print_subsection("Configuration Validation")

    # Valid configuration
    try:
        valid_config = Config()
        valid_config.embedding.batch_size = 32
        valid_config.text_processing.chunk_size = 1000
        valid_config.text_processing.chunk_overlap = 200
        valid_config.max_workers = 4

        # Validate relationships
        if (
            valid_config.text_processing.chunk_overlap
            >= valid_config.text_processing.chunk_size
        ):
            raise ValueError("Chunk overlap must be less than chunk size")

        if valid_config.embedding.batch_size <= 0:
            raise ValueError("Batch size must be positive")

        print("\nValid configuration passed all checks")

    except Exception as e:
        print(f"\nValidation error: {e}")

    # Invalid configurations

    invalid_configs = [
        {
            "name": "Negative batch size",
            "batch_size": -1,
            "chunk_size": 1000,
            "chunk_overlap": 200,
        },
        {
            "name": "Overlap >= chunk size",
            "batch_size": 32,
            "chunk_size": 500,
            "chunk_overlap": 500,
        },
        {
            "name": "Zero chunk size",
            "batch_size": 32,
            "chunk_size": 0,
            "chunk_overlap": 100,
        },
    ]

    for invalid_config in invalid_configs:
        try:
            config = Config()
            config.embedding.batch_size = invalid_config["batch_size"]
            config.text_processing.chunk_size = invalid_config["chunk_size"]
            config.text_processing.chunk_overlap = invalid_config["chunk_overlap"]

            # Validate
            if config.embedding.batch_size <= 0:
                raise ValueError("Batch size must be positive")
            if config.text_processing.chunk_size <= 0:
                raise ValueError("Chunk size must be positive")
            if (
                config.text_processing.chunk_overlap
                >= config.text_processing.chunk_size
            ):
                raise ValueError("Chunk overlap must be less than chunk size")

            print(f"Config '{invalid_config['name']}' unexpectedly passed validation")

        except Exception as e:
            print(f"\nConfig '{invalid_config['name']}' correctly failed: {e}")


def main() -> None:
    """
    Demonstrate configuration patterns for the PDF Vector System.

    This function shows configuration techniques for
    production-ready deployments.
    """
    with example_context("Configuration Patterns"):
        print_section("Configuration Patterns")

        # Demonstrate provider fallback
        demonstrate_provider_fallback()

        # Demonstrate environment-specific configurations
        demonstrate_environment_specific_configs()

        # Demonstrate performance optimization
        demonstrate_performance_optimization()

        # Demonstrate dynamic configuration
        demonstrate_dynamic_configuration()

        # Demonstrate configuration validation
        demonstrate_configuration_validation()

        print_section("Best Practices")


if __name__ == "__main__":
    main()
