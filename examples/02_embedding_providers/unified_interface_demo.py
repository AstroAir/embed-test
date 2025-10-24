"""
Unified Embedding Interface Demo

This example demonstrates how all embedding providers implement the same unified interface,
making it easy to switch between different providers without changing your code.
"""

from pdf_vector_system.config.settings import EmbeddingConfig, EmbeddingModelType
from pdf_vector_system.embeddings import (
    EmbeddingProviderRegistry,
    EmbeddingServiceFactory,
)


def demonstrate_unified_interface():
    """Demonstrate that all providers implement the same interface."""
    print("=" * 80)
    print("UNIFIED EMBEDDING INTERFACE DEMONSTRATION")
    print("=" * 80)
    print()

    # Sample texts for embedding
    texts = [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Natural language processing enables computers to understand human language.",
    ]

    # Example 1: List all available providers
    print("1. AVAILABLE EMBEDDING PROVIDERS")
    print("-" * 80)
    providers = EmbeddingProviderRegistry.list_all_providers()
    for provider in providers:
        print(f"\n  Provider: {provider['name']}")
        print(f"    Type: {provider['type']}")
        print(f"    Local: {provider['is_local']}")
        print(f"    Requires API Key: {provider['requires_api_key']}")
        print(f"    Default Model: {provider['default_model']}")
        print(f"    Available Models: {provider['model_count']}")

    # Example 2: Get detailed provider information
    print("\n\n2. DETAILED PROVIDER INFORMATION")
    print("-" * 80)
    openai_info = EmbeddingProviderRegistry.get_provider_info(EmbeddingModelType.OPENAI)
    print(f"\nOpenAI Provider Details:")
    print(f"  Supported Models: {', '.join(openai_info['supported_models'])}")
    print(f"  Required Packages: {', '.join(openai_info['required_packages'])}")
    print(f"  Install Command: {openai_info['install_command']}")
    print(f"  Config Parameters: {', '.join(openai_info['config_params'])}")
    print(f"  Max Sequence Length: {openai_info['max_sequence_length']}")

    # Example 3: Get recommended configuration for different use cases
    print("\n\n3. RECOMMENDED CONFIGURATIONS BY USE CASE")
    print("-" * 80)

    use_cases = ["general", "semantic_search", "multilingual", "high_quality"]
    for use_case in use_cases:
        # Local preference
        local_rec = EmbeddingProviderRegistry.get_recommended_config(
            use_case=use_case, prefer_local=True
        )
        print(f"\n  Use Case: {use_case.upper()} (Local)")
        print(f"    Provider: {local_rec['model_type'].value}")
        print(f"    Model: {local_rec['model_name']}")
        print(f"    Reason: {local_rec['reason']}")

    # Example 4: Validate configuration
    print("\n\n4. CONFIGURATION VALIDATION")
    print("-" * 80)

    # Valid configuration
    valid_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )
    is_valid, errors = EmbeddingProviderRegistry.validate_provider_config(valid_config)
    print(f"\n  Configuration 1: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"    Provider: {valid_config.model_type.value}")
    print(f"    Model: {valid_config.model_name}")
    if errors:
        print(f"    Errors: {errors}")

    # Invalid configuration (missing API key for OpenAI)
    invalid_config = EmbeddingConfig(
        model_type=EmbeddingModelType.OPENAI,
        model_name="text-embedding-3-small",
        # Missing API key
    )
    is_valid, errors = EmbeddingProviderRegistry.validate_provider_config(
        invalid_config
    )
    print(f"\n  Configuration 2: {'✓ VALID' if is_valid else '✗ INVALID'}")
    print(f"    Provider: {invalid_config.model_type.value}")
    print(f"    Model: {invalid_config.model_name}")
    if errors:
        for error in errors:
            print(f"    Error: {error}")

    # Example 5: Using the unified interface (local model)
    print("\n\n5. USING THE UNIFIED INTERFACE - LOCAL MODEL")
    print("-" * 80)

    # Create a local embedding service (doesn't require API key)
    local_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
    )

    print(f"\n  Creating service: {local_config.model_type.value}")
    print(f"  Model: {local_config.model_name}")

    service = EmbeddingServiceFactory.create_service(local_config)

    # All services implement the same interface methods:
    print(f"\n  Service Methods Available:")
    print(f"    - embed_texts(texts: list[str]) -> EmbeddingResult")
    print(f"    - embed_single(text: str) -> list[float]")
    print(f"    - get_embedding_dimension() -> int")
    print(f"    - health_check() -> bool")
    print(f"    - get_model_info() -> dict")

    # Get model information
    model_info = service.get_model_info()
    print(f"\n  Model Information:")
    print(f"    Embedding Dimension: {model_info['embedding_dimension']}")
    print(f"    Service Type: {model_info['service_type']}")

    # Generate a single embedding
    print(f"\n  Generating single embedding...")
    single_embedding = service.embed_single(texts[0])
    print(f"    Result: Vector of dimension {len(single_embedding)}")
    print(f"    First 5 values: {single_embedding[:5]}")

    # Generate batch embeddings
    print(f"\n  Generating batch embeddings for {len(texts)} texts...")
    batch_result = service.embed_texts(texts)
    print(f"    Generated: {batch_result.count} embeddings")
    print(f"    Dimension: {batch_result.embedding_dimension}")
    print(f"    Processing Time: {batch_result.processing_time:.4f}s")
    print(f"    Throughput: {batch_result.texts_per_second:.2f} texts/sec")

    # Health check
    print(f"\n  Performing health check...")
    is_healthy = service.health_check()
    print(f"    Status: {'✓ Healthy' if is_healthy else '✗ Unhealthy'}")

    # Example 6: Provider comparison
    print("\n\n6. PROVIDER FEATURE COMPARISON")
    print("-" * 80)
    print(f"\n  {'Provider':<25} {'Type':<8} {'API Key':<10} {'Batch':<8} {'Avg Dim'}")
    print(f"  {'-'*25} {'-'*8} {'-'*10} {'-'*8} {'-'*8}")

    for provider in providers:
        info = EmbeddingProviderRegistry.get_provider_info(
            EmbeddingModelType(provider["type"])
        )
        provider_type = "Local" if provider["is_local"] else "API"
        api_key = "Required" if provider["requires_api_key"] else "No"
        batch = "Yes" if info.get("supports_batch") else "No"
        avg_dim = sum(info["typical_dimensions"]) // len(info["typical_dimensions"])

        print(
            f"  {provider['name']:<25} {provider_type:<8} {api_key:<10} {batch:<8} {avg_dim}"
        )

    print("\n\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    print("\nKey Takeaways:")
    print("  1. All providers implement the same EmbeddingService interface")
    print("  2. Easy to switch between providers by changing configuration")
    print("  3. Comprehensive validation ensures correct setup")
    print("  4. Provider registry provides discovery and recommendations")
    print("  5. Unified interface enables consistent application code")


if __name__ == "__main__":
    demonstrate_unified_interface()
