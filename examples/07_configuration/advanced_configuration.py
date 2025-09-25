"""
Advanced Configuration Example

This example demonstrates sophisticated configuration patterns for the PDF Vector System:
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
    python advanced_configuration.py

Expected Output:
    - Advanced configuration patterns demonstrated
    - Provider fallback mechanisms
    - Performance optimization examples
    - Configuration validation results

Learning Objectives:
- Master advanced configuration patterns
- Learn provider fallback strategies
- Understand performance optimization
- See production-ready configuration examples
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config, PDFVectorPipeline
from pdf_vector_system.config.settings import (
    EmbeddingModelType, LogLevel, EmbeddingConfig,
    TextProcessingConfig, ChromaDBConfig, LoggingConfig
)
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    check_api_key, get_available_providers
)


@dataclass
class ProviderConfig:
    """Configuration for a specific embedding provider."""
    model_type: EmbeddingModelType
    model_name: str
    batch_size: int
    description: str
    requires_api_key: bool = False
    api_key_env: Optional[str] = None


def get_provider_configurations() -> List[ProviderConfig]:
    """Get configurations for all supported providers."""
    return [
        ProviderConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            description="Fast, lightweight local model",
            requires_api_key=False
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-mpnet-base-v2",
            batch_size=16,
            description="High-quality local model",
            requires_api_key=False
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            batch_size=100,
            description="OpenAI's efficient embedding model",
            requires_api_key=True,
            api_key_env="OPENAI_API_KEY"
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-large",
            batch_size=50,
            description="OpenAI's high-quality embedding model",
            requires_api_key=True,
            api_key_env="OPENAI_API_KEY"
        ),
        ProviderConfig(
            model_type=EmbeddingModelType.COHERE,
            model_name="embed-english-v3.0",
            batch_size=48,
            description="Cohere's multilingual embedding model",
            requires_api_key=True,
            api_key_env="COHERE_API_KEY"
        )
    ]


def demonstrate_provider_fallback() -> None:
    """Demonstrate automatic provider fallback configuration."""
    print_subsection("Provider Fallback Configuration")
    
    provider_configs = get_provider_configurations()
    available_providers = get_available_providers()
    
    print("🔄 Configuring provider fallback chain...")
    
    # Define preferred order (best to fallback)
    preferred_order = [
        EmbeddingModelType.OPENAI,
        EmbeddingModelType.COHERE,
        EmbeddingModelType.SENTENCE_TRANSFORMERS
    ]
    
    selected_config = None
    
    for preferred_type in preferred_order:
        # Find configurations for this provider type
        type_configs = [c for c in provider_configs if c.model_type == preferred_type]
        
        for config in type_configs:
            provider_key = config.model_type.value.replace('-', '_')
            
            if config.requires_api_key:
                if available_providers.get(provider_key, False):
                    print(f"✅ {config.description} - API key available")
                    selected_config = config
                    break
                else:
                    print(f"⚠️  {config.description} - API key missing")
            else:
                print(f"✅ {config.description} - Local model available")
                selected_config = config
                break
        
        if selected_config:
            break
    
    if selected_config:
        print(f"\n🎯 Selected: {selected_config.description}")
        
        # Create configuration with selected provider
        config = Config()
        config.embedding.model_type = selected_config.model_type
        config.embedding.model_name = selected_config.model_name
        config.embedding.batch_size = selected_config.batch_size
        
        print(f"   - Model: {config.embedding.model_name}")
        print(f"   - Batch size: {config.embedding.batch_size}")
        
        return config
    else:
        print("❌ No available providers found")
        return None


def demonstrate_environment_specific_configs() -> Dict[str, Config]:
    """Demonstrate environment-specific configurations."""
    print_subsection("Environment-Specific Configurations")
    
    configs = {}
    
    # Development configuration
    print("🔧 Development Configuration:")
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
    
    print(f"   - Fast local model for quick iteration")
    print(f"   - Small chunks for faster processing")
    print(f"   - Debug logging enabled")
    print(f"   - Limited workers to avoid resource contention")
    
    configs["development"] = dev_config
    
    # Testing configuration
    print("\n🧪 Testing Configuration:")
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
    
    print(f"   - Minimal resources for fast test execution")
    print(f"   - Small batch sizes for predictable behavior")
    print(f"   - Single worker for deterministic results")
    
    configs["testing"] = test_config
    
    # Production configuration
    print("\n🚀 Production Configuration:")
    prod_config = Config()
    
    # Use best available provider for production
    available_providers = get_available_providers()
    if available_providers.get("openai"):
        prod_config.embedding.model_type = EmbeddingModelType.OPENAI
        prod_config.embedding.model_name = "text-embedding-3-small"
        prod_config.embedding.batch_size = 100
        print(f"   - Using OpenAI for high-quality embeddings")
    else:
        prod_config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
        prod_config.embedding.model_name = "all-mpnet-base-v2"
        prod_config.embedding.batch_size = 32
        print(f"   - Using high-quality local model")
    
    prod_config.text_processing.chunk_size = 1200
    prod_config.text_processing.chunk_overlap = 200
    prod_config.chroma_db.collection_name = "production_documents"
    prod_config.chroma_db.persist_directory = Path("/data/chroma_db")
    prod_config.logging.level = LogLevel.INFO
    prod_config.debug = False
    prod_config.max_workers = 8
    
    print(f"   - Optimized chunk sizes for quality")
    print(f"   - Production logging level")
    print(f"   - Maximum workers for throughput")
    
    configs["production"] = prod_config
    
    return configs


def demonstrate_performance_optimization() -> None:
    """Demonstrate performance optimization configurations."""
    print_subsection("Performance Optimization")
    
    print("⚡ Performance Optimization Patterns:")
    
    # High-throughput configuration
    print("\n🚀 High-Throughput Configuration:")
    throughput_config = Config()
    throughput_config.embedding.batch_size = 200  # Large batches
    throughput_config.text_processing.chunk_size = 1500  # Larger chunks
    throughput_config.max_workers = 16  # Many workers
    
    print(f"   - Large batch size: {throughput_config.embedding.batch_size}")
    print(f"   - Large chunks: {throughput_config.text_processing.chunk_size}")
    print(f"   - Many workers: {throughput_config.max_workers}")
    print(f"   - Best for: Bulk processing large document sets")
    
    # Low-latency configuration
    print("\n⚡ Low-Latency Configuration:")
    latency_config = Config()
    latency_config.embedding.batch_size = 16  # Small batches
    latency_config.text_processing.chunk_size = 600  # Smaller chunks
    latency_config.max_workers = 4  # Moderate workers
    
    print(f"   - Small batch size: {latency_config.embedding.batch_size}")
    print(f"   - Small chunks: {latency_config.text_processing.chunk_size}")
    print(f"   - Moderate workers: {latency_config.max_workers}")
    print(f"   - Best for: Interactive applications, real-time processing")
    
    # Memory-optimized configuration
    print("\n🧠 Memory-Optimized Configuration:")
    memory_config = Config()
    memory_config.embedding.batch_size = 8  # Very small batches
    memory_config.text_processing.chunk_size = 400  # Small chunks
    memory_config.max_workers = 2  # Few workers
    
    print(f"   - Very small batches: {memory_config.embedding.batch_size}")
    print(f"   - Small chunks: {memory_config.text_processing.chunk_size}")
    print(f"   - Few workers: {memory_config.max_workers}")
    print(f"   - Best for: Resource-constrained environments")


def demonstrate_dynamic_configuration() -> None:
    """Demonstrate dynamic configuration switching."""
    print_subsection("Dynamic Configuration Switching")
    
    print("🔄 Dynamic Configuration Based on Conditions:")
    
    # Simulate different conditions
    conditions = {
        "document_count": 100,
        "available_memory_gb": 8,
        "has_gpu": False,
        "is_production": False
    }
    
    print(f"📊 Current conditions:")
    for key, value in conditions.items():
        print(f"   - {key}: {value}")
    
    # Create adaptive configuration
    config = Config()
    
    # Adjust based on document count
    if conditions["document_count"] > 1000:
        config.embedding.batch_size = 100
        config.max_workers = 8
        print(f"\n📈 Large document set detected:")
        print(f"   - Increased batch size to {config.embedding.batch_size}")
        print(f"   - Increased workers to {config.max_workers}")
    elif conditions["document_count"] > 100:
        config.embedding.batch_size = 50
        config.max_workers = 4
        print(f"\n📊 Medium document set detected:")
        print(f"   - Set batch size to {config.embedding.batch_size}")
        print(f"   - Set workers to {config.max_workers}")
    else:
        config.embedding.batch_size = 16
        config.max_workers = 2
        print(f"\n📉 Small document set detected:")
        print(f"   - Set batch size to {config.embedding.batch_size}")
        print(f"   - Set workers to {config.max_workers}")
    
    # Adjust based on available memory
    if conditions["available_memory_gb"] < 4:
        config.text_processing.chunk_size = 400
        config.embedding.batch_size = min(config.embedding.batch_size, 8)
        print(f"\n🧠 Low memory detected:")
        print(f"   - Reduced chunk size to {config.text_processing.chunk_size}")
        print(f"   - Limited batch size to {config.embedding.batch_size}")
    
    # Adjust based on environment
    if conditions["is_production"]:
        config.logging.level = LogLevel.WARNING
        config.debug = False
        print(f"\n🚀 Production environment:")
        print(f"   - Set logging to {config.logging.level.value}")
        print(f"   - Disabled debug mode")
    else:
        config.logging.level = LogLevel.DEBUG
        config.debug = True
        print(f"\n🔧 Development environment:")
        print(f"   - Set logging to {config.logging.level.value}")
        print(f"   - Enabled debug mode")


def demonstrate_configuration_validation() -> None:
    """Demonstrate advanced configuration validation."""
    print_subsection("Configuration Validation")
    
    print("✅ Configuration Validation Examples:")
    
    # Valid configuration
    print("\n1. Valid Configuration:")
    try:
        valid_config = Config()
        valid_config.embedding.batch_size = 32
        valid_config.text_processing.chunk_size = 1000
        valid_config.text_processing.chunk_overlap = 200
        valid_config.max_workers = 4
        
        # Validate relationships
        if valid_config.text_processing.chunk_overlap >= valid_config.text_processing.chunk_size:
            raise ValueError("Chunk overlap must be less than chunk size")
        
        if valid_config.embedding.batch_size <= 0:
            raise ValueError("Batch size must be positive")
        
        print(f"   ✅ All validations passed")
        print(f"   - Batch size: {valid_config.embedding.batch_size}")
        print(f"   - Chunk size: {valid_config.text_processing.chunk_size}")
        print(f"   - Chunk overlap: {valid_config.text_processing.chunk_overlap}")
        
    except Exception as e:
        print(f"   ❌ Validation failed: {e}")
    
    # Invalid configurations
    print("\n2. Invalid Configurations:")
    
    invalid_configs = [
        {
            "name": "Negative batch size",
            "batch_size": -1,
            "chunk_size": 1000,
            "chunk_overlap": 200
        },
        {
            "name": "Overlap >= chunk size",
            "batch_size": 32,
            "chunk_size": 500,
            "chunk_overlap": 500
        },
        {
            "name": "Zero chunk size",
            "batch_size": 32,
            "chunk_size": 0,
            "chunk_overlap": 100
        }
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
            if config.text_processing.chunk_overlap >= config.text_processing.chunk_size:
                raise ValueError("Chunk overlap must be less than chunk size")
            
            print(f"   ⚠️  {invalid_config['name']}: Unexpectedly passed validation")
            
        except Exception as e:
            print(f"   ✅ {invalid_config['name']}: Correctly caught - {e}")


def main() -> None:
    """
    Demonstrate advanced configuration patterns for the PDF Vector System.
    
    This function shows sophisticated configuration techniques for
    production-ready deployments.
    """
    with example_context("Advanced Configuration"):
        
        print_section("Advanced Configuration Patterns")
        
        # Demonstrate provider fallback
        fallback_config = demonstrate_provider_fallback()
        
        # Demonstrate environment-specific configurations
        env_configs = demonstrate_environment_specific_configs()
        
        # Demonstrate performance optimization
        demonstrate_performance_optimization()
        
        # Demonstrate dynamic configuration
        demonstrate_dynamic_configuration()
        
        # Demonstrate configuration validation
        demonstrate_configuration_validation()
        
        print_section("Configuration Best Practices")
        
        print("🎯 Advanced Configuration Guidelines:")
        print()
        print("1. 🔄 Provider Fallback:")
        print("   - Always have a local model fallback")
        print("   - Check API key availability before using cloud providers")
        print("   - Consider cost and performance trade-offs")
        print()
        print("2. 🌍 Environment-Specific:")
        print("   - Development: Fast iteration, verbose logging")
        print("   - Testing: Minimal resources, deterministic behavior")
        print("   - Production: Optimized performance, appropriate logging")
        print()
        print("3. ⚡ Performance Optimization:")
        print("   - High throughput: Large batches, many workers")
        print("   - Low latency: Small batches, moderate workers")
        print("   - Memory constrained: Small batches, few workers")
        print()
        print("4. 🔄 Dynamic Configuration:")
        print("   - Adapt to runtime conditions")
        print("   - Monitor resource usage")
        print("   - Implement graceful degradation")
        print()
        print("5. ✅ Validation:")
        print("   - Validate configuration early")
        print("   - Check parameter relationships")
        print("   - Provide clear error messages")
        print()
        print("Next steps:")
        print("- Try pipeline_operations.py for advanced pipeline patterns")
        print("- Explore performance examples in 09_performance/")
        print("- Check production examples in 10_production/")


if __name__ == "__main__":
    main()
