"""
Configuration Basics Example

This example demonstrates different ways to configure the PDF Vector System:
- Environment variables
- Configuration objects
- .env files
- Programmatic configuration

Prerequisites:
- PDF Vector System installed
- Understanding of environment variables

Usage:
    python configuration_basics.py

Expected Output:
    - Different configuration methods demonstrated
    - Configuration validation examples
    - Best practices for different environments

Learning Objectives:
- Learn all configuration methods
- Understand configuration validation
- See environment-specific patterns
- Learn security best practices
"""

import os
import sys
from pathlib import Path
from typing import Dict, Any

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from pdf_vector_system import Config
from pdf_vector_system.config.settings import (
    EmbeddingModelType, LogLevel, EmbeddingConfig, 
    TextProcessingConfig, ChromaDBConfig, LoggingConfig
)
from utils.example_helpers import (
    print_section, print_subsection, example_context,
    check_api_key, get_available_providers
)


def demonstrate_basic_configuration() -> None:
    """Demonstrate basic configuration creation."""
    print_subsection("Basic Configuration")
    
    # Create default configuration
    config = Config()
    
    print("✅ Default configuration created")
    print(f"   - Embedding model: {config.embedding.model_type.value}")
    print(f"   - Model name: {config.embedding.model_name}")
    print(f"   - Batch size: {config.embedding.batch_size}")
    print(f"   - Chunk size: {config.text_processing.chunk_size}")
    print(f"   - Collection: {config.chroma_db.collection_name}")
    print(f"   - Debug mode: {config.debug}")


def demonstrate_programmatic_configuration() -> None:
    """Demonstrate programmatic configuration."""
    print_subsection("Programmatic Configuration")
    
    # Create configuration with custom settings
    config = Config()
    
    # Configure embedding settings
    config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
    config.embedding.model_name = "all-mpnet-base-v2"  # Higher quality model
    config.embedding.batch_size = 32
    config.embedding.max_retries = 5
    
    # Configure text processing
    config.text_processing.chunk_size = 1200
    config.text_processing.chunk_overlap = 200
    config.text_processing.min_chunk_size = 100
    
    # Configure ChromaDB
    config.chroma_db.collection_name = "custom_documents"
    config.chroma_db.persist_directory = Path("./custom_chroma_db")
    config.chroma_db.max_results = 20
    
    # Configure logging
    config.logging.level = LogLevel.INFO
    config.logging.file_path = Path("./logs/custom_pipeline.log")
    
    # Global settings
    config.debug = False
    config.max_workers = 4
    
    print("✅ Programmatic configuration created")
    print(f"   - Embedding model: {config.embedding.model_name}")
    print(f"   - Chunk size: {config.text_processing.chunk_size}")
    print(f"   - Collection: {config.chroma_db.collection_name}")
    print(f"   - Log level: {config.logging.level.value}")
    print(f"   - Max workers: {config.max_workers}")


def demonstrate_environment_configuration() -> None:
    """Demonstrate environment variable configuration."""
    print_subsection("Environment Variable Configuration")
    
    # Set some environment variables for demonstration
    env_vars = {
        "EMBEDDING__MODEL_TYPE": "sentence-transformers",
        "EMBEDDING__MODEL_NAME": "all-MiniLM-L6-v2",
        "EMBEDDING__BATCH_SIZE": "16",
        "TEXT_PROCESSING__CHUNK_SIZE": "800",
        "TEXT_PROCESSING__CHUNK_OVERLAP": "100",
        "CHROMA_DB__COLLECTION_NAME": "env_documents",
        "DEBUG": "true",
        "MAX_WORKERS": "2"
    }
    
    # Temporarily set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value
    
    try:
        # Create configuration (will read from environment)
        config = Config()
        
        print("✅ Environment configuration loaded")
        print(f"   - Model type: {config.embedding.model_type.value}")
        print(f"   - Model name: {config.embedding.model_name}")
        print(f"   - Batch size: {config.embedding.batch_size}")
        print(f"   - Chunk size: {config.text_processing.chunk_size}")
        print(f"   - Collection: {config.chroma_db.collection_name}")
        print(f"   - Debug: {config.debug}")
        print(f"   - Workers: {config.max_workers}")
        
    finally:
        # Restore original environment
        for key, original_value in original_env.items():
            if original_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = original_value


def demonstrate_dotenv_configuration() -> None:
    """Demonstrate .env file configuration."""
    print_subsection(".env File Configuration")
    
    # Create a sample .env file
    env_content = """
# Sample .env configuration
EMBEDDING__MODEL_TYPE=sentence-transformers
EMBEDDING__MODEL_NAME=all-MiniLM-L6-v2
EMBEDDING__BATCH_SIZE=24

TEXT_PROCESSING__CHUNK_SIZE=1000
TEXT_PROCESSING__CHUNK_OVERLAP=150

CHROMA_DB__COLLECTION_NAME=dotenv_documents
CHROMA_DB__PERSIST_DIRECTORY=./dotenv_chroma_db

LOGGING__LEVEL=INFO
LOGGING__FILE_PATH=./logs/dotenv_pipeline.log

DEBUG=false
MAX_WORKERS=3
""".strip()
    
    # Write sample .env file
    env_file = Path("example.env")
    env_file.write_text(env_content)
    
    print(f"✅ Created sample .env file: {env_file}")
    print("   Content:")
    for line in env_content.split('\n'):
        if line.strip() and not line.startswith('#'):
            print(f"     {line}")
    
    print()
    print("   To use this configuration:")
    print("   1. Copy example.env to .env")
    print("   2. Modify values as needed")
    print("   3. Create Config() - it will automatically load .env")
    
    # Clean up
    env_file.unlink()


def demonstrate_provider_specific_configuration() -> None:
    """Demonstrate configuration for different embedding providers."""
    print_subsection("Provider-Specific Configuration")
    
    providers = get_available_providers()
    
    # Sentence Transformers (always available)
    print("🤖 Sentence Transformers Configuration:")
    st_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=32
    )
    print(f"   - Model: {st_config.model_name}")
    print(f"   - Batch size: {st_config.batch_size}")
    print(f"   - Available: ✅")
    
    # OpenAI (if API key available)
    print("\n🔑 OpenAI Configuration:")
    if providers.get("openai"):
        openai_config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            batch_size=100,
            max_retries=3,
            timeout_seconds=60
        )
        print(f"   - Model: {openai_config.model_name}")
        print(f"   - Batch size: {openai_config.batch_size}")
        print(f"   - Available: ✅ (API key found)")
    else:
        print(f"   - Available: ❌ (Set OPENAI_API_KEY environment variable)")
    
    # Cohere (if API key available)
    print("\n🌟 Cohere Configuration:")
    if providers.get("cohere"):
        print(f"   - Available: ✅ (API key found)")
        print(f"   - Recommended model: embed-english-v3.0")
        print(f"   - Recommended batch size: 48")
    else:
        print(f"   - Available: ❌ (Set COHERE_API_KEY environment variable)")


def demonstrate_configuration_validation() -> None:
    """Demonstrate configuration validation."""
    print_subsection("Configuration Validation")
    
    print("✅ Valid configurations:")
    
    # Valid configuration
    try:
        valid_config = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            max_retries=3
        )
        print(f"   - Embedding config: {valid_config.model_name}")
    except Exception as e:
        print(f"   ❌ Unexpected error: {e}")
    
    print("\n❌ Invalid configurations:")
    
    # Invalid batch size
    try:
        invalid_config = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=0  # Invalid: must be positive
        )
        print(f"   - This should not print: {invalid_config.batch_size}")
    except Exception as e:
        print(f"   - Invalid batch size (0): {type(e).__name__}")
    
    # Invalid chunk size
    try:
        invalid_text_config = TextProcessingConfig(
            chunk_size=0  # Invalid: must be positive
        )
        print(f"   - This should not print: {invalid_text_config.chunk_size}")
    except Exception as e:
        print(f"   - Invalid chunk size (0): {type(e).__name__}")


def demonstrate_environment_specific_patterns() -> None:
    """Demonstrate patterns for different environments."""
    print_subsection("Environment-Specific Patterns")
    
    environments = {
        "development": {
            "embedding_model": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 16,
            "debug": True,
            "workers": 2,
            "description": "Fast local models, verbose logging"
        },
        "testing": {
            "embedding_model": "sentence-transformers", 
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 8,
            "debug": True,
            "workers": 1,
            "description": "Minimal resources, fast execution"
        },
        "production": {
            "embedding_model": "openai",
            "model_name": "text-embedding-3-small",
            "batch_size": 100,
            "debug": False,
            "workers": 8,
            "description": "High quality, optimized performance"
        }
    }
    
    for env_name, env_config in environments.items():
        print(f"\n📋 {env_name.title()} Environment:")
        print(f"   - Description: {env_config['description']}")
        print(f"   - Embedding: {env_config['embedding_model']}")
        print(f"   - Model: {env_config['model_name']}")
        print(f"   - Batch size: {env_config['batch_size']}")
        print(f"   - Debug: {env_config['debug']}")
        print(f"   - Workers: {env_config['workers']}")


def main() -> None:
    """
    Demonstrate various configuration methods and patterns.
    
    This function shows all the ways to configure the PDF Vector System
    and provides guidance on best practices for different scenarios.
    """
    with example_context("Configuration Basics"):
        
        print_section("Configuration Methods")
        
        # Demonstrate different configuration approaches
        demonstrate_basic_configuration()
        demonstrate_programmatic_configuration()
        demonstrate_environment_configuration()
        demonstrate_dotenv_configuration()
        
        print_section("Provider Configuration")
        demonstrate_provider_specific_configuration()
        
        print_section("Validation and Best Practices")
        demonstrate_configuration_validation()
        demonstrate_environment_specific_patterns()
        
        print_section("Configuration Summary")
        
        print("🎯 Key Takeaways:")
        print()
        print("1. 📝 Configuration Methods:")
        print("   - Default: Config() uses sensible defaults")
        print("   - Programmatic: Set properties directly in code")
        print("   - Environment: Use environment variables")
        print("   - .env files: Store configuration in files")
        print()
        print("2. 🔧 Best Practices:")
        print("   - Use environment variables for production")
        print("   - Use .env files for development")
        print("   - Validate configuration early")
        print("   - Keep API keys secure")
        print()
        print("3. 🌍 Environment Patterns:")
        print("   - Development: Local models, verbose logging")
        print("   - Testing: Minimal resources, fast execution")
        print("   - Production: High quality, optimized performance")
        print()
        print("4. 🔐 Security:")
        print("   - Never commit API keys to version control")
        print("   - Use environment variables for secrets")
        print("   - Set appropriate file permissions on .env files")
        print()
        print("Next steps:")
        print("- Try first_search.py to see configuration in action")
        print("- Explore examples/sample_data/config_templates/ for templates")
        print("- Check the embedding providers examples for advanced configuration")


if __name__ == "__main__":
    main()
