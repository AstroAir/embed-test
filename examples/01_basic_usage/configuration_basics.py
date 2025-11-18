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

import contextlib
import os
import sys
from pathlib import Path

from examples.utils.example_helpers import (
    example_context,
    get_available_providers,
    print_section,
    print_subsection,
)

from vectorflow import Config
from vectorflow.core.config.settings import (
    EmbeddingConfig,
    EmbeddingModelType,
    LogLevel,
    TextProcessingConfig,
)

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


def demonstrate_basic_configuration() -> None:
    """Demonstrate basic configuration creation."""
    print_subsection("Basic Configuration")

    # Create default configuration
    config = Config()

    print("Created default configuration with:")
    print(
        f"  Embedding model: {config.embedding.model_type} ({config.embedding.model_name})"
    )
    print(f"  Chroma collection: {config.chroma_db.collection_name}")
    print(f"  Persist directory: {config.chroma_db.persist_directory}")
    print(f"  Debug mode: {config.debug}")


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

    print("Created programmatic configuration with:")
    print(
        f"  Embedding model: {config.embedding.model_type} ({config.embedding.model_name})"
    )
    print(f"  Batch size: {config.embedding.batch_size}")
    print(
        "  Chunk size/overlap: "
        f"{config.text_processing.chunk_size}/{config.text_processing.chunk_overlap}"
    )
    print(f"  Collection: {config.chroma_db.collection_name}")
    print(f"  Log level: {config.logging.level}")
    print(f"  Debug: {config.debug}, max_workers: {config.max_workers}")


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
        "MAX_WORKERS": "2",
    }

    # Temporarily set environment variables
    original_env = {}
    for key, value in env_vars.items():
        original_env[key] = os.environ.get(key)
        os.environ[key] = value

    try:
        # Create configuration (will read from environment)
        config = Config()

        print("Loaded configuration from environment variables:")
        for key, value in env_vars.items():
            print(f"  {key}={value}")

        print(
            "  Effective embedding model: "
            f"{config.embedding.model_type} ({config.embedding.model_name})"
        )
        print(f"  Batch size: {config.embedding.batch_size}")
        print(
            "  Chunk size/overlap: "
            f"{config.text_processing.chunk_size}/{config.text_processing.chunk_overlap}"
        )
        print(f"  Collection: {config.chroma_db.collection_name}")
        print(f"  Debug: {config.debug}, max_workers: {config.max_workers}")

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

    for line in env_content.split("\n"):
        if line.strip() and not line.startswith("#"):
            key, _, value = line.partition("=")
            print(f"  {key.strip()}={value.strip()}")

    # Clean up
    print(f"Created sample .env file at: {env_file}")
    print(
        "In real projects you would keep this file and load it via your configuration system."
    )
    env_file.unlink()


def demonstrate_provider_specific_configuration() -> None:
    """Demonstrate configuration for different embedding providers."""
    print_subsection("Provider-Specific Configuration")

    providers = get_available_providers()

    # Sentence Transformers (always available)
    st_config = EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=32,
    )
    print("Sentence Transformers (local) configuration:")
    print(f"  Model: {st_config.model_name}, batch_size={st_config.batch_size}")

    # OpenAI (if API key available)
    if providers.get("openai"):
        openai_config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            batch_size=100,
            max_retries=3,
            timeout_seconds=60,
        )
        print("OpenAI configuration:")
        print(
            f"  Model: {openai_config.model_name}, "
            f"batch_size={openai_config.batch_size}"
        )
    else:
        print(
            "OpenAI provider not available (OPENAI_API_KEY not set). Skipping OpenAI example."
        )

    # Cohere (if API key available)
    if providers.get("cohere"):
        cohere_config = EmbeddingConfig(
            model_type=EmbeddingModelType.COHERE,
            model_name="embed-english-v3.0",
            batch_size=48,
        )
        print("Cohere configuration:")
        print(
            f"  Model: {cohere_config.model_name}, "
            f"batch_size={cohere_config.batch_size}"
        )
    else:
        print(
            "Cohere provider not available (COHERE_API_KEY not set). Skipping Cohere example."
        )


def demonstrate_configuration_validation() -> None:
    """Demonstrate configuration validation."""
    print_subsection("Configuration Validation")

    # Valid configuration
    try:
        EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=32,
            max_retries=3,
        )
        print("Valid configuration created successfully.")
    except Exception as e:
        print(f"Unexpected error for valid configuration: {e}")

    # Invalid batch size
    try:
        EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=0,  # Invalid: must be positive
        )
        print("Error: invalid batch size configuration unexpectedly succeeded.")
    except Exception as e:
        print(f"Correctly failed invalid batch size: {e}")

    # Invalid chunk size
    try:
        TextProcessingConfig(chunk_size=0)  # Invalid: must be positive
        print("Error: invalid chunk size configuration unexpectedly succeeded.")
    except Exception as e:
        print(f"Correctly failed invalid chunk size: {e}")


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
            "description": "Fast local models, verbose logging",
        },
        "testing": {
            "embedding_model": "sentence-transformers",
            "model_name": "all-MiniLM-L6-v2",
            "batch_size": 8,
            "debug": True,
            "workers": 1,
            "description": "Minimal resources, fast execution",
        },
        "production": {
            "embedding_model": "openai",
            "model_name": "text-embedding-3-small",
            "batch_size": 100,
            "debug": False,
            "workers": 8,
            "description": "High quality, optimized performance",
        },
    }

    for env_name, env_config in environments.items():
        print(f"\nEnvironment: {env_name}")
        print(
            "  Embedding model: "
            f"{env_config['embedding_model']} ({env_config['model_name']})"
        )
        print(
            f"  Batch size: {env_config['batch_size']}, "
            f"workers: {env_config['workers']}"
        )
        print(f"  Debug: {env_config['debug']}")
        print(f"  Description: {env_config['description']}")


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


if __name__ == "__main__":
    main()
