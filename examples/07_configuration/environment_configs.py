"""
Environment Configuration Example

This example demonstrates how to manage configurations across different environments:
- Development environment setup
- Testing environment configuration
- Production environment patterns
- Configuration inheritance and overrides
- Secrets management

Prerequisites:
- PDF Vector System installed
- Understanding of environment variables
- Access to different deployment environments

Usage:
    python environment_configs.py

Expected Output:
    - Environment-specific configuration examples
    - Configuration inheritance patterns
    - Secrets management demonstrations
    - Best practices for each environment

Learning Objectives:
- Master environment-specific configuration
- Learn configuration inheritance patterns
- Understand secrets management
- See deployment-ready configuration examples
"""

import contextlib
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from examples.utils.example_helpers import (
    example_context,
    print_section,
    print_subsection,
)

from vectorflow import Config
from vectorflow.core.config.settings import EmbeddingModelType, LogLevel

# Add parent directory to path for imports
sys.path.append(str(Path(__file__).parent.parent))


class Environment(Enum):
    """Supported deployment environments."""

    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class EnvironmentConfig:
    """Configuration template for a specific environment."""

    name: str
    description: str
    embedding_model_type: EmbeddingModelType
    embedding_model_name: str
    batch_size: int
    chunk_size: int
    chunk_overlap: int
    max_workers: int
    log_level: LogLevel
    debug: bool
    collection_suffix: str
    persist_directory: str
    additional_settings: dict[str, Any] = field(default_factory=dict)


def get_environment_templates() -> dict[Environment, EnvironmentConfig]:
    """Get configuration templates for each environment."""
    return {
        Environment.DEVELOPMENT: EnvironmentConfig(
            name="Development",
            description="Fast iteration with local models and verbose logging",
            embedding_model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            embedding_model_name="all-MiniLM-L6-v2",
            batch_size=16,
            chunk_size=800,
            chunk_overlap=100,
            max_workers=2,
            log_level=LogLevel.DEBUG,
            debug=True,
            collection_suffix="dev",
            persist_directory="./dev_chroma_db",
            additional_settings={
                "create_sample_data": True,
                "enable_profiling": True,
                "auto_reload": True,
            },
        ),
        Environment.TESTING: EnvironmentConfig(
            name="Testing",
            description="Minimal resources for fast, deterministic test execution",
            embedding_model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            embedding_model_name="all-MiniLM-L6-v2",
            batch_size=8,
            chunk_size=500,
            chunk_overlap=50,
            max_workers=1,
            log_level=LogLevel.INFO,
            debug=True,
            collection_suffix="test",
            persist_directory="./test_chroma_db",
            additional_settings={
                "cleanup_after_test": True,
                "deterministic_mode": True,
                "timeout_seconds": 30,
            },
        ),
        Environment.STAGING: EnvironmentConfig(
            name="Staging",
            description="Production-like environment for final testing",
            embedding_model_type=EmbeddingModelType.OPENAI,
            embedding_model_name="text-embedding-3-small",
            batch_size=50,
            chunk_size=1000,
            chunk_overlap=150,
            max_workers=4,
            log_level=LogLevel.INFO,
            debug=False,
            collection_suffix="staging",
            persist_directory="/data/staging_chroma_db",
            additional_settings={
                "enable_monitoring": True,
                "backup_enabled": True,
                "rate_limiting": True,
            },
        ),
        Environment.PRODUCTION: EnvironmentConfig(
            name="Production",
            description="Optimized for performance, reliability, and security",
            embedding_model_type=EmbeddingModelType.OPENAI,
            embedding_model_name="text-embedding-3-small",
            batch_size=100,
            chunk_size=1200,
            chunk_overlap=200,
            max_workers=8,
            log_level=LogLevel.WARNING,
            debug=False,
            collection_suffix="prod",
            persist_directory="/data/production_chroma_db",
            additional_settings={
                "enable_monitoring": True,
                "backup_enabled": True,
                "rate_limiting": True,
                "security_scanning": True,
                "performance_tracking": True,
            },
        ),
    }


def create_config_from_template(
    template: EnvironmentConfig, overrides: Optional[dict[str, Any]] = None
) -> Config:
    """Create a Config object from an environment template."""
    config = Config()

    # Apply template settings
    config.embedding.model_type = template.embedding_model_type
    config.embedding.model_name = template.embedding_model_name
    config.embedding.batch_size = template.batch_size

    config.text_processing.chunk_size = template.chunk_size
    config.text_processing.chunk_overlap = template.chunk_overlap

    config.chroma_db.collection_name = f"documents_{template.collection_suffix}"
    config.chroma_db.persist_directory = Path(template.persist_directory)

    config.logging.level = template.log_level
    config.debug = template.debug
    config.max_workers = template.max_workers

    # Apply overrides if provided
    if overrides:
        for key, value in overrides.items():
            if hasattr(config, key):
                setattr(config, key, value)
            elif "." in key:
                # Handle nested attributes like 'embedding.batch_size'
                parts = key.split(".")
                obj = config
                for part in parts[:-1]:
                    obj = getattr(obj, part)
                setattr(obj, parts[-1], value)

    return config


def demonstrate_environment_configs() -> None:
    """Demonstrate configuration for each environment."""
    print_subsection("Environment-Specific Configurations")

    templates = get_environment_templates()

    for env, template in templates.items():
        print(f"\nEnvironment: {env.value}")
        print(f"  Name: {template.name}")
        print(f"  Description: {template.description}")
        print(
            "  Embedding: "
            f"{template.embedding_model_type.value} ({template.embedding_model_name})"
        )
        print(
            f"  Batch size: {template.batch_size}, "
            f"chunk size/overlap: {template.chunk_size}/{template.chunk_overlap}"
        )
        print(
            f"  Collection suffix: {template.collection_suffix}, "
            f"persist dir: {template.persist_directory}"
        )
        print(f"  Max workers: {template.max_workers}, debug: {template.debug}")

        if template.additional_settings:
            print("  Additional settings:")
            for key, value in template.additional_settings.items():
                print(f"    - {key} = {value}")


def demonstrate_configuration_inheritance() -> None:
    """Demonstrate configuration inheritance patterns."""
    print_subsection("Configuration Inheritance")

    # Base configuration (common settings)
    base_config = {
        "text_processing.min_chunk_size": 100,
        "chroma_db.max_results": 20,
        "embedding.max_retries": 3,
        "embedding.timeout_seconds": 60,
    }

    print("\nBase configuration (shared across environments):")
    for key, value in base_config.items():
        print(f"  {key} = {value}")

    # Environment-specific overrides
    environment_overrides = {
        Environment.DEVELOPMENT: {
            "embedding.timeout_seconds": 30,  # Shorter timeout for dev
            "debug": True,
        },
        Environment.TESTING: {
            "embedding.timeout_seconds": 15,  # Very short timeout for tests
            "chroma_db.max_results": 5,  # Fewer results for faster tests
        },
        Environment.PRODUCTION: {
            "embedding.max_retries": 5,  # More retries for production
            "embedding.timeout_seconds": 120,  # Longer timeout for reliability
        },
    }

    templates = get_environment_templates()

    for env, overrides in environment_overrides.items():
        template = templates[env]

        # Combine base config with environment overrides
        combined_overrides = {**base_config, **overrides}

        print(f"\nOverrides for {env.value} environment:")
        for key, value in overrides.items():
            print(f"  {key} -> {value}")

        # Create config with inheritance
        create_config_from_template(template, combined_overrides)


def demonstrate_secrets_management() -> None:
    """Demonstrate secure secrets management."""
    print_subsection("Secrets Management")

    # Environment variable patterns
    secrets_env_vars = [
        "OPENAI_API_KEY",
        "COHERE_API_KEY",
        "GOOGLE_GEMINI_API_KEY",
        "DATABASE_PASSWORD",
        "ENCRYPTION_KEY",
    ]

    for var in secrets_env_vars:
        value = os.environ.get(var)
        if value:
            masked = f"{value[:8]}..." if len(value) > 8 else "***"
            print(f"  {var}: set ({masked})")
        else:
            print(f"  {var}: NOT set")

    # Configuration file patterns
    security_practices = [
        "Never commit .env files with real secrets",
        "Use .env.example as a template",
        "Set restrictive file permissions (600)",
        "Use different secrets for each environment",
        "Rotate secrets regularly",
        "Use secret management services in production",
    ]

    print("\nSecret management best practices:")
    for practice in security_practices:
        print(f"  - {practice}")

    # Demonstrate secure config loading

    def load_secure_config(environment: Environment) -> Config:
        """Load configuration with secure secret handling."""
        templates = get_environment_templates()
        template = templates[environment]

        # Create base config
        config = create_config_from_template(template)

        # Handle API keys securely
        if template.embedding_model_type == EmbeddingModelType.OPENAI:
            api_key = os.environ.get("OPENAI_API_KEY")
            if not api_key:
                # Fallback to local model
                config.embedding.model_type = EmbeddingModelType.SENTENCE_TRANSFORMERS
                config.embedding.model_name = "all-MiniLM-L6-v2"
            else:
                # API key is present; we keep OpenAI configuration
                masked = f"{api_key[:8]}..." if len(api_key) > 8 else "***"
                print(
                    f"  Using OpenAI embeddings for {environment.value} "
                    f"with key {masked}"
                )

        return config

    # Test secure loading for each environment
    for env in Environment:
        with contextlib.suppress(Exception):
            load_secure_config(env)


def demonstrate_deployment_patterns() -> None:
    """Demonstrate deployment-specific configuration patterns."""
    print_subsection("Deployment Patterns")

    # Container deployment
    container_env_vars = [
        "ENVIRONMENT=production",
        "EMBEDDING__MODEL_TYPE=openai",
        "EMBEDDING__MODEL_NAME=text-embedding-3-small",
        "CHROMA_DB__PERSIST_DIRECTORY=/data/chroma_db",
        "LOGGING__FILE_PATH=/var/log/pdf_vector_system.log",
        "MAX_WORKERS=8",
    ]

    print("\nContainer deployment environment variables:")
    for var in container_env_vars:
        print(f"  - {var}")

    # Kubernetes deployment
    k8s_patterns = [
        "Use ConfigMaps for non-sensitive configuration",
        "Use Secrets for API keys and passwords",
        "Mount configuration as volumes",
        "Use environment-specific namespaces",
        "Implement health checks and readiness probes",
    ]

    print("\nKubernetes deployment patterns:")
    for pattern in k8s_patterns:
        print(f"  - {pattern}")

    # Cloud deployment
    cloud_patterns = [
        "Use cloud secret managers (AWS Secrets Manager, Azure Key Vault)",
        "Implement IAM roles for service authentication",
        "Use managed databases when possible",
        "Configure auto-scaling based on load",
        "Set up monitoring and alerting",
    ]

    print("\nCloud deployment patterns:")
    for pattern in cloud_patterns:
        print(f"  - {pattern}")


def main() -> None:
    """
    Demonstrate environment-specific configuration management.

    This function shows how to manage configurations across
    different deployment environments securely and efficiently.
    """
    with example_context("Environment Configuration"):
        print_section("Environment Configuration Management")

        # Show environment-specific configurations
        demonstrate_environment_configs()

        # Show configuration inheritance
        demonstrate_configuration_inheritance()

        # Show secrets management
        demonstrate_secrets_management()

        # Show deployment patterns
        demonstrate_deployment_patterns()

        print_section("Environment Configuration Summary")


if __name__ == "__main__":
    main()
