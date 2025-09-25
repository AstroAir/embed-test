"""
Helper utilities for PDF Vector System examples.

This module provides common functionality used across multiple examples,
including configuration setup, logging, and utility functions.
"""

import os
import sys
from pathlib import Path
from typing import Optional, Dict, Any
from contextlib import contextmanager

from pdf_vector_system import Config
from pdf_vector_system.config.settings import EmbeddingModelType, LogLevel


def setup_example_environment(example_name: str = "example") -> Config:
    """
    Set up a standard example environment with configuration and logging.
    
    Args:
        example_name: Name of the example for logging context
        
    Returns:
        Configured Config object
    """
    # Load configuration
    config = Config()
    
    # Set up example-specific defaults
    config.debug = True
    config.logging.level = LogLevel.INFO
    
    # Create output directories
    create_example_directories()
    
    # Ensure sample data exists
    ensure_sample_data_exists()
    
    return config


def create_example_directories() -> None:
    """Create necessary directories for examples."""
    directories = [
        "example_outputs",
        "logs",
        "example_chroma_db"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)


def ensure_sample_data_exists() -> bool:
    """
    Ensure sample data directory exists and contains sample files.
    
    Returns:
        True if sample data is available, False otherwise
    """
    sample_data_dir = Path("examples/sample_data")
    
    if not sample_data_dir.exists():
        print("⚠️  Sample data directory not found. Creating...")
        sample_data_dir.mkdir(parents=True, exist_ok=True)
        return False
    
    # Check for PDF files
    pdf_files = list(sample_data_dir.glob("*.pdf"))
    if not pdf_files:
        print("📄 No sample PDF files found.")
        print(f"   Add PDF files to {sample_data_dir} or examples will create minimal test data.")
        return False
    
    return True


def print_section(title: str, width: int = 60) -> None:
    """
    Print a formatted section header.
    
    Args:
        title: Section title
        width: Width of the header line
    """
    print("\n" + "=" * width)
    print(f" {title}")
    print("=" * width)


def print_subsection(title: str, width: int = 40) -> None:
    """
    Print a formatted subsection header.
    
    Args:
        title: Subsection title
        width: Width of the header line
    """
    print(f"\n{title}")
    print("-" * width)


def format_bytes(bytes_value: int) -> str:
    """
    Format bytes into human-readable string.
    
    Args:
        bytes_value: Number of bytes
        
    Returns:
        Formatted string (e.g., "1.5 MB")
    """
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_value < 1024.0:
            return f"{bytes_value:.1f} {unit}"
        bytes_value /= 1024.0
    return f"{bytes_value:.1f} TB"


def format_duration(seconds: float) -> str:
    """
    Format duration into human-readable string.
    
    Args:
        seconds: Duration in seconds
        
    Returns:
        Formatted string (e.g., "2m 30s")
    """
    if seconds < 60:
        return f"{seconds:.1f}s"
    elif seconds < 3600:
        minutes = int(seconds // 60)
        remaining_seconds = seconds % 60
        return f"{minutes}m {remaining_seconds:.0f}s"
    else:
        hours = int(seconds // 3600)
        remaining_minutes = int((seconds % 3600) // 60)
        return f"{hours}h {remaining_minutes}m"


def check_api_key(provider: str) -> bool:
    """
    Check if API key is available for a provider.
    
    Args:
        provider: Provider name (openai, cohere, google_gemini)
        
    Returns:
        True if API key is available, False otherwise
    """
    key_mapping = {
        "openai": "OPENAI_API_KEY",
        "cohere": "COHERE_API_KEY",
        "google_gemini": "GOOGLE_GEMINI_API_KEY",
        "google_cloud": "GOOGLE_APPLICATION_CREDENTIALS"
    }
    
    env_var = key_mapping.get(provider.lower())
    if not env_var:
        return False
    
    return bool(os.getenv(env_var))


def get_available_providers() -> Dict[str, bool]:
    """
    Get list of available embedding providers.
    
    Returns:
        Dictionary mapping provider names to availability
    """
    providers = {
        "sentence_transformers": True,  # Always available
        "openai": check_api_key("openai"),
        "cohere": check_api_key("cohere"),
        "google_gemini": check_api_key("google_gemini"),
        "huggingface": True,  # Usually available
        "google_use": True,   # Usually available
    }
    
    return providers


def select_best_provider() -> tuple[EmbeddingModelType, str]:
    """
    Select the best available embedding provider.
    
    Returns:
        Tuple of (model_type, model_name)
    """
    providers = get_available_providers()
    
    # Preference order: OpenAI > Sentence Transformers > Others
    if providers.get("openai"):
        return EmbeddingModelType.OPENAI, "text-embedding-3-small"
    elif providers.get("sentence_transformers"):
        return EmbeddingModelType.SENTENCE_TRANSFORMERS, "all-MiniLM-L6-v2"
    elif providers.get("cohere"):
        return EmbeddingModelType.COHERE, "embed-english-v3.0"
    else:
        # Fallback to sentence transformers
        return EmbeddingModelType.SENTENCE_TRANSFORMERS, "all-MiniLM-L6-v2"


@contextmanager
def example_context(name: str):
    """
    Context manager for running examples with proper setup and cleanup.
    
    Args:
        name: Example name for logging
    """
    print_section(f"Running Example: {name}")
    
    try:
        yield
        print(f"\n✅ Example '{name}' completed successfully!")
        
    except KeyboardInterrupt:
        print(f"\n⚠️  Example '{name}' interrupted by user")
        
    except Exception as e:
        print(f"\n❌ Example '{name}' failed: {e}")
        raise
        
    finally:
        print(f"\n{'='*60}")


def safe_import(module_name: str, package: Optional[str] = None) -> Optional[Any]:
    """
    Safely import a module, returning None if not available.
    
    Args:
        module_name: Name of the module to import
        package: Package name for relative imports
        
    Returns:
        Imported module or None if not available
    """
    try:
        if package:
            return __import__(module_name, fromlist=[package])
        else:
            return __import__(module_name)
    except ImportError:
        return None


def create_sample_text_file(path: Path, content_type: str = "research") -> None:
    """
    Create a sample text file for testing.
    
    Args:
        path: Path where to create the file
        content_type: Type of content (research, technical, business)
    """
    content_templates = {
        "research": """
        Machine Learning and Artificial Intelligence Research

        Abstract
        This paper presents a comprehensive analysis of machine learning algorithms
        and their applications in artificial intelligence. We explore various
        approaches including supervised learning, unsupervised learning, and
        reinforcement learning paradigms.

        Introduction
        Machine learning has revolutionized the field of artificial intelligence
        by enabling computers to learn from data without being explicitly programmed.
        This capability has led to breakthroughs in numerous domains including
        computer vision, natural language processing, and robotics.

        Methodology
        Our research methodology involves a systematic comparison of different
        machine learning algorithms across multiple datasets. We evaluate
        performance metrics including accuracy, precision, recall, and F1-score.

        Results
        The experimental results demonstrate that deep learning approaches
        consistently outperform traditional machine learning methods on
        complex datasets. However, simpler algorithms remain effective for
        smaller datasets and interpretability requirements.

        Conclusion
        This research contributes to the understanding of machine learning
        algorithm selection and provides guidelines for practitioners in
        choosing appropriate methods for their specific use cases.
        """,
        
        "technical": """
        Software Architecture Documentation

        System Overview
        This document describes the architecture of a distributed microservices
        system designed for high availability and scalability. The system
        consists of multiple independent services communicating through
        well-defined APIs.

        Architecture Components
        The system architecture includes the following key components:
        - API Gateway for request routing and authentication
        - Microservices for business logic implementation
        - Message queues for asynchronous communication
        - Database clusters for data persistence
        - Monitoring and logging infrastructure

        Design Patterns
        We employ several design patterns to ensure maintainability:
        - Circuit Breaker pattern for fault tolerance
        - CQRS pattern for read/write separation
        - Event Sourcing for audit trails
        - Saga pattern for distributed transactions

        Deployment Strategy
        The system is deployed using containerization with Kubernetes
        orchestration. This approach provides scalability, resilience,
        and efficient resource utilization.
        """,
        
        "business": """
        Quarterly Business Report

        Executive Summary
        This report presents the financial and operational performance
        for Q3 2024. Overall, the company has achieved strong growth
        across all key metrics, with revenue increasing by 15% compared
        to the previous quarter.

        Financial Performance
        Revenue for Q3 2024 reached $2.5 million, representing a 15%
        increase from Q2 2024. Gross profit margin improved to 65%,
        driven by operational efficiency improvements and cost optimization
        initiatives.

        Market Analysis
        The market conditions remain favorable for our products and services.
        Customer demand has increased significantly, particularly in the
        enterprise segment. Competition remains intense, but our unique
        value proposition continues to differentiate us in the marketplace.

        Strategic Initiatives
        Key strategic initiatives for the next quarter include:
        - Expansion into new geographic markets
        - Product line diversification
        - Technology infrastructure upgrades
        - Talent acquisition and retention programs

        Outlook
        Looking ahead, we expect continued growth momentum driven by
        strong market demand and successful execution of our strategic
        initiatives. We project 20% revenue growth for Q4 2024.
        """
    }
    
    content = content_templates.get(content_type, content_templates["research"])
    
    with open(path, 'w', encoding='utf-8') as f:
        f.write(content.strip())
