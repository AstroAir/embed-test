"""Embedding generation module supporting multiple models."""

# Check availability of optional providers using importlib
import importlib.util

from pdf_vector_system.embeddings.base import (
    EmbeddingBatch,
    EmbeddingResult,
    EmbeddingService,
)
from pdf_vector_system.embeddings.factory import (
    BatchEmbeddingProcessor,
    EmbeddingServiceFactory,
    EnhancedBatchEmbeddingProcessor,
)
from pdf_vector_system.embeddings.health_check import (
    HealthCheckManager,
    ProviderHealthChecker,
)
from pdf_vector_system.embeddings.openai_service import OpenAIEmbeddingService
from pdf_vector_system.embeddings.retry import (
    ProviderCircuitBreaker,
    RetryConfig,
    RetryHandler,
)
from pdf_vector_system.embeddings.sentence_transformers_service import (
    SentenceTransformersService,
)

_COHERE_AVAILABLE = (
    importlib.util.find_spec("pdf_vector_system.embeddings.cohere_service") is not None
)
_HUGGINGFACE_AVAILABLE = (
    importlib.util.find_spec("pdf_vector_system.embeddings.huggingface_service")
    is not None
)
_GOOGLE_USE_AVAILABLE = (
    importlib.util.find_spec("pdf_vector_system.embeddings.google_use_service")
    is not None
)
_AZURE_OPENAI_AVAILABLE = (
    importlib.util.find_spec("pdf_vector_system.embeddings.azure_openai_service")
    is not None
)
_GOOGLE_GEMINI_AVAILABLE = (
    importlib.util.find_spec("pdf_vector_system.embeddings.gemini_service") is not None
)

__all__ = [
    "BatchEmbeddingProcessor",
    "EmbeddingBatch",
    "EmbeddingResult",
    "EmbeddingService",
    "EmbeddingServiceFactory",
    "EnhancedBatchEmbeddingProcessor",
    "HealthCheckManager",
    "OpenAIEmbeddingService",
    "ProviderCircuitBreaker",
    "ProviderHealthChecker",
    "RetryConfig",
    "RetryHandler",
    "SentenceTransformersService",
]

# Add optional services to __all__ if available and import them dynamically
if _COHERE_AVAILABLE:
    from pdf_vector_system.embeddings.cohere_service import (  # noqa: F401
        CohereEmbeddingService,
    )

    __all__.append("CohereEmbeddingService")

if _HUGGINGFACE_AVAILABLE:
    from pdf_vector_system.embeddings.huggingface_service import (  # noqa: F401
        HuggingFaceEmbeddingService,
    )

    __all__.append("HuggingFaceEmbeddingService")

if _GOOGLE_USE_AVAILABLE:
    from pdf_vector_system.embeddings.google_use_service import (  # noqa: F401
        GoogleUSEService,
    )

    __all__.append("GoogleUSEService")

if _AZURE_OPENAI_AVAILABLE:
    from pdf_vector_system.embeddings.azure_openai_service import (  # noqa: F401
        AzureOpenAIEmbeddingService,
    )

    __all__.append("AzureOpenAIEmbeddingService")

if _GOOGLE_GEMINI_AVAILABLE:
    from pdf_vector_system.embeddings.gemini_service import (  # noqa: F401
        GeminiEmbeddingService,
    )

    __all__.append("GeminiEmbeddingService")


def get_available_providers() -> dict[str, bool]:
    """
    Get availability status of all embedding providers.

    Returns:
        Dictionary mapping provider names to availability status
    """
    return {
        "sentence_transformers": True,  # Always available (core dependency)
        "openai": True,  # Always available (core dependency)
        "cohere": _COHERE_AVAILABLE,
        "huggingface": _HUGGINGFACE_AVAILABLE,
        "google_use": _GOOGLE_USE_AVAILABLE,
        "google_gemini": _GOOGLE_GEMINI_AVAILABLE,
        "azure_openai": _AZURE_OPENAI_AVAILABLE,
    }


def check_provider_dependencies() -> dict[str, str]:
    """
    Check which provider dependencies are missing.

    Returns:
        Dictionary mapping provider names to installation commands for missing dependencies
    """
    missing = {}

    if not _COHERE_AVAILABLE:
        missing["cohere"] = "pip install cohere>=4.0.0"

    if not _HUGGINGFACE_AVAILABLE:
        missing["huggingface"] = "pip install transformers>=4.21.0 torch>=2.0.0"

    if not _GOOGLE_USE_AVAILABLE:
        missing["google_use"] = "pip install tensorflow>=2.13.0 tensorflow-hub>=0.15.0"

    if not _AZURE_OPENAI_AVAILABLE:
        missing["azure_openai"] = "pip install openai>=1.0.0"  # Same as regular OpenAI

    if not _GOOGLE_GEMINI_AVAILABLE:
        missing["google_gemini"] = "pip install requests>=2.25.0"  # Basic HTTP client

    return missing
