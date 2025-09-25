"""Embedding generation module supporting multiple models."""

from typing import Dict

from .base import EmbeddingService, EmbeddingResult, EmbeddingBatch
from .sentence_transformers_service import SentenceTransformersService
from .openai_service import OpenAIEmbeddingService
from .factory import EmbeddingServiceFactory, BatchEmbeddingProcessor, EnhancedBatchEmbeddingProcessor
from .retry import RetryHandler, RetryConfig, ProviderCircuitBreaker
from .health_check import HealthCheckManager, ProviderHealthChecker

# Optional imports for new providers (graceful degradation if dependencies not installed)
try:
    from .cohere_service import CohereEmbeddingService
    _COHERE_AVAILABLE = True
except ImportError:
    _COHERE_AVAILABLE = False

try:
    from .huggingface_service import HuggingFaceEmbeddingService
    _HUGGINGFACE_AVAILABLE = True
except ImportError:
    _HUGGINGFACE_AVAILABLE = False

try:
    from .google_use_service import GoogleUSEService
    _GOOGLE_USE_AVAILABLE = True
except ImportError:
    _GOOGLE_USE_AVAILABLE = False

try:
    from .azure_openai_service import AzureOpenAIEmbeddingService
    _AZURE_OPENAI_AVAILABLE = True
except ImportError:
    _AZURE_OPENAI_AVAILABLE = False

try:
    from .gemini_service import GeminiEmbeddingService
    _GOOGLE_GEMINI_AVAILABLE = True
except ImportError:
    _GOOGLE_GEMINI_AVAILABLE = False

__all__ = [
    "EmbeddingService",
    "EmbeddingResult",
    "EmbeddingBatch",
    "SentenceTransformersService",
    "OpenAIEmbeddingService",
    "EmbeddingServiceFactory",
    "BatchEmbeddingProcessor",
    "EnhancedBatchEmbeddingProcessor",
    "RetryHandler",
    "RetryConfig",
    "ProviderCircuitBreaker",
    "HealthCheckManager",
    "ProviderHealthChecker"
]

# Add optional services to __all__ if available
if _COHERE_AVAILABLE:
    __all__.append("CohereEmbeddingService")

if _HUGGINGFACE_AVAILABLE:
    __all__.append("HuggingFaceEmbeddingService")

if _GOOGLE_USE_AVAILABLE:
    __all__.append("GoogleUSEService")

if _AZURE_OPENAI_AVAILABLE:
    __all__.append("AzureOpenAIEmbeddingService")

if _GOOGLE_GEMINI_AVAILABLE:
    __all__.append("GeminiEmbeddingService")


def get_available_providers() -> Dict[str, bool]:
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


def check_provider_dependencies() -> Dict[str, str]:
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
