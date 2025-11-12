"""Pytest configuration and fixtures for embedding service tests."""

from unittest.mock import Mock

import pytest

from pdf_vector_system.core.config.settings import EmbeddingConfig, EmbeddingModelType
from pdf_vector_system.core.embeddings.base import EmbeddingBatch, EmbeddingResult
from tests.mocks.embedding_mocks import (
    MockEmbeddingService,
    MockOpenAIService,
    MockSentenceTransformersService,
)


@pytest.fixture
def mock_embedding_service() -> MockEmbeddingService:
    """Create a mock embedding service for testing."""
    return MockEmbeddingService(model_name="test-model", embedding_dim=384)


@pytest.fixture
def mock_sentence_transformers_service() -> MockSentenceTransformersService:
    """Create a mock sentence transformers service."""
    return MockSentenceTransformersService(
        model_name="all-MiniLM-L6-v2", embedding_dim=384
    )


@pytest.fixture
def mock_openai_service() -> MockOpenAIService:
    """Create a mock OpenAI service."""
    return MockOpenAIService(
        model_name="text-embedding-3-small", api_key="test-key", embedding_dim=1536
    )


@pytest.fixture
def sample_embedding_result() -> EmbeddingResult:
    """Create a sample embedding result for testing."""
    return EmbeddingResult(
        embeddings=[[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9]],
        model_name="test-model",
        embedding_dimension=3,
        processing_time=1.5,
        token_count=15,
        metadata={"test": True},
    )


@pytest.fixture
def sample_embedding_batch() -> EmbeddingBatch:
    """Create a sample embedding batch for testing."""
    return EmbeddingBatch(
        texts=["text 1", "text 2", "text 3"],
        batch_id="test_batch_1",
        metadata={"source": "test"},
    )


@pytest.fixture
def embedding_config_sentence_transformers() -> EmbeddingConfig:
    """Create embedding config for sentence transformers."""
    return EmbeddingConfig(
        model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
        model_name="all-MiniLM-L6-v2",
        batch_size=16,
    )


@pytest.fixture
def embedding_config_openai() -> EmbeddingConfig:
    """Create embedding config for OpenAI."""
    return EmbeddingConfig(
        model_type=EmbeddingModelType.OPENAI,
        model_name="text-embedding-3-small",
        batch_size=100,
        openai_api_key="test-key",
    )


@pytest.fixture
def sample_texts() -> list[str]:
    """Provide sample texts for embedding tests."""
    return [
        "This is the first sample text for embedding.",
        "Here is another piece of text to embed.",
        "The third text contains different content.",
        "Final sample text for comprehensive testing.",
    ]


@pytest.fixture
def sample_embeddings_3d() -> list[list[float]]:
    """Provide sample 3D embeddings for testing."""
    return [[0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [0.7, 0.8, 0.9], [0.2, 0.3, 0.4]]


@pytest.fixture
def sample_embeddings_384d() -> list[list[float]]:
    """Provide sample 384D embeddings for testing."""
    import random

    random.seed(42)  # For reproducible tests
    return [[random.random() for _ in range(384)] for _ in range(4)]


@pytest.fixture
def mock_retry_config():
    """Create a mock retry configuration."""
    from pdf_vector_system.core.embeddings.retry import RetryConfig

    return RetryConfig(
        max_retries=3,
        initial_delay=0.1,
        max_delay=1.0,
        exponential_base=2.0,
        jitter=False,
    )


@pytest.fixture
def mock_health_check_manager():
    """Create a mock health check manager."""
    mock = Mock()
    mock.check_service_health.return_value = True
    mock.get_health_status.return_value = {
        "healthy": True,
        "last_check": "2024-01-01T00:00:00",
        "response_time": 0.1,
    }
    return mock
