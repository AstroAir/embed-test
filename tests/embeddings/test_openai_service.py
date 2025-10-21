"""Tests for OpenAIEmbeddingService."""

from unittest.mock import Mock, patch

import pytest

from pdf_vector_system.embeddings.base import EmbeddingResult
from pdf_vector_system.embeddings.openai_service import OpenAIEmbeddingService
from tests.mocks.embedding_mocks import MockOpenAIService


class TestOpenAIEmbeddingService:
    """Test OpenAIEmbeddingService with mocking."""

    def test_mock_service_functionality(self):
        """Test mock OpenAI service."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key", embedding_dim=1536
        )

        # Test single embedding
        embedding = service.embed_single("test text")
        assert len(embedding) == 1536
        assert all(isinstance(x, float) for x in embedding)

        # Test multiple embeddings
        texts = ["text 1", "text 2", "text 3"]
        result = service.embed_texts(texts)
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 3
        assert all(len(emb) == 1536 for emb in result.embeddings)
        assert result.model_name == "text-embedding-3-small"
        assert result.embedding_dimension == 1536

        # Test model info
        info = service.get_model_info()
        assert info["api_provider"] == "openai"
        assert info["has_api_key"] is True
        assert service.request_count > 0

    def test_model_configurations(self):
        """Test model configuration constants."""
        # Test that model configs exist
        assert "text-embedding-3-small" in OpenAIEmbeddingService.MODEL_CONFIGS
        assert "text-embedding-3-large" in OpenAIEmbeddingService.MODEL_CONFIGS
        assert "text-embedding-ada-002" in OpenAIEmbeddingService.MODEL_CONFIGS

        # Test config structure
        config = OpenAIEmbeddingService.MODEL_CONFIGS["text-embedding-3-small"]
        assert "dimension" in config
        assert "max_tokens" in config

        # Test specific values
        assert config["dimension"] == 1536
        assert config["max_tokens"] == 8191

    def test_mock_service_initialization(self):
        """Test mock service initialization."""
        service = MockOpenAIService(
            model_name="text-embedding-3-large",
            api_key="test-api-key",
            embedding_dim=3072,
        )

        assert service.model_name == "text-embedding-3-large"
        assert service.api_key == "test-api-key"
        assert service.embedding_dimension == 3072
        assert service.request_count == 0

    def test_mock_service_health_check(self):
        """Test mock service health check."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key"
        )

        assert service.health_check() is True

    def test_mock_service_batch_processing(self):
        """Test mock service batch processing."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key", embedding_dim=1536
        )

        # Test with larger batch
        texts = [f"Text number {i}" for i in range(20)]
        result = service.embed_texts(texts)

        assert len(result.embeddings) == 20
        assert all(len(emb) == 1536 for emb in result.embeddings)
        assert result.text_count == 20
        assert result.processing_time > 0
        assert service.request_count > 0

    def test_mock_service_empty_input(self):
        """Test mock service with empty input."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key"
        )

        result = service.embed_texts([])

        assert len(result.embeddings) == 0
        assert result.text_count == 0
        assert result.model_name == "text-embedding-3-small"

    def test_mock_service_api_key_handling(self):
        """Test mock service API key handling."""
        # Test with API key
        service_with_key = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="valid-key"
        )

        info = service_with_key.get_model_info()
        assert info["has_api_key"] is True

        # Test without API key
        service_without_key = MockOpenAIService(
            model_name="text-embedding-3-small", api_key=None
        )

        info = service_without_key.get_model_info()
        assert info["has_api_key"] is False

    def test_mock_service_model_info_structure(self):
        """Test that model info has expected structure."""
        service = MockOpenAIService(
            model_name="text-embedding-3-large", api_key="test-key", embedding_dim=3072
        )

        info = service.get_model_info()

        # Check required fields
        required_fields = [
            "model_name",
            "api_provider",
            "embedding_dimension",
            "has_api_key",
            "is_mock",
        ]
        for field in required_fields:
            assert field in info

        # Check values
        assert info["model_name"] == "text-embedding-3-large"
        assert info["api_provider"] == "openai"
        assert info["embedding_dimension"] == 3072
        assert info["has_api_key"] is True
        assert info["is_mock"] is True

    def test_mock_service_request_tracking(self):
        """Test that mock service tracks API requests."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key"
        )

        initial_count = service.request_count

        # Make some requests
        service.embed_single("test text 1")
        assert service.request_count == initial_count + 1

        service.embed_texts(["text 1", "text 2"])
        assert service.request_count == initial_count + 2

    def test_mock_service_error_simulation(self):
        """Test mock service error simulation."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key"
        )

        # Mock service should handle edge cases gracefully
        # Test with None input
        with pytest.raises((TypeError, ValueError)):
            service.embed_single(None)

        # Test with empty string
        embedding = service.embed_single("")
        assert len(embedding) == service.embedding_dimension

    def test_mock_service_performance_metrics(self):
        """Test that mock service provides performance metrics."""
        service = MockOpenAIService(
            model_name="text-embedding-3-small", api_key="test-key"
        )

        texts = ["text 1", "text 2", "text 3"]
        result = service.embed_texts(texts)

        # Should have performance metrics
        assert result.processing_time > 0
        assert result.texts_per_second > 0
        assert result.text_count == 3

        # Should have token count for OpenAI
        assert result.token_count is not None
        assert result.token_count > 0

    def test_model_config_completeness(self):
        """Test that all model configs have required fields."""
        for model_name, config in OpenAIEmbeddingService.MODEL_CONFIGS.items():
            assert "dimension" in config, f"Model {model_name} missing dimension"
            assert "max_tokens" in config, f"Model {model_name} missing max_tokens"
            assert isinstance(config["dimension"], int)
            assert isinstance(config["max_tokens"], int)
            assert config["dimension"] > 0
            assert config["max_tokens"] > 0

    def test_supported_models(self):
        """Test that service supports expected models."""
        expected_models = [
            "text-embedding-3-small",
            "text-embedding-3-large",
            "text-embedding-ada-002",
        ]

        for model in expected_models:
            assert model in OpenAIEmbeddingService.MODEL_CONFIGS

    @patch("pdf_vector_system.embeddings.openai_service.OpenAI")
    def test_real_service_initialization(self, mock_openai_class):
        """Test real service initialization with mocked OpenAI client."""
        mock_client = Mock()
        mock_openai_class.return_value = mock_client

        service = OpenAIEmbeddingService(
            model_name="text-embedding-3-small", api_key="test-key"
        )

        assert service.model_name == "text-embedding-3-small"
        assert service.api_key == "test-key"
        mock_openai_class.assert_called_once_with(
            api_key="test-key", base_url=None, max_retries=3, timeout=60.0
        )
