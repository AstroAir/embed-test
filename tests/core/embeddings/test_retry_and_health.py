"""Tests for new embedding providers."""

from unittest.mock import Mock, patch

import pytest

from vectorflow.core.config.settings import EmbeddingConfig, EmbeddingModelType
from vectorflow.core.embeddings.base import EmbeddingResult, EmbeddingServiceError
from vectorflow.core.embeddings.factory import (
    EmbeddingServiceFactory,
    EnhancedBatchEmbeddingProcessor,
)
from vectorflow.core.embeddings.health_check import (
    HealthCheckManager,
    ProviderHealthChecker,
)
from vectorflow.core.embeddings.retry import (
    ProviderCircuitBreaker,
    RetryConfig,
    RetryHandler,
)


class TestRetryMechanism:
    """Test retry mechanism functionality."""

    def test_retry_config_creation(self):
        """Test retry configuration creation."""
        config = RetryConfig(max_retries=5, base_delay=2.0, max_delay=60.0)

        assert config.max_retries == 5
        assert config.base_delay == 2.0
        assert config.max_delay == 60.0

    def test_retry_handler_success(self):
        """Test retry handler with successful function."""
        config = RetryConfig(max_retries=3)
        handler = RetryHandler(config, "test")

        mock_func = Mock(return_value="success")
        result = handler.execute(mock_func, "arg1", kwarg1="value1")

        assert result == "success"
        mock_func.assert_called_once_with("arg1", kwarg1="value1")

    def test_retry_handler_with_retries(self):
        """Test retry handler with failures then success."""
        config = RetryConfig(max_retries=3, base_delay=0.01)  # Fast for testing
        handler = RetryHandler(config, "test")

        mock_func = Mock(
            side_effect=[Exception("fail1"), Exception("fail2"), "success"]
        )
        result = handler.execute(mock_func, "arg1")

        assert result == "success"
        assert mock_func.call_count == 3

    def test_circuit_breaker(self):
        """Test circuit breaker functionality."""
        breaker = ProviderCircuitBreaker(
            provider_name="test_provider", failure_threshold=2, recovery_timeout=0.1
        )

        # Initially closed
        assert breaker.can_execute()

        # Record failures
        breaker.record_failure()
        assert breaker.can_execute()  # Still closed

        breaker.record_failure()
        assert not breaker.can_execute()  # Now open

        # Wait for recovery
        import time

        time.sleep(0.2)
        assert breaker.can_execute()  # Half-open

        # Record success to close
        breaker.record_success()
        assert breaker.can_execute()


class TestHealthCheckSystem:
    """Test health check system."""

    def test_health_check_manager(self):
        """Test health check manager."""
        manager = HealthCheckManager()

        # Mock embedding service
        mock_service = Mock()
        mock_service.embed_single.return_value = [0.1, 0.2, 0.3]

        # Register provider
        manager.register_provider("test_provider", mock_service)

        # Check health
        result = manager.check_provider_health("test_provider", force=True)

        assert result.provider_name == "test_provider"
        assert result.is_healthy

    def test_provider_health_checker(self):
        """Test individual provider health checker."""
        mock_service = Mock()
        mock_service.embed_single.return_value = [0.1, 0.2, 0.3]

        checker = ProviderHealthChecker("test_provider", mock_service)
        result = checker.check_health()

        assert result.provider_name == "test_provider"
        assert result.is_healthy
        mock_service.embed_single.assert_called_once()


class TestEnhancedBatchProcessor:
    """Test enhanced batch processing."""

    def test_enhanced_batch_processor_creation(self):
        """Test enhanced batch processor creation."""
        mock_service = Mock()
        mock_service.model_name = "test-model"
        mock_service.embedding_dimension = 384
        mock_service.__class__.__name__ = "MockEmbeddingService"

        processor = EnhancedBatchEmbeddingProcessor(
            embedding_service=mock_service,
            max_workers=2,
            batch_size=16,
            adaptive_batch_sizing=True,
        )

        assert processor.embedding_service == mock_service
        assert processor.max_workers == 2
        assert processor.current_batch_size == 16
        assert processor.adaptive_batch_sizing is True

    @patch("psutil.virtual_memory")
    def test_adaptive_batch_processing(self, mock_memory):
        """Test adaptive batch processing."""
        # Mock memory usage
        mock_memory.return_value.percent = 50.0

        mock_service = Mock()
        mock_service.model_name = "test-model"
        mock_service.embedding_dimension = 4  # Match the mock embeddings
        mock_service.__class__.__name__ = "MockEmbeddingService"

        # Mock embed_batch to return results
        def mock_embed_batch(batch):
            embeddings = [[0.1, 0.2, 0.3, 0.4] for _ in batch.texts]
            return EmbeddingResult(
                embeddings=embeddings,
                model_name="test-model",
                embedding_dimension=4,
                processing_time=0.1,
            )

        mock_service.embed_batch = mock_embed_batch

        processor = EnhancedBatchEmbeddingProcessor(
            embedding_service=mock_service, batch_size=2, adaptive_batch_sizing=True
        )

        texts = ["text1", "text2", "text3", "text4"]
        result = processor.process_texts_adaptive(texts, show_progress=False)

        assert len(result.embeddings) == 4
        assert result.model_name == "test-model"
        assert result.metadata["enhanced_batch_processor"] is True


class TestNewProviderConfigurations:
    """Test new provider configurations."""

    def test_extended_embedding_model_types(self):
        """Test that new model types are available."""
        assert EmbeddingModelType.COHERE == "cohere"
        assert EmbeddingModelType.HUGGINGFACE == "huggingface"
        assert EmbeddingModelType.GOOGLE_USE == "google_use"
        assert EmbeddingModelType.AZURE_OPENAI == "azure_openai"

    def test_embedding_config_with_new_providers(self):
        """Test embedding configuration with new provider settings."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.COHERE,
            model_name="embed-english-v3.0",
            cohere_api_key="test-key",
            batch_size=48,
            adaptive_batch_sizing=True,
            memory_limit_mb=256,
        )

        assert config.model_type == EmbeddingModelType.COHERE
        assert config.cohere_api_key == "test-key"
        assert config.adaptive_batch_sizing is True
        assert config.memory_limit_mb == 256

    def test_azure_openai_config(self):
        """Test Azure OpenAI specific configuration."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.AZURE_OPENAI,
            model_name="text-embedding-3-small",
            azure_openai_api_key="test-key",
            azure_openai_endpoint="https://test.openai.azure.com",
            azure_openai_deployment_name="test-deployment",
        )

        assert config.model_type == EmbeddingModelType.AZURE_OPENAI
        assert config.azure_openai_api_key == "test-key"
        assert config.azure_openai_endpoint == "https://test.openai.azure.com"
        assert config.azure_openai_deployment_name == "test-deployment"


class TestFactoryEnhancements:
    """Test factory enhancements for new providers."""

    def test_factory_get_available_models(self):
        """Test that factory returns models for all providers."""
        models = EmbeddingServiceFactory.get_available_models()

        # Check that new providers are included
        assert "cohere" in models
        assert "huggingface" in models
        assert "google_use" in models
        assert "azure_openai" in models

        # Check some specific models
        assert "embed-english-v3.0" in models["cohere"]
        assert "universal-sentence-encoder" in models["google_use"]

    def test_factory_create_cohere_service(self):
        """Test factory creation of Cohere service."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.COHERE,
            model_name="embed-english-v3.0",
            cohere_api_key="test-key",
        )

        # This should raise an EmbeddingServiceError if Cohere is not installed
        try:
            service = EmbeddingServiceFactory.create_service(config)
            assert service is not None
            # If we get here, Cohere is installed and working
        except EmbeddingServiceError as e:
            # Expected if Cohere is not installed
            assert "Cohere package is required" in str(e)
        except ImportError:
            # Also acceptable
            pass

    def test_factory_unsupported_provider_error(self):
        """Test factory raises error for unsupported provider."""
        # Test with a mock config that bypasses Pydantic validation
        from unittest.mock import Mock

        mock_config = Mock()
        mock_config.model_type = "unsupported_provider"

        with pytest.raises(ValueError, match="Unsupported embedding model type"):
            EmbeddingServiceFactory.create_service(mock_config)


class TestProviderAvailability:
    """Test provider availability checking."""

    def test_get_available_providers(self):
        """Test getting available providers."""
        from vectorflow.core.embeddings import get_available_providers

        providers = get_available_providers()

        # Core providers should always be available
        assert providers["sentence_transformers"] is True
        assert providers["openai"] is True

        # Optional providers depend on installed packages
        assert "cohere" in providers
        assert "huggingface" in providers
        assert "google_use" in providers
        assert "azure_openai" in providers

    def test_check_provider_dependencies(self):
        """Test checking provider dependencies."""
        from vectorflow.core.embeddings import check_provider_dependencies

        missing = check_provider_dependencies()

        # Should return installation commands for missing dependencies
        assert isinstance(missing, dict)

        # If a provider is missing, should have installation command
        for _provider, command in missing.items():
            assert "pip install" in command


@pytest.mark.integration
class TestProviderIntegration:
    """Integration tests for provider functionality."""

    def test_sentence_transformers_still_works(self):
        """Test that existing sentence transformers provider still works."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
        )

        # This should not raise an error
        service = EmbeddingServiceFactory.create_service(config)
        assert service is not None
        assert service.model_name == "all-MiniLM-L6-v2"

    def test_openai_still_works(self):
        """Test that existing OpenAI provider still works."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            openai_api_key="test-key",
        )

        # This should not raise an error
        service = EmbeddingServiceFactory.create_service(config)
        assert service is not None
        assert service.model_name == "text-embedding-3-small"
