"""Tests for provider configuration utilities."""

from pdf_vector_system.embeddings.provider_configs import (
    ProviderBatchConfigs,
    ProviderHealthConfigs,
    ProviderRetryConfigs,
)
from pdf_vector_system.embeddings.retry import FailureType, RetryConfig, RetryStrategy


class TestProviderRetryConfigs:
    """Test provider retry configurations."""

    def test_get_openai_config(self):
        """Test OpenAI retry configuration."""
        config = ProviderRetryConfigs.get_openai_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries >= 3
        assert config.base_delay > 0
        assert config.max_delay > config.base_delay
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF
        assert config.jitter is True

    def test_get_cohere_config(self):
        """Test Cohere retry configuration."""
        config = ProviderRetryConfigs.get_cohere_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries >= 3
        assert config.base_delay > 0
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_get_azure_openai_config(self):
        """Test Azure OpenAI retry configuration."""
        config = ProviderRetryConfigs.get_azure_openai_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries >= 3
        assert config.base_delay > 0
        # Azure might have different settings than regular OpenAI
        assert config.max_delay >= 60.0

    def test_get_google_gemini_config(self):
        """Test Google Gemini retry configuration."""
        config = ProviderRetryConfigs.get_google_gemini_config()

        assert isinstance(config, RetryConfig)
        assert config.max_retries >= 3
        assert config.base_delay > 0
        assert config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF

    def test_get_local_model_config(self):
        """Test local model retry configuration."""
        config = ProviderRetryConfigs.get_local_model_config()

        assert isinstance(config, RetryConfig)
        # Local models should have fewer retries and shorter delays
        assert config.max_retries <= 3
        assert config.base_delay <= 2.0
        assert config.max_delay <= 30.0

    def test_get_config_for_provider(self):
        """Test getting config for specific providers."""
        providers = [
            "openai",
            "cohere",
            "azure_openai",
            "google_gemini",
            "huggingface",
            "google_use",
            "sentence_transformers",
        ]

        for provider in providers:
            config = ProviderRetryConfigs.get_config_for_provider(provider)
            assert isinstance(config, RetryConfig)
            assert config.max_retries > 0
            assert config.base_delay > 0

    def test_get_config_for_unknown_provider(self):
        """Test getting config for unknown provider."""
        config = ProviderRetryConfigs.get_config_for_provider("unknown_provider")

        # Should return a default config
        assert isinstance(config, RetryConfig)
        assert config.max_retries > 0

    def test_failure_specific_configs(self):
        """Test that configs have failure-specific settings."""
        config = ProviderRetryConfigs.get_openai_config()

        # Should have specific configurations for different failure types
        assert isinstance(config.failure_strategies, dict)

        # Check for common failure types
        if FailureType.RATE_LIMIT in config.failure_strategies:
            rate_limit_config = config.failure_strategies[FailureType.RATE_LIMIT]
            assert rate_limit_config.base_delay >= config.base_delay

    def test_circuit_breaker_settings(self):
        """Test circuit breaker settings in configs."""
        config = ProviderRetryConfigs.get_openai_config()

        assert isinstance(config.circuit_breaker_enabled, bool)
        if config.circuit_breaker_enabled:
            assert config.circuit_breaker_failure_threshold > 0
            assert config.circuit_breaker_recovery_timeout > 0


class TestProviderBatchConfigs:
    """Test provider batch configurations."""

    def test_get_batch_config_for_openai(self):
        """Test batch config for OpenAI."""
        config = ProviderBatchConfigs.get_batch_config_for_provider("openai")

        assert isinstance(config, dict)
        assert "max_batch_size" in config
        assert "optimal_batch_size" in config
        assert "max_tokens_per_batch" in config
        assert "parallel_batches" in config

        # OpenAI should support large batches
        assert config["max_batch_size"] >= 100
        assert config["optimal_batch_size"] <= config["max_batch_size"]

    def test_get_batch_config_for_cohere(self):
        """Test batch config for Cohere."""
        config = ProviderBatchConfigs.get_batch_config_for_provider("cohere")

        assert isinstance(config, dict)
        assert config["max_batch_size"] > 0
        assert config["optimal_batch_size"] > 0
        assert config["parallel_batches"] > 0

    def test_get_batch_config_for_local_models(self):
        """Test batch config for local models."""
        local_providers = ["sentence_transformers", "huggingface", "google_use"]

        for provider in local_providers:
            config = ProviderBatchConfigs.get_batch_config_for_provider(provider)

            assert isinstance(config, dict)
            assert config["max_batch_size"] > 0
            # Local models typically have smaller batch sizes
            assert config["max_batch_size"] <= 64
            # Usually single GPU/process
            assert config["parallel_batches"] <= 2

    def test_get_batch_config_for_unknown_provider(self):
        """Test batch config for unknown provider."""
        config = ProviderBatchConfigs.get_batch_config_for_provider("unknown")

        # Should return default config
        assert isinstance(config, dict)
        assert "max_batch_size" in config
        assert config["max_batch_size"] > 0

    def test_batch_config_consistency(self):
        """Test that batch configs are internally consistent."""
        providers = ["openai", "cohere", "sentence_transformers", "huggingface"]

        for provider in providers:
            config = ProviderBatchConfigs.get_batch_config_for_provider(provider)

            # Optimal should be <= max
            assert config["optimal_batch_size"] <= config["max_batch_size"]

            # All values should be positive
            assert config["max_batch_size"] > 0
            assert config["optimal_batch_size"] > 0
            assert config["parallel_batches"] > 0

            # Memory limit should be reasonable
            if "memory_limit_mb" in config:
                assert config["memory_limit_mb"] > 0
                # Reasonable upper bound
                assert config["memory_limit_mb"] <= 8192


class TestProviderHealthConfigs:
    """Test provider health check configurations."""

    def test_get_health_config_for_openai(self):
        """Test health config for OpenAI."""
        config = ProviderHealthConfigs.get_health_config_for_provider("openai")

        assert isinstance(config, dict)
        assert "test_text" in config
        assert "timeout" in config
        assert "check_interval" in config
        assert "failure_threshold" in config

        # OpenAI should have reasonable timeouts
        assert config["timeout"] > 0
        assert config["timeout"] <= 60.0
        assert config["failure_threshold"] > 0

    def test_get_health_config_for_local_models(self):
        """Test health config for local models."""
        local_providers = ["sentence_transformers", "huggingface", "google_use"]

        for provider in local_providers:
            config = ProviderHealthConfigs.get_health_config_for_provider(provider)

            assert isinstance(config, dict)
            assert "test_text" in config
            assert "timeout" in config

            # Local models might need longer timeouts for loading
            assert config["timeout"] >= 30.0

    def test_get_health_config_for_api_providers(self):
        """Test health config for API-based providers."""
        api_providers = ["openai", "cohere", "azure_openai", "google_gemini"]

        for provider in api_providers:
            config = ProviderHealthConfigs.get_health_config_for_provider(provider)

            assert isinstance(config, dict)
            assert "test_text" in config
            assert "timeout" in config

            # API providers should have shorter timeouts
            assert config["timeout"] <= 60.0

            # Should have check intervals
            assert "check_interval" in config
            assert config["check_interval"] > 0

    def test_health_config_test_text(self):
        """Test that health configs have appropriate test text."""
        providers = ["openai", "cohere", "sentence_transformers"]

        for provider in providers:
            config = ProviderHealthConfigs.get_health_config_for_provider(provider)

            test_text = config["test_text"]
            assert isinstance(test_text, str)
            assert len(test_text) > 0
            assert len(test_text) <= 100  # Should be short for health checks

    def test_health_config_expected_dimensions(self):
        """Test expected dimensions in health configs."""
        # Some providers have known dimensions
        config = ProviderHealthConfigs.get_health_config_for_provider("google_use")
        if config.get("expected_dimension"):
            assert isinstance(config["expected_dimension"], int)
            assert config["expected_dimension"] > 0

    def test_get_health_config_for_unknown_provider(self):
        """Test health config for unknown provider."""
        config = ProviderHealthConfigs.get_health_config_for_provider("unknown")

        # Should return default config
        assert isinstance(config, dict)
        assert "test_text" in config
        assert "timeout" in config
        assert config["timeout"] > 0

    def test_health_config_consistency(self):
        """Test that health configs are internally consistent."""
        providers = ["openai", "cohere", "sentence_transformers", "huggingface"]

        for provider in providers:
            config = ProviderHealthConfigs.get_health_config_for_provider(provider)

            # All required fields should be present
            required_fields = [
                "test_text",
                "timeout",
                "check_interval",
                "failure_threshold",
            ]
            for field in required_fields:
                assert field in config

            # Values should be reasonable
            assert config["timeout"] > 0
            assert config["check_interval"] > 0
            assert config["failure_threshold"] > 0

            # Check interval should be longer than timeout
            assert config["check_interval"] > config["timeout"]
