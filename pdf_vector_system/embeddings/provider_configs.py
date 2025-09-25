"""Provider-specific configurations and retry strategies."""

from typing import Dict, Any
from .retry import RetryConfig, RetryStrategy, FailureType


class ProviderRetryConfigs:
    """Predefined retry configurations for different embedding providers."""
    
    @staticmethod
    def get_openai_config() -> RetryConfig:
        """Get retry configuration optimized for OpenAI API."""
        return RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            jitter_range=0.1,
            backoff_multiplier=2.0,
            failure_strategies={
                FailureType.RATE_LIMIT: RetryConfig(
                    max_retries=5,
                    base_delay=2.0,
                    max_delay=120.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    backoff_multiplier=2.5,
                    jitter=True,
                    jitter_range=0.2
                ),
                FailureType.SERVER_ERROR: RetryConfig(
                    max_retries=4,
                    base_delay=1.0,
                    max_delay=30.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    backoff_multiplier=2.0
                ),
                FailureType.NETWORK_ERROR: RetryConfig(
                    max_retries=3,
                    base_delay=0.5,
                    max_delay=10.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    backoff_multiplier=1.5
                )
            },
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=5,
            circuit_breaker_recovery_timeout=60.0
        )
    
    @staticmethod
    def get_cohere_config() -> RetryConfig:
        """Get retry configuration optimized for Cohere API."""
        return RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=45.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            jitter_range=0.15,
            backoff_multiplier=2.0,
            failure_strategies={
                FailureType.RATE_LIMIT: RetryConfig(
                    max_retries=4,
                    base_delay=3.0,
                    max_delay=90.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    backoff_multiplier=2.0,
                    jitter=True,
                    jitter_range=0.25
                ),
                FailureType.QUOTA_EXCEEDED: RetryConfig(
                    max_retries=2,
                    base_delay=10.0,
                    max_delay=60.0,
                    strategy=RetryStrategy.LINEAR_BACKOFF
                )
            },
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=4,
            circuit_breaker_recovery_timeout=45.0
        )
    
    @staticmethod
    def get_azure_openai_config() -> RetryConfig:
        """Get retry configuration optimized for Azure OpenAI API."""
        return RetryConfig(
            max_retries=4,
            base_delay=1.5,
            max_delay=90.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            jitter_range=0.2,
            backoff_multiplier=2.0,
            failure_strategies={
                FailureType.RATE_LIMIT: RetryConfig(
                    max_retries=6,
                    base_delay=2.0,
                    max_delay=180.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    backoff_multiplier=2.0,
                    jitter=True,
                    jitter_range=0.3
                ),
                FailureType.AUTHENTICATION: RetryConfig(
                    max_retries=1,
                    base_delay=1.0,
                    max_delay=5.0,
                    strategy=RetryStrategy.FIXED_DELAY
                ),
                FailureType.SERVER_ERROR: RetryConfig(
                    max_retries=5,
                    base_delay=2.0,
                    max_delay=60.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                    backoff_multiplier=1.8
                )
            },
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=6,
            circuit_breaker_recovery_timeout=90.0
        )

    @staticmethod
    def get_google_gemini_config() -> RetryConfig:
        """Get retry configuration optimized for Google Gemini API."""
        return RetryConfig(
            max_retries=3,
            base_delay=1.0,
            max_delay=60.0,
            strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
            jitter=True,
            failure_strategies={
                FailureType.RATE_LIMIT: RetryConfig(
                    max_retries=5,
                    base_delay=2.0,
                    max_delay=120.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                ),
                FailureType.QUOTA_EXCEEDED: RetryConfig(
                    max_retries=3,
                    base_delay=5.0,
                    max_delay=300.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                ),
                FailureType.TIMEOUT: RetryConfig(
                    max_retries=3,
                    base_delay=1.0,
                    max_delay=30.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                ),
                FailureType.SERVER_ERROR: RetryConfig(
                    max_retries=4,
                    base_delay=2.0,
                    max_delay=60.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                ),
                FailureType.NETWORK_ERROR: RetryConfig(
                    max_retries=3,
                    base_delay=1.0,
                    max_delay=30.0,
                    strategy=RetryStrategy.EXPONENTIAL_BACKOFF
                )
            },
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=5,
            circuit_breaker_recovery_timeout=60.0
        )

    @staticmethod
    def get_local_model_config() -> RetryConfig:
        """Get retry configuration for local models (Hugging Face, Google USE)."""
        return RetryConfig(
            max_retries=2,
            base_delay=0.5,
            max_delay=5.0,
            strategy=RetryStrategy.LINEAR_BACKOFF,
            jitter=False,  # Less jitter needed for local models
            failure_strategies={
                FailureType.TIMEOUT: RetryConfig(
                    max_retries=1,
                    base_delay=1.0,
                    max_delay=10.0,
                    strategy=RetryStrategy.FIXED_DELAY
                ),
                FailureType.UNKNOWN: RetryConfig(
                    max_retries=1,
                    base_delay=0.1,
                    max_delay=1.0,
                    strategy=RetryStrategy.FIXED_DELAY
                )
            },
            circuit_breaker_enabled=True,
            circuit_breaker_failure_threshold=3,
            circuit_breaker_recovery_timeout=30.0
        )
    
    @staticmethod
    def get_config_for_provider(provider: str) -> RetryConfig:
        """
        Get retry configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            RetryConfig for the provider
        """
        provider_configs = {
            "openai": ProviderRetryConfigs.get_openai_config,
            "cohere": ProviderRetryConfigs.get_cohere_config,
            "azure_openai": ProviderRetryConfigs.get_azure_openai_config,
            "google_gemini": ProviderRetryConfigs.get_google_gemini_config,
            "huggingface": ProviderRetryConfigs.get_local_model_config,
            "google_use": ProviderRetryConfigs.get_local_model_config,
            "sentence_transformers": ProviderRetryConfigs.get_local_model_config,
        }
        
        config_func = provider_configs.get(provider.lower())
        if config_func:
            return config_func()
        else:
            # Default configuration
            return RetryConfig()


class ProviderBatchConfigs:
    """Batch processing configurations for different providers."""
    
    @staticmethod
    def get_batch_config(provider: str) -> Dict[str, Any]:
        """
        Get batch configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with batch configuration
        """
        configs = {
            "openai": {
                "max_batch_size": 100,  # OpenAI API limit
                "optimal_batch_size": 50,
                "max_tokens_per_batch": 8191 * 50,  # Rough estimate
                "parallel_batches": 3,
                "memory_limit_mb": 512
            },
            "cohere": {
                "max_batch_size": 96,  # Cohere API limit
                "optimal_batch_size": 48,
                "max_tokens_per_batch": 2048 * 48,
                "parallel_batches": 2,
                "memory_limit_mb": 256
            },
            "azure_openai": {
                "max_batch_size": 100,
                "optimal_batch_size": 40,  # More conservative for Azure
                "max_tokens_per_batch": 8191 * 40,
                "parallel_batches": 2,
                "memory_limit_mb": 512
            },
            "google_gemini": {
                "max_batch_size": 100,  # Conservative estimate for Gemini API
                "optimal_batch_size": 50,
                "max_tokens_per_batch": 2048 * 50,  # Based on model max tokens
                "parallel_batches": 3,
                "memory_limit_mb": 512
            },
            "huggingface": {
                "max_batch_size": 32,  # Depends on model and GPU memory
                "optimal_batch_size": 16,
                "max_tokens_per_batch": 512 * 16,
                "parallel_batches": 1,  # Usually single GPU
                "memory_limit_mb": 2048  # Higher for local models
            },
            "google_use": {
                "max_batch_size": 64,
                "optimal_batch_size": 32,
                "max_tokens_per_batch": 512 * 32,
                "parallel_batches": 1,
                "memory_limit_mb": 1024
            },
            "sentence_transformers": {
                "max_batch_size": 32,
                "optimal_batch_size": 16,
                "max_tokens_per_batch": 512 * 16,
                "parallel_batches": 1,
                "memory_limit_mb": 1024
            }
        }
        
        return configs.get(provider.lower(), {
            "max_batch_size": 32,
            "optimal_batch_size": 16,
            "max_tokens_per_batch": 512 * 16,
            "parallel_batches": 1,
            "memory_limit_mb": 512
        })


class ProviderHealthCheckConfigs:
    """Health check configurations for different providers."""
    
    @staticmethod
    def get_health_check_config(provider: str) -> Dict[str, Any]:
        """
        Get health check configuration for a specific provider.
        
        Args:
            provider: Provider name
            
        Returns:
            Dictionary with health check configuration
        """
        configs = {
            "openai": {
                "test_text": "Health check test.",
                "timeout": 30.0,
                "expected_dimension": None,  # Will be determined dynamically
                "check_interval": 300.0,  # 5 minutes
                "failure_threshold": 3
            },
            "cohere": {
                "test_text": "Health check test.",
                "timeout": 20.0,
                "expected_dimension": None,
                "check_interval": 300.0,
                "failure_threshold": 3
            },
            "azure_openai": {
                "test_text": "Health check test.",
                "timeout": 45.0,
                "expected_dimension": None,
                "check_interval": 300.0,
                "failure_threshold": 4
            },
            "google_gemini": {
                "test_text": "Health check test.",
                "timeout": 60.0,
                "expected_dimension": 768,  # Gemini embedding dimension
                "check_interval": 300.0,
                "failure_threshold": 3
            },
            "huggingface": {
                "test_text": "Health check test.",
                "timeout": 60.0,  # Local models may take longer to load
                "expected_dimension": None,
                "check_interval": 600.0,  # 10 minutes
                "failure_threshold": 2
            },
            "google_use": {
                "test_text": "Health check test.",
                "timeout": 45.0,
                "expected_dimension": 512,  # USE typically has 512 dimensions
                "check_interval": 600.0,
                "failure_threshold": 2
            },
            "sentence_transformers": {
                "test_text": "Health check test.",
                "timeout": 30.0,
                "expected_dimension": None,
                "check_interval": 600.0,
                "failure_threshold": 2
            }
        }
        
        return configs.get(provider.lower(), {
            "test_text": "Health check test.",
            "timeout": 30.0,
            "expected_dimension": None,
            "check_interval": 300.0,
            "failure_threshold": 3
        })
