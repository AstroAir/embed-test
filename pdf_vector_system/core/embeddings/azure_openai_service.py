"""Azure OpenAI embedding service implementation."""

import os
import time
from typing import Any, Optional

from pdf_vector_system.core.embeddings.base import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
)
from pdf_vector_system.core.embeddings.provider_configs import ProviderRetryConfigs
from pdf_vector_system.core.embeddings.retry import (
    FailureType,
    RetryableError,
    RetryHandler,
)
from pdf_vector_system.core.utils.progress import PerformanceTimer


class AzureOpenAIEmbeddingService(EmbeddingService):
    """Embedding service using Azure OpenAI's embedding API."""

    # Model configurations (same as OpenAI but may have different deployment names)
    MODEL_CONFIGS = {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191},
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        endpoint: Optional[str] = None,
        api_version: str = "2024-02-01",
        deployment_name: Optional[str] = None,
        max_retries: int = 4,
        timeout: float = 60.0,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Azure OpenAI embedding service.

        Args:
            model_name: Name of the embedding model
            api_key: Azure OpenAI API key
            endpoint: Azure OpenAI endpoint URL
            api_version: Azure OpenAI API version
            deployment_name: Azure deployment name (defaults to model_name)
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for API requests
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = min(
            batch_size, 100
        )  # Azure OpenAI limit is similar to OpenAI
        self.api_version = api_version
        self.deployment_name = deployment_name or model_name

        # Get API key and endpoint
        self.api_key = api_key or os.getenv("AZURE_OPENAI_API_KEY")
        self.endpoint = endpoint or os.getenv("AZURE_OPENAI_ENDPOINT")

        if not self.api_key:
            raise ValueError(
                "Azure OpenAI API key is required. Set AZURE_OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        if not self.endpoint:
            raise ValueError(
                "Azure OpenAI endpoint is required. Set AZURE_OPENAI_ENDPOINT environment variable "
                "or pass endpoint parameter."
            )

        # Ensure endpoint format is correct
        if not self.endpoint.startswith("https://"):
            self.endpoint = f"https://{self.endpoint}"
        if not self.endpoint.endswith("/"):
            self.endpoint = f"{self.endpoint}/"

        # Initialize Azure OpenAI client
        try:
            from openai import AzureOpenAI

            self.client = AzureOpenAI(
                api_key=self.api_key,
                azure_endpoint=self.endpoint,
                api_version=self.api_version,
                timeout=timeout,
                max_retries=0,  # We handle retries ourselves
            )

            self.logger.info(
                f"Initialized Azure OpenAI client with model: {model_name}, "
                f"deployment: {self.deployment_name}, endpoint: {self.endpoint}"
            )

        except ImportError as e:
            error_msg = "OpenAI package is required. Install with: pip install openai"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize Azure OpenAI client: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

        # Initialize retry handler
        retry_config = ProviderRetryConfigs.get_azure_openai_config()
        self.retry_handler = RetryHandler(
            retry_config, f"AzureOpenAIService-{model_name}"
        )

        # Validate model
        if model_name not in self.MODEL_CONFIGS:
            self.logger.warning(
                f"Model {model_name} not in known configurations. "
                f"Known models: {list(self.MODEL_CONFIGS.keys())}"
            )

    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult containing the generated embeddings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        # Validate and preprocess texts
        validated_texts = self.validate_texts(texts)

        start_time = time.time()
        all_embeddings = []
        total_tokens = 0

        try:
            # Process texts in batches
            with PerformanceTimer(
                f"Generating Azure OpenAI embeddings for {len(texts)} texts",
                log_result=False,
            ):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i : i + self.batch_size]

                    self.logger.debug(
                        f"Processing batch {i // self.batch_size + 1} with {len(batch_texts)} texts"
                    )

                    # Make API request with retry logic
                    response = self.retry_handler.execute(
                        self._make_embedding_request, batch_texts
                    )

                    # Extract embeddings and token usage
                    batch_embeddings = [item.embedding for item in response.data]
                    all_embeddings.extend(batch_embeddings)

                    if hasattr(response, "usage") and response.usage:
                        total_tokens += response.usage.total_tokens

            processing_time = time.time() - start_time

            # Get embedding dimension
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 0

            result = EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                token_count=total_tokens,
                metadata={
                    "api_provider": "azure_openai",
                    "deployment_name": self.deployment_name,
                    "api_version": self.api_version,
                    "endpoint": self.endpoint,
                    "batch_count": (len(texts) + self.batch_size - 1)
                    // self.batch_size,
                    "batch_size": self.batch_size,
                    "total_tokens": total_tokens,
                    "tokens_per_text": total_tokens / len(texts) if texts else 0,
                },
            )

            self.logger.debug(
                f"Generated {len(all_embeddings)} Azure OpenAI embeddings "
                f"(dim: {embedding_dim}, tokens: {total_tokens}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to generate Azure OpenAI embeddings: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """
        if not text or not text.strip():
            raise ValueError("Text cannot be empty")

        try:
            response = self.retry_handler.execute(
                self._make_embedding_request, [text.strip()]
            )
            return list(response.data[0].embedding)

        except Exception as e:
            error_msg = (
                f"Failed to generate Azure OpenAI embedding for single text: {e!s}"
            )
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension
        """
        # Try to get from model configuration
        if self.model_name in self.MODEL_CONFIGS:
            return self.MODEL_CONFIGS[self.model_name]["dimension"]

        # Fallback: generate a test embedding
        try:
            test_embedding = self.embed_single("test")
            return len(test_embedding)
        except Exception as e:
            error_msg = f"Failed to determine embedding dimension: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def _make_embedding_request(self, texts: list[str]) -> Any:
        """
        Make an embedding request to Azure OpenAI API.

        Args:
            texts: List of texts to embed

        Returns:
            Azure OpenAI API response
        """
        try:
            return self.client.embeddings.create(
                model=self.deployment_name,  # Use deployment name for Azure
                input=texts,
                encoding_format="float",
            )

        except Exception as e:
            # Classify the error for retry logic
            error_str = str(e).lower()

            if "rate limit" in error_str or "429" in error_str:
                raise RetryableError(str(e), FailureType.RATE_LIMIT) from e
            if "quota" in error_str or "limit" in error_str:
                raise RetryableError(str(e), FailureType.QUOTA_EXCEEDED) from e
            if "timeout" in error_str:
                raise RetryableError(str(e), FailureType.TIMEOUT) from e
            if "401" in error_str or "403" in error_str:
                raise RetryableError(str(e), FailureType.AUTHENTICATION) from e
            if any(code in error_str for code in ["500", "502", "503", "504"]):
                raise RetryableError(str(e), FailureType.SERVER_ERROR) from e
            if "network" in error_str or "connection" in error_str:
                raise RetryableError(str(e), FailureType.NETWORK_ERROR) from e
            raise RetryableError(str(e), FailureType.UNKNOWN) from e

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()

        model_config = self.MODEL_CONFIGS.get(self.model_name, {})

        return {
            **base_info,
            "api_provider": "azure_openai",
            "deployment_name": self.deployment_name,
            "endpoint": self.endpoint,
            "api_version": self.api_version,
            "max_tokens": model_config.get("max_tokens"),
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> bool:
        """
        Perform a health check on the Azure OpenAI embedding service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a simple text
            test_text = "This is a test."
            embedding = self.embed_single(test_text)

            # Validate the result
            expected_dim = self.get_embedding_dimension()
            if not embedding or len(embedding) != expected_dim:
                return False

            # Check if embedding contains valid numbers
            return all(isinstance(x, (int, float)) for x in embedding)

        except Exception as e:
            self.logger.error(f"Azure OpenAI health check failed: {e!s}")
            return False
