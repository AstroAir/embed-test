"""Cohere embedding service implementation."""

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

# Module-level import for testing purposes
try:
    import cohere
except ImportError:
    cohere = None


class CohereEmbeddingService(EmbeddingService):
    """Embedding service using Cohere's embedding API."""

    # Model configurations
    MODEL_CONFIGS = {
        "embed-english-v3.0": {"dimension": 1024, "max_tokens": 512},
        "embed-multilingual-v3.0": {"dimension": 1024, "max_tokens": 512},
        "embed-english-light-v3.0": {"dimension": 384, "max_tokens": 512},
        "embed-multilingual-light-v3.0": {"dimension": 384, "max_tokens": 512},
        "embed-english-v2.0": {"dimension": 4096, "max_tokens": 512},
        "embed-multilingual-v2.0": {"dimension": 768, "max_tokens": 512},
    }

    def __init__(
        self,
        config_or_model_name="embed-english-v3.0",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 96,
        input_type: str = "search_document",
        **kwargs: Any,
    ) -> None:
        """
        Initialize Cohere embedding service.

        Args:
            config_or_model_name: Either an EmbeddingConfig object or model name string
            api_key: Cohere API key (if None, will try to get from environment)
            base_url: Custom base URL for Cohere API
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for API requests
            input_type: Type of input for embeddings (search_document, search_query, etc.)
            **kwargs: Additional configuration parameters
        """
        # Handle both EmbeddingConfig objects and individual parameters
        from pdf_vector_system.core.config.settings import EmbeddingConfig

        if isinstance(config_or_model_name, EmbeddingConfig):
            # Using EmbeddingConfig object
            config_obj = config_or_model_name
            model_name = config_or_model_name.model_name
            api_key = api_key or config_or_model_name.cohere_api_key
            base_url = base_url or config_or_model_name.cohere_base_url
            max_retries = config_or_model_name.max_retries
            timeout = config_or_model_name.timeout_seconds
            batch_size = config_or_model_name.batch_size
        else:
            # Using individual parameters
            config_obj = None
            model_name = config_or_model_name

        super().__init__(model_name, **kwargs)

        # Set config after parent init (parent overwrites self.config)
        if config_obj is not None:
            self.config = config_obj

        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = min(batch_size, 96)  # Cohere limit is 96
        self.input_type = input_type

        # Get API key
        self.api_key = api_key or os.getenv("COHERE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Cohere API key is required. Set COHERE_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize Cohere client
        try:
            if cohere is None:
                raise ImportError("Cohere package not available")

            self.client = cohere.Client(
                api_key=self.api_key, base_url=base_url, timeout=timeout
            )

            self.logger.info(f"Initialized Cohere client with model: {model_name}")

        except ImportError as e:
            error_msg = "Cohere package is required. Install with: pip install cohere"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize Cohere client: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

        # Initialize retry handler
        retry_config = ProviderRetryConfigs.get_cohere_config()
        self.retry_handler = RetryHandler(retry_config, f"CohereService-{model_name}")

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
        all_embeddings: list[list[float]] = []

        try:
            # Process texts in batches
            with PerformanceTimer(
                f"Generating Cohere embeddings for {len(texts)} texts", log_result=False
            ):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i : i + self.batch_size]

                    self.logger.debug(
                        f"Processing batch {i // self.batch_size + 1} with {len(batch_texts)} texts"
                    )

                    # Make API request with retry logic
                    batch_embeddings = self.retry_handler.execute(
                        self._make_embedding_request, batch_texts
                    )
                    all_embeddings.extend(batch_embeddings)

            processing_time = time.time() - start_time

            # Get embedding dimension
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 0

            result = EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                metadata={
                    "api_provider": "cohere",
                    "batch_count": (len(texts) + self.batch_size - 1)
                    // self.batch_size,
                    "batch_size": self.batch_size,
                    "input_type": self.input_type,
                },
            )

            self.logger.debug(
                f"Generated {len(all_embeddings)} Cohere embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to generate Cohere embeddings: {e!s}"
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
            embeddings = self.retry_handler.execute(
                self._make_embedding_request, [text.strip()]
            )
            return embeddings[0]

        except Exception as e:
            error_msg = f"Failed to generate Cohere embedding for single text: {e!s}"
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

    def _make_embedding_request(self, texts: list[str]) -> list[list[float]]:
        """
        Make an embedding request to Cohere API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embed(
                texts=texts, model=self.model_name, input_type=self.input_type
            )
            embeddings_raw = response.embeddings  # Expected list-like

            # Normalize to List[List[float]]
            if not isinstance(embeddings_raw, list):
                try:
                    embeddings_iter = list(embeddings_raw)
                except TypeError as err:
                    raise RetryableError(
                        "Embeddings response not iterable", FailureType.UNKNOWN
                    ) from err
            else:
                embeddings_iter = embeddings_raw

            embeddings: list[list[float]] = []
            for idx, emb in enumerate(embeddings_iter):
                if not isinstance(emb, (list, tuple)):
                    raise RetryableError(
                        f"Embedding at index {idx} has invalid type {type(emb)}",
                        FailureType.UNKNOWN,
                    )
                try:
                    embeddings.append([float(x) for x in emb])
                except (TypeError, ValueError) as err:
                    raise RetryableError(
                        f"Embedding at index {idx} contains non-numeric values",
                        FailureType.UNKNOWN,
                    ) from err

            if not embeddings:
                raise RetryableError("Empty embeddings returned", FailureType.UNKNOWN)
            return embeddings

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
            "api_provider": "cohere",
            "max_tokens": model_config.get("max_tokens"),
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "input_type": self.input_type,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> bool:
        """
        Perform a health check on the Cohere embedding service.

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
            self.logger.error(f"Cohere health check failed: {e!s}")
            return False
