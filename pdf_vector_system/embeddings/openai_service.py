"""OpenAI embedding service implementation."""

import os
import time
from typing import Any, Optional

import openai
from openai import OpenAI

from pdf_vector_system.embeddings.base import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
)
from pdf_vector_system.utils.progress import PerformanceTimer


class OpenAIEmbeddingService(EmbeddingService):
    """Embedding service using OpenAI's embedding API."""

    # Model configurations
    MODEL_CONFIGS = {
        "text-embedding-3-small": {"dimension": 1536, "max_tokens": 8191},
        "text-embedding-3-large": {"dimension": 3072, "max_tokens": 8191},
        "text-embedding-ada-002": {"dimension": 1536, "max_tokens": 8191},
    }

    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize OpenAI embedding service.

        Args:
            model_name: Name of the OpenAI embedding model
            api_key: OpenAI API key (if None, will try to get from environment)
            base_url: Custom base URL for OpenAI API
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for API requests
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.max_retries = max_retries
        self.timeout = timeout
        self.batch_size = min(batch_size, 100)  # OpenAI limit is 100

        # Get API key
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "OpenAI API key is required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )

        # Initialize OpenAI client
        try:
            self.client = OpenAI(
                api_key=self.api_key,
                base_url=base_url,
                timeout=timeout,
                max_retries=max_retries,
            )

            self.logger.info(f"Initialized OpenAI client with model: {model_name}")

        except Exception as e:
            error_msg = f"Failed to initialize OpenAI client: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

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
                f"Generating OpenAI embeddings for {len(texts)} texts", log_result=False
            ):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i : i + self.batch_size]

                    self.logger.debug(
                        f"Processing batch {i // self.batch_size + 1} with {len(batch_texts)} texts"
                    )

                    # Make API request
                    response = self._make_embedding_request(batch_texts)

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
                    "api_provider": "openai",
                    "batch_count": (len(texts) + self.batch_size - 1)
                    // self.batch_size,
                    "batch_size": self.batch_size,
                    "total_tokens": total_tokens,
                    "tokens_per_text": total_tokens / len(texts) if texts else 0,
                },
            )

            self.logger.debug(
                f"Generated {len(all_embeddings)} OpenAI embeddings "
                f"(dim: {embedding_dim}, tokens: {total_tokens}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to generate OpenAI embeddings: {e!s}"
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
            response = self._make_embedding_request([text.strip()])
            return list(response.data[0].embedding)

        except Exception as e:
            error_msg = f"Failed to generate OpenAI embedding for single text: {e!s}"
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
        Make an embedding request to OpenAI API with retry logic.

        Args:
            texts: List of texts to embed

        Returns:
            OpenAI API response
        """
        for attempt in range(self.max_retries + 1):
            try:
                return self.client.embeddings.create(
                    model=self.model_name, input=texts, encoding_format="float"
                )

            except openai.RateLimitError as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt  # Exponential backoff
                    self.logger.warning(
                        f"Rate limit hit, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(wait_time)
                    continue
                raise EmbeddingServiceError(
                    f"Rate limit exceeded after {self.max_retries} retries"
                ) from e

            except openai.APIError as e:
                status_code = getattr(e, "status_code", None)
                if attempt < self.max_retries and status_code and status_code >= 500:
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"API error {status_code}, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}"
                    )
                    time.sleep(wait_time)
                    continue
                raise EmbeddingServiceError(f"OpenAI API error: {e!s}") from e

            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2**attempt
                    self.logger.warning(
                        f"Request failed, waiting {wait_time}s before retry {attempt + 1}/{self.max_retries}: {e!s}"
                    )
                    time.sleep(wait_time)
                    continue
                raise EmbeddingServiceError(
                    f"Request failed after {self.max_retries} retries: {e!s}"
                ) from e
        return None

    def estimate_cost(self, texts: list[str]) -> dict[str, Any]:
        """
        Estimate the cost of embedding the given texts.

        Args:
            texts: List of texts to estimate cost for

        Returns:
            Dictionary containing cost estimation
        """
        # Rough token estimation (actual tokenization would be more accurate)
        estimated_tokens = sum(
            len(text.split()) * 1.3 for text in texts
        )  # Rough approximation

        # Pricing (as of 2024 - should be updated)
        pricing = {
            "text-embedding-3-small": 0.00002 / 1000,  # $0.00002 per 1K tokens
            "text-embedding-3-large": 0.00013 / 1000,  # $0.00013 per 1K tokens
            "text-embedding-ada-002": 0.0001 / 1000,  # $0.0001 per 1K tokens
        }

        price_per_token = pricing.get(
            self.model_name, 0.0001 / 1000
        )  # Default fallback
        estimated_cost = estimated_tokens * price_per_token

        return {
            "estimated_tokens": int(estimated_tokens),
            "estimated_cost_usd": round(estimated_cost, 6),
            "model": self.model_name,
            "text_count": len(texts),
            "note": "This is a rough estimation. Actual cost may vary.",
        }

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
            "api_provider": "openai",
            "max_tokens": model_config.get("max_tokens"),
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> bool:
        """
        Perform a health check on the OpenAI embedding service.

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
            self.logger.error(f"OpenAI health check failed: {e!s}")
            return False
