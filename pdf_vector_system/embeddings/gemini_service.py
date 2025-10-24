"""Google Gemini embedding service implementation."""

import os
import time
from typing import Any, Optional, cast

import requests  # type: ignore[import-untyped]

from pdf_vector_system.embeddings.base import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
)
from pdf_vector_system.embeddings.provider_configs import ProviderRetryConfigs
from pdf_vector_system.embeddings.retry import FailureType, RetryableError, RetryHandler
from pdf_vector_system.utils.progress import PerformanceTimer


class GeminiEmbeddingService(EmbeddingService):
    """Embedding service using Google Gemini's embedding API."""

    # Model configurations
    MODEL_CONFIGS: dict[str, dict[str, Any]] = {
        "gemini-embedding-001": {
            "dimension": 768,
            "max_tokens": 2048,
            "description": "Latest Gemini embedding model",
        },
        "gemini-embedding-exp-03-07": {
            "dimension": 768,
            "max_tokens": 2048,
            "description": "Experimental Gemini embedding model",
        },
        "text-embedding-004": {
            "dimension": 768,
            "max_tokens": 2048,
            "description": "Text embedding model v4",
        },
    }

    # API endpoints
    GOOGLE_AI_BASE_URL: str = "https://generativelanguage.googleapis.com/v1beta"
    VERTEX_AI_BASE_URL: str = (
        "https://{location}-aiplatform.googleapis.com/v1/projects/{project_id}/locations/{location}/publishers/google/models"
    )

    def __init__(
        self,
        model_name: str = "gemini-embedding-001",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        project_id: Optional[str] = None,
        location: str = "us-central1",
        use_vertex_ai: bool = False,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 100,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Gemini embedding service.

        Args:
            model_name: Name of the Gemini embedding model
            api_key: Google AI API key (for Google AI API) or service account key (for Vertex AI)
            base_url: Custom base URL for the API
            project_id: Google Cloud project ID (required for Vertex AI)
            location: Google Cloud location (for Vertex AI)
            use_vertex_ai: Whether to use Vertex AI endpoint instead of Google AI API
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for API requests
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.max_retries: int = max_retries
        self.timeout: float = timeout
        self.batch_size: int = min(batch_size, 100)  # Conservative batch size
        self.use_vertex_ai: bool = use_vertex_ai
        self.project_id: Optional[str] = project_id
        self.location: str = location

        # Get API key
        self.api_key: Optional[str] = (
            api_key
            or os.getenv("GOOGLE_GEMINI_API_KEY")
            or os.getenv("GOOGLE_AI_API_KEY")
        )
        if not self.api_key:
            raise ValueError(
                "Google Gemini API key is required. Set GOOGLE_GEMINI_API_KEY or GOOGLE_AI_API_KEY "
                "environment variable or pass api_key parameter."
            )

        # Set up base URL
        if base_url:
            self.base_url: str = base_url
        elif use_vertex_ai:
            if not project_id:
                raise ValueError("project_id is required when using Vertex AI")
            self.base_url = self.VERTEX_AI_BASE_URL.format(
                location=location, project_id=project_id
            )
        else:
            self.base_url = self.GOOGLE_AI_BASE_URL

        # Initialize retry handler
        retry_config = ProviderRetryConfigs.get_google_gemini_config()
        self.retry_handler: RetryHandler = RetryHandler(
            retry_config, f"GeminiService-{model_name}"
        )

        # Validate model
        if model_name not in self.MODEL_CONFIGS:
            self.logger.warning(
                f"Model {model_name} not in known configurations. "
                f"Known models: {list(self.MODEL_CONFIGS.keys())}"
            )

        self.logger.info(
            f"Initialized Gemini embedding service with model: {model_name}, "
            f"endpoint: {'Vertex AI' if use_vertex_ai else 'Google AI API'}"
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
        validated_texts: list[str] = self.validate_texts(texts)

        start_time: float = time.time()
        all_embeddings: list[list[float]] = []
        total_tokens: int = 0

        try:
            # Process texts in batches
            with PerformanceTimer(
                f"Generating Gemini embeddings for {len(texts)} texts", log_result=False
            ):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts: list[str] = validated_texts[i : i + self.batch_size]

                    self.logger.debug(
                        f"Processing batch {i // self.batch_size + 1} with {len(batch_texts)} texts"
                    )

                    # Make API request with retry logic
                    response_data: dict[str, Any] = self.retry_handler.execute(
                        self._make_embedding_request, batch_texts
                    )

                    # Extract embeddings
                    batch_embeddings: list[list[float]] = self._extract_embeddings(
                        response_data
                    )
                    all_embeddings.extend(batch_embeddings)

                    # Track token usage if available
                    usage_data = response_data.get("usage")
                    if usage_data and isinstance(usage_data, dict):
                        token_count = usage_data.get("totalTokens")
                        if isinstance(token_count, int):
                            total_tokens += token_count

            processing_time: float = time.time() - start_time

            # Get embedding dimension
            embedding_dim: int = len(all_embeddings[0]) if all_embeddings else 0

            result: EmbeddingResult = EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                token_count=total_tokens if total_tokens > 0 else None,
                metadata={
                    "api_provider": "google_gemini",
                    "endpoint_type": "vertex_ai" if self.use_vertex_ai else "google_ai",
                    "project_id": self.project_id,
                    "location": self.location,
                    "batch_count": (len(texts) + self.batch_size - 1)
                    // self.batch_size,
                    "batch_size": self.batch_size,
                    "total_tokens": total_tokens,
                    "tokens_per_text": (
                        total_tokens / len(texts) if total_tokens > 0 and texts else 0
                    ),
                },
            )

            self.logger.debug(
                f"Generated {len(all_embeddings)} Gemini embeddings "
                f"(dim: {embedding_dim}, tokens: {total_tokens}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            error_msg: str = f"Failed to generate Gemini embeddings: {e!s}"
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
            response_data: dict[str, Any] = self.retry_handler.execute(
                self._make_embedding_request, [text.strip()]
            )
            embeddings: list[list[float]] = self._extract_embeddings(response_data)
            return embeddings[0]

        except Exception as e:
            error_msg: str = (
                f"Failed to generate Gemini embedding for single text: {e!s}"
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
        model_config = self.MODEL_CONFIGS.get(self.model_name)
        if model_config:
            dimension = model_config.get("dimension")
            if isinstance(dimension, int):
                return dimension

        # Fallback: generate a test embedding
        try:
            test_embedding: list[float] = self.embed_single("test")
            return len(test_embedding)
        except Exception as e:
            error_msg: str = f"Failed to determine embedding dimension: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def _make_embedding_request(self, texts: list[str]) -> dict[str, Any]:
        """
        Make an embedding request to Gemini API.

        Args:
            texts: List of texts to embed

        Returns:
            API response data
        """
        try:
            if self.use_vertex_ai:
                return self._make_vertex_ai_request(texts)
            return self._make_google_ai_request(texts)

        except Exception as e:
            # Classify the error for retry logic
            error_str: str = str(e).lower()

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

    def _make_google_ai_request(self, texts: list[str]) -> dict[str, Any]:
        """Make request to Google AI API."""
        url: str = f"{self.base_url}/models/{self.model_name}:embedContent"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            "x-goog-api-key": cast(
                "str", self.api_key
            ),  # api_key is validated in __init__
        }

        # Prepare requests for each text
        requests_data: list[dict[str, Any]] = []
        for text in texts:
            requests_data.append(
                {
                    "model": f"models/{self.model_name}",
                    "content": {"parts": [{"text": text}]},
                }
            )

        # For batch requests, use the batch endpoint
        payload: dict[str, Any]
        if len(texts) > 1:
            payload = {"requests": requests_data}
            url = f"{self.base_url}:batchEmbedContents"
        else:
            payload = requests_data[0]

        response: requests.Response = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )

        if not response.ok:
            raise Exception(f"Gemini API error {response.status_code}: {response.text}")

        return cast("dict[str, Any]", response.json())

    def _make_vertex_ai_request(self, texts: list[str]) -> dict[str, Any]:
        """Make request to Vertex AI API."""
        url: str = f"{self.base_url}/{self.model_name}:predict"

        headers: dict[str, str] = {
            "Content-Type": "application/json",
            # Assumes service account token
            "Authorization": f"Bearer {cast('str', self.api_key)}",
        }

        # Vertex AI format
        instances: list[dict[str, str]] = []
        for text in texts:
            instances.append({"content": text})

        payload: dict[str, list[dict[str, str]]] = {"instances": instances}

        response: requests.Response = requests.post(
            url, headers=headers, json=payload, timeout=self.timeout
        )

        if not response.ok:
            raise Exception(f"Vertex AI error {response.status_code}: {response.text}")

        return cast("dict[str, Any]", response.json())

    def _extract_embeddings(self, response_data: dict[str, Any]) -> list[list[float]]:
        """
        Extract embeddings from API response.

        Args:
            response_data: API response data

        Returns:
            List of embedding vectors
        """
        embeddings: list[list[float]] = []

        if self.use_vertex_ai:
            # Vertex AI response format
            predictions = response_data.get("predictions")
            if predictions and isinstance(predictions, list):
                for prediction in predictions:
                    if not isinstance(prediction, dict):
                        continue

                    # Check for embeddings format
                    embedding_data = prediction.get("embeddings")
                    if embedding_data and isinstance(embedding_data, dict):
                        values = embedding_data.get("values")
                        if values and isinstance(values, list):
                            embeddings.append(values)
                    # Check for direct values format
                    elif "values" in prediction:
                        values = prediction["values"]
                        if values and isinstance(values, list):
                            embeddings.append(values)
        else:
            # Google AI API response format
            batch_embeddings = response_data.get("embeddings")
            if batch_embeddings and isinstance(batch_embeddings, list):
                # Batch response
                for embedding_data in batch_embeddings:
                    if isinstance(embedding_data, dict) and "values" in embedding_data:
                        values = embedding_data["values"]
                        if values and isinstance(values, list):
                            embeddings.append(values)
            else:
                # Single response
                single_embedding = response_data.get("embedding")
                if single_embedding and isinstance(single_embedding, dict):
                    values = single_embedding.get("values")
                    if values and isinstance(values, list):
                        embeddings.append(values)

        if not embeddings:
            raise EmbeddingServiceError("No embeddings found in API response")

        return embeddings

    def get_model_info(self) -> dict[str, Any]:
        """
        Get comprehensive information about the model.

        Returns:
            Dictionary containing model information
        """
        base_info: dict[str, Any] = super().get_model_info()

        model_config: dict[str, Any] = self.MODEL_CONFIGS.get(self.model_name, {})
        max_tokens: Optional[int] = None
        if "max_tokens" in model_config:
            max_tokens_value = model_config["max_tokens"]
            if isinstance(max_tokens_value, int):
                max_tokens = max_tokens_value

        return {
            **base_info,
            "api_provider": "google_gemini",
            "endpoint_type": "vertex_ai" if self.use_vertex_ai else "google_ai",
            "base_url": self.base_url,
            "project_id": self.project_id,
            "location": self.location,
            "max_tokens": max_tokens,
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
        }

    def health_check(self) -> bool:
        """
        Perform a health check on the Gemini embedding service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a simple text
            test_text: str = "This is a test."
            embedding: list[float] = self.embed_single(test_text)

            # Validate the result
            expected_dim: int = self.get_embedding_dimension()
            if not embedding or len(embedding) != expected_dim:
                return False

            # Check if embedding contains valid numbers
            return all(isinstance(x, (int, float)) for x in embedding)

        except Exception as e:
            self.logger.error(f"Gemini health check failed: {e!s}")
            return False
