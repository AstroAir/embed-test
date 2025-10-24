"""Google Universal Sentence Encoder embedding service implementation."""

import gc
import time
import warnings
from typing import Any, Optional

import numpy as np

from pdf_vector_system.embeddings.base import (
    EmbeddingResult,
    EmbeddingService,
    EmbeddingServiceError,
    ModelNotFoundError,
)
from pdf_vector_system.embeddings.provider_configs import ProviderRetryConfigs
from pdf_vector_system.embeddings.retry import FailureType, RetryableError, RetryHandler
from pdf_vector_system.utils.progress import PerformanceTimer

# Import TensorFlow Hub at module level for test mocking
try:
    import tensorflow_hub as hub
except ImportError:
    hub = None


class GoogleUSEService(EmbeddingService):
    """Embedding service using Google Universal Sentence Encoder via TensorFlow Hub."""

    # Model configurations
    MODEL_CONFIGS = {
        "universal-sentence-encoder": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder/4",
            "dimension": 512,
            "version": "4",
        },
        "universal-sentence-encoder-large": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder-large/5",
            "dimension": 512,
            "version": "5",
        },
        "universal-sentence-encoder-multilingual": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder-multilingual/3",
            "dimension": 512,
            "version": "3",
        },
        "universal-sentence-encoder-multilingual-large": {
            "url": "https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3",
            "dimension": 512,
            "version": "3",
        },
    }

    def __init__(
        self,
        model_name: str = "universal-sentence-encoder",
        cache_dir: Optional[str] = None,
        version: str = "4",
        batch_size: int = 32,
        **kwargs: Any,
    ) -> None:
        """
        Initialize Google USE embedding service.

        Args:
            model_name: Name of the USE model
            cache_dir: Directory to cache downloaded models
            version: Model version to use
            batch_size: Batch size for processing
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)

        self.cache_dir = cache_dir
        self.version = version
        self.batch_size = batch_size

        self._model: Optional[Any] = None
        self._session: Optional[Any] = None

        # Suppress TensorFlow warnings
        warnings.filterwarnings("ignore", category=FutureWarning, module="tensorflow")

        # Initialize retry handler
        retry_config = ProviderRetryConfigs.get_local_model_config()
        self.retry_handler = RetryHandler(
            retry_config, f"GoogleUSEService-{model_name}"
        )

        self.logger.info(f"Configured GoogleUSEService with model: {model_name}")

    @property
    def model(self) -> Any:
        """Get or load the Google USE model."""
        if self._model is None:
            self._load_model()
        return self._model

    def _load_model(self) -> None:
        """Load the Google USE model from TensorFlow Hub."""
        try:
            import tensorflow as tf

            if hub is None:
                raise ImportError("TensorFlow Hub is not available")

            # Configure TensorFlow
            tf.get_logger().setLevel("ERROR")  # Reduce TF logging

            # Set cache directory if provided
            if self.cache_dir:
                import os

                os.environ["TFHUB_CACHE_DIR"] = self.cache_dir

            with PerformanceTimer(f"Loading Google USE model {self.model_name}"):
                # Get model URL
                model_config = self.MODEL_CONFIGS.get(self.model_name)
                if not model_config:
                    # Fallback: construct URL from model name and version
                    model_url = (
                        f"https://tfhub.dev/google/{self.model_name}/{self.version}"
                    )
                else:
                    assert model_config is not None  # Checked above
                    model_url = str(model_config["url"])

                self.logger.info(f"Loading model from: {model_url}")

                # Load model
                self._model = hub.load(model_url)

                self.logger.info(f"Loaded Google USE model {self.model_name}")

        except ImportError as e:
            error_msg = (
                "TensorFlow and TensorFlow Hub are required. "
                "Install with: pip install tensorflow tensorflow-hub"
            )
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to load Google USE model {self.model_name}: {e!s}"
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e

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

        try:
            # Process texts in batches
            with PerformanceTimer(
                f"Generating Google USE embeddings for {len(texts)} texts",
                log_result=False,
            ):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i : i + self.batch_size]

                    self.logger.debug(
                        f"Processing batch {i // self.batch_size + 1} with {len(batch_texts)} texts"
                    )

                    # Generate embeddings for batch with retry logic
                    batch_embeddings = self.retry_handler.execute(
                        self._generate_batch_embeddings, batch_texts
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
                    "model_type": "google_use",
                    "version": self.version,
                    "batch_size": self.batch_size,
                    "batch_count": (len(texts) + self.batch_size - 1)
                    // self.batch_size,
                    "cache_dir": self.cache_dir,
                },
            )

            self.logger.debug(
                f"Generated {len(all_embeddings)} Google USE embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )

            return result

        except Exception as e:
            error_msg = f"Failed to generate Google USE embeddings: {e!s}"
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
                self._generate_batch_embeddings, [text.strip()]
            )
            return embeddings[0]

        except Exception as e:
            error_msg = (
                f"Failed to generate Google USE embedding for single text: {e!s}"
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
            dimension = model_config["dimension"]
            return int(dimension) if isinstance(dimension, (int, str)) else 512

        # Fallback: generate a test embedding
        try:
            test_embedding = self.embed_single("test")
            return len(test_embedding)
        except Exception as e:
            error_msg = f"Failed to determine embedding dimension: {e!s}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e

    def _generate_batch_embeddings(self, texts: list[str]) -> list[list[float]]:
        """
        Generate embeddings for a batch of texts.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            import tensorflow as tf

            # Convert texts to tensor
            text_tensor = tf.constant(texts)

            # Generate embeddings
            embeddings = self.model(text_tensor)

            # Convert to numpy and then to list
            embeddings_np = embeddings.numpy()
            embeddings_list: list[list[float]] = embeddings_np.tolist()

            return embeddings_list

        except Exception as e:
            # Classify error for retry logic
            error_str = str(e).lower()

            if "out of memory" in error_str or "resource exhausted" in error_str:
                raise RetryableError(str(e), FailureType.UNKNOWN) from e
            if "timeout" in error_str:
                raise RetryableError(str(e), FailureType.TIMEOUT) from e
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
            "model_type": "google_use",
            "version": self.version,
            "batch_size": self.batch_size,
            "cache_dir": self.cache_dir,
            "model_loaded": self._model is not None,
            "model_url": model_config.get("url"),
        }

    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None

        # Run garbage collection
        gc.collect()

        self.logger.info(f"Unloaded Google USE model {self.model_name}")

    def health_check(self) -> bool:
        """
        Perform a health check on the Google USE service.

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
            return all(
                isinstance(x, (int, float)) and not np.isnan(x) for x in embedding
            )

        except Exception as e:
            self.logger.error(f"Google USE health check failed: {e!s}")
            return False

    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass  # Ignore errors during cleanup
