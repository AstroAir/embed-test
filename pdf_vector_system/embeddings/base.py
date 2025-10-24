"""Base classes and interfaces for embedding services."""

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional, Union

import numpy as np

from pdf_vector_system.utils.logging import LoggerMixin


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""

    embeddings: list[list[float]]
    model_name: str
    embedding_dimension: int
    processing_time: float = 0.0
    token_count: Optional[int] = None
    metadata: Optional[dict[str, Any]] = None
    texts: Optional[list[str]] = None

    def __post_init__(self) -> None:
        """Validate embedding result after initialization."""
        # Allow empty embeddings for empty input cases
        # if not self.embeddings:
        #     raise ValueError("Embeddings list cannot be empty")

        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")

        if self.processing_time < 0:
            raise ValueError("processing_time must be non-negative")

        # Validate metadata type if provided
        if self.metadata is not None and not isinstance(self.metadata, dict):
            raise TypeError(f"metadata must be a dict, got {type(self.metadata)}")

        # Validate that embeddings and texts have the same length if texts provided
        if self.texts is not None and len(self.embeddings) != len(self.texts):
            raise ValueError("embeddings and texts must have the same length")

        # Validate that all embeddings have the same dimension (if any embeddings exist)
        if self.embeddings:
            for i, embedding in enumerate(self.embeddings):
                if len(embedding) != self.embedding_dimension:
                    raise ValueError(
                        f"Embedding {i} has dimension {len(embedding)}, "
                        f"expected {self.embedding_dimension}"
                    )

    @property
    def count(self) -> int:
        """Get the number of embeddings."""
        return len(self.embeddings)

    @property
    def text_count(self) -> int:
        """Get the number of texts that were embedded."""
        return len(self.embeddings)

    @property
    def texts_per_second(self) -> float:
        """Calculate texts processed per second."""
        if self.processing_time <= 0:
            return 0.0
        return self.text_count / self.processing_time

    def to_numpy(self) -> np.ndarray:
        """Convert embeddings to numpy array."""
        return np.array(self.embeddings)

    def get_embedding(self, index: int) -> list[float]:
        """Get a specific embedding by index."""
        if index < 0 or index >= len(self.embeddings):
            raise IndexError(f"Embedding index {index} out of range")
        return self.embeddings[index]


@dataclass
class EmbeddingBatch:
    """Batch of texts for embedding generation."""

    texts: list[str]
    batch_id: Optional[str] = None
    metadata: Optional[Union[dict[str, Any], list[dict[str, Any]]]] = None
    batch_size: Optional[int] = None

    def __post_init__(self) -> None:
        """Validate batch after initialization."""
        # Validate that texts is a list
        if not isinstance(self.texts, list):
            raise TypeError(f"texts must be a list, got {type(self.texts)}")

        if not self.texts:
            raise ValueError("texts cannot be empty")

        # Validate that all texts are strings
        for i, text in enumerate(self.texts):
            if not isinstance(text, str):
                raise TypeError(
                    f"All texts must be strings, got {type(text)} at index {i}"
                )

        # Auto-generate batch_id if not provided
        if self.batch_id is None:
            import uuid

            self.batch_id = str(uuid.uuid4())

        # Initialize metadata as empty dict if not provided
        if self.metadata is None:
            self.metadata = {}

        # Validate batch_size if provided
        if self.batch_size is not None and self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

    @property
    def size(self) -> int:
        """Get the batch size."""
        return len(self.texts)

    def get_text(self, index: int) -> str:
        """Get text at specific index."""
        if index < 0 or index >= len(self.texts):
            raise IndexError(f"Text index {index} out of range")
        return self.texts[index]

    def get_metadata(self, index: int) -> dict[str, Any]:
        """Get metadata at specific index."""
        if self.metadata is None:
            return {}
        # Handle both dict (batch-level) and list (per-text) metadata
        if isinstance(self.metadata, list):
            if index < 0 or index >= len(self.metadata):
                return {}
            return self.metadata[index]
        # For dict, return the same metadata for all texts
        return self.metadata


class EmbeddingService(ABC, LoggerMixin):
    """Abstract base class for embedding services."""

    def __init__(self, model_name: str, **kwargs: Any) -> None:
        """
        Initialize embedding service.

        Args:
            model_name: Name of the embedding model
            **kwargs: Additional configuration parameters
        """
        self.model_name = model_name
        self.config = kwargs
        self._embedding_dimension: Optional[int] = None
        self.logger.info(
            f"Initialized {self.__class__.__name__} with model: {model_name}"
        )

    @abstractmethod
    def embed_texts(self, texts: list[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.

        Args:
            texts: List of texts to embed

        Returns:
            EmbeddingResult containing the generated embeddings
        """

    @abstractmethod
    def embed_single(self, text: str) -> list[float]:
        """
        Generate embedding for a single text.

        Args:
            text: Text to embed

        Returns:
            Embedding vector as list of floats
        """

    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.

        Returns:
            Embedding dimension
        """

    @property
    def embedding_dimension(self) -> int:
        """Get the embedding dimension (cached)."""
        if self._embedding_dimension is None:
            self._embedding_dimension = self.get_embedding_dimension()
        return self._embedding_dimension

    def embed_batch(self, batch: EmbeddingBatch) -> EmbeddingResult:
        """
        Generate embeddings for a batch of texts.

        Args:
            batch: EmbeddingBatch containing texts and metadata

        Returns:
            EmbeddingResult with embeddings and metadata
        """
        start_time = time.time()

        self.logger.debug(f"Processing batch of {batch.size} texts")

        result = self.embed_texts(batch.texts)

        # Add batch metadata to result
        if batch.metadata:
            result.metadata = result.metadata or {}
            result.metadata["batch_metadata"] = batch.metadata
            result.metadata["batch_id"] = batch.batch_id

        processing_time = time.time() - start_time
        result.processing_time = processing_time

        self.logger.debug(
            f"Batch processing completed in {processing_time:.2f}s "
            f"({batch.size / processing_time:.1f} texts/sec)"
        )

        return result

    def validate_texts(self, texts: list[str]) -> list[str]:
        """
        Validate and preprocess texts before embedding.

        Args:
            texts: List of texts to validate

        Returns:
            List of validated texts
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")

        validated_texts = []
        for i, text in enumerate(texts):
            if not isinstance(text, str):
                raise TypeError(f"Text at index {i} must be a string, got {type(text)}")

            # Remove excessive whitespace
            cleaned_text = " ".join(text.split())

            if not cleaned_text:
                self.logger.warning(f"Empty text at index {i}, using placeholder")
                cleaned_text = "[EMPTY]"

            validated_texts.append(cleaned_text)

        return validated_texts

    def get_model_info(self) -> dict[str, Any]:
        """
        Get information about the embedding model.

        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "service_type": self.__class__.__name__,
            "config": self.config,
        }

    def health_check(self) -> bool:
        """
        Perform a health check on the embedding service.

        Returns:
            True if service is healthy, False otherwise
        """
        try:
            # Test with a simple text
            test_text = "This is a test."
            embedding = self.embed_single(test_text)

            # Validate the result
            if not embedding or len(embedding) != self.embedding_dimension:
                return False

            # Check if embedding contains valid numbers
            return all(
                isinstance(x, (int, float)) and not np.isnan(x) for x in embedding
            )

        except Exception as e:
            self.logger.error(f"Health check failed: {e!s}")
            return False


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""


class ModelNotFoundError(EmbeddingServiceError):
    """Raised when the specified model is not found."""


class EmbeddingGenerationError(EmbeddingServiceError):
    """Raised when embedding generation fails."""


class InvalidInputError(EmbeddingServiceError):
    """Raised when input validation fails."""
