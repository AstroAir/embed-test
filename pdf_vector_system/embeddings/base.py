"""Base classes and interfaces for embedding services."""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Union
from dataclasses import dataclass
import time

import numpy as np
from loguru import logger

from ..utils.logging import LoggerMixin


@dataclass
class EmbeddingResult:
    """Result of embedding generation."""
    embeddings: List[List[float]]
    model_name: str
    embedding_dimension: int
    processing_time: float
    token_count: Optional[int] = None
    metadata: Optional[Dict[str, Any]] = None
    
    def __post_init__(self) -> None:
        """Validate embedding result after initialization."""
        if not self.embeddings:
            raise ValueError("Embeddings list cannot be empty")

        if self.embedding_dimension <= 0:
            raise ValueError("Embedding dimension must be positive")
        
        # Validate that all embeddings have the same dimension
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
    
    def to_numpy(self) -> np.ndarray:
        """Convert embeddings to numpy array."""
        return np.array(self.embeddings)
    
    def get_embedding(self, index: int) -> List[float]:
        """Get a specific embedding by index."""
        if index < 0 or index >= len(self.embeddings):
            raise IndexError(f"Embedding index {index} out of range")
        return self.embeddings[index]


@dataclass
class EmbeddingBatch:
    """Batch of texts for embedding generation."""
    texts: List[str]
    batch_id: Optional[str] = None
    metadata: Optional[List[Dict[str, Any]]] = None
    
    def __post_init__(self) -> None:
        """Validate batch after initialization."""
        if not self.texts:
            raise ValueError("Texts list cannot be empty")

        if self.metadata is not None and len(self.metadata) != len(self.texts):
            raise ValueError("Metadata list must have same length as texts list")
    
    @property
    def size(self) -> int:
        """Get the batch size."""
        return len(self.texts)
    
    def get_text(self, index: int) -> str:
        """Get text at specific index."""
        if index < 0 or index >= len(self.texts):
            raise IndexError(f"Text index {index} out of range")
        return self.texts[index]
    
    def get_metadata(self, index: int) -> Optional[Dict[str, Any]]:
        """Get metadata at specific index."""
        if self.metadata is None:
            return None
        if index < 0 or index >= len(self.metadata):
            raise IndexError(f"Metadata index {index} out of range")
        return self.metadata[index]


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
        self.logger.info(f"Initialized {self.__class__.__name__} with model: {model_name}")
    
    @abstractmethod
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """
        Generate embeddings for a list of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            EmbeddingResult containing the generated embeddings
        """
        pass
    
    @abstractmethod
    def embed_single(self, text: str) -> List[float]:
        """
        Generate embedding for a single text.
        
        Args:
            text: Text to embed
            
        Returns:
            Embedding vector as list of floats
        """
        pass
    
    @abstractmethod
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension
        """
        pass
    
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
    
    def validate_texts(self, texts: List[str]) -> List[str]:
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
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get information about the embedding model.
        
        Returns:
            Dictionary containing model information
        """
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "service_type": self.__class__.__name__,
            "config": self.config
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
            if not all(isinstance(x, (int, float)) and not np.isnan(x) for x in embedding):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Health check failed: {str(e)}")
            return False


class EmbeddingServiceError(Exception):
    """Base exception for embedding service errors."""
    pass


class ModelNotFoundError(EmbeddingServiceError):
    """Raised when the specified model is not found."""
    pass


class EmbeddingGenerationError(EmbeddingServiceError):
    """Raised when embedding generation fails."""
    pass


class InvalidInputError(EmbeddingServiceError):
    """Raised when input validation fails."""
    pass
