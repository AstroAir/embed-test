"""Sentence Transformers embedding service implementation."""

import time
from typing import List, Dict, Any, Optional
import warnings

import numpy as np
from sentence_transformers import SentenceTransformer
from loguru import logger

from .base import EmbeddingService, EmbeddingResult, EmbeddingServiceError, ModelNotFoundError
from ..utils.progress import PerformanceTimer


class SentenceTransformersService(EmbeddingService):
    """Embedding service using sentence-transformers library."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_folder: Optional[str] = None,
        trust_remote_code: bool = False,
        **kwargs: Any
    ) -> None:
        """
        Initialize sentence-transformers embedding service.
        
        Args:
            model_name: Name of the sentence-transformers model
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_folder: Folder to cache downloaded models
            trust_remote_code: Whether to trust remote code in models
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.device = device
        self.cache_folder = cache_folder
        self.trust_remote_code = trust_remote_code
        self._model: Optional[SentenceTransformer] = None
        
        # Suppress some warnings from sentence-transformers
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        
        self.logger.info(f"Configured SentenceTransformersService with model: {model_name}")
    
    @property
    def model(self) -> SentenceTransformer:
        """Get or load the sentence transformer model."""
        if self._model is None:
            self._load_model()
        if self._model is None:
            raise RuntimeError("Failed to load model - model is still None after loading")
        return self._model
    
    def _load_model(self) -> None:
        """Load the sentence transformer model."""
        try:
            with PerformanceTimer(f"Loading model {self.model_name}"):
                self._model = SentenceTransformer(
                    self.model_name,
                    device=self.device,
                    cache_folder=self.cache_folder,
                    trust_remote_code=self.trust_remote_code
                )
                
                self.logger.info(
                    f"Loaded model {self.model_name} on device: {self._model.device}"
                )
                
        except Exception as e:
            error_msg = f"Failed to load model {self.model_name}: {str(e)}"
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
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
        
        try:
            # Generate embeddings
            with PerformanceTimer(f"Generating embeddings for {len(texts)} texts", log_result=False):
                embeddings = self.model.encode(
                    validated_texts,
                    convert_to_numpy=True,
                    show_progress_bar=len(texts) > 10,  # Show progress for larger batches
                    batch_size=32,  # Default batch size
                    normalize_embeddings=True  # Normalize for better similarity search
                )
            
            processing_time = time.time() - start_time
            
            # Convert numpy array to list of lists
            embeddings_list = embeddings.tolist()
            
            # Get model information
            embedding_dim = embeddings.shape[1]
            
            result = EmbeddingResult(
                embeddings=embeddings_list,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                metadata={
                    "device": str(self.model.device),
                    "model_type": "sentence-transformers",
                    "normalized": True,
                    "batch_size": len(texts)
                }
            )
            
            self.logger.debug(
                f"Generated {len(embeddings_list)} embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate embeddings: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def embed_single(self, text: str) -> List[float]:
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
            # Generate embedding for single text
            embedding = self.model.encode(
                [text.strip()],
                convert_to_numpy=True,
                normalize_embeddings=True
            )
            
            return list(embedding[0].tolist())
            
        except Exception as e:
            error_msg = f"Failed to generate embedding for single text: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension
        """
        try:
            # Get dimension from model
            dimension = self.model.get_sentence_embedding_dimension()
            if dimension is None:
                raise ValueError("Model returned None for embedding dimension")
            return int(dimension)
            
        except Exception as e:
            # Fallback: generate a test embedding to get dimension
            try:
                test_embedding = self.embed_single("test")
                return len(test_embedding)
            except Exception:
                error_msg = f"Failed to determine embedding dimension: {str(e)}"
                self.logger.error(error_msg)
                raise EmbeddingServiceError(error_msg) from e
    
    def encode_batch(
        self,
        texts: List[str],
        batch_size: int = 32,
        show_progress: bool = True,
        normalize: bool = True
    ) -> np.ndarray:
        """
        Encode texts in batches with more control over the process.
        
        Args:
            texts: List of texts to encode
            batch_size: Size of each batch
            show_progress: Whether to show progress bar
            normalize: Whether to normalize embeddings
            
        Returns:
            Numpy array of embeddings
        """
        if not texts:
            raise ValueError("Texts list cannot be empty")
        
        validated_texts = self.validate_texts(texts)
        
        try:
            embeddings = self.model.encode(
                validated_texts,
                batch_size=batch_size,
                show_progress_bar=show_progress,
                convert_to_numpy=True,
                normalize_embeddings=normalize
            )
            
            self.logger.debug(
                f"Encoded {len(texts)} texts in batches of {batch_size} "
                f"(shape: {embeddings.shape})"
            )
            
            return embeddings
            
        except Exception as e:
            error_msg = f"Failed to encode batch: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def similarity(self, embeddings1: np.ndarray, embeddings2: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of embeddings.
        
        Args:
            embeddings1: First set of embeddings
            embeddings2: Second set of embeddings
            
        Returns:
            Similarity matrix
        """
        try:
            from sentence_transformers.util import cos_sim
            return cos_sim(embeddings1, embeddings2).numpy()
            
        except Exception as e:
            error_msg = f"Failed to compute similarity: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def get_model_info(self) -> Dict[str, Any]:
        """
        Get comprehensive information about the model.
        
        Returns:
            Dictionary containing model information
        """
        base_info = super().get_model_info()
        
        try:
            model_info = {
                **base_info,
                "device": str(self.model.device) if self._model else self.device,
                "max_seq_length": getattr(self.model, "max_seq_length", None),
                "model_loaded": self._model is not None,
                "cache_folder": self.cache_folder,
                "trust_remote_code": self.trust_remote_code,
            }
            
            # Add tokenizer info if available
            if self._model and hasattr(self.model, "tokenizer"):
                model_info["tokenizer_vocab_size"] = len(self.model.tokenizer)
            
            return model_info
            
        except Exception as e:
            self.logger.warning(f"Error getting extended model info: {str(e)}")
            return base_info
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
            self.logger.info(f"Unloaded model {self.model_name}")
    
    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass  # Ignore errors during cleanup
