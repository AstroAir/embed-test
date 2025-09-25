"""Hugging Face transformers embedding service implementation."""

import time
import warnings
from typing import List, Dict, Any, Optional, Union
import psutil
import gc

import numpy as np
from loguru import logger

from .base import EmbeddingService, EmbeddingResult, EmbeddingServiceError, ModelNotFoundError
from .retry import RetryHandler, RetryableError, FailureType
from .provider_configs import ProviderRetryConfigs
from ..utils.progress import PerformanceTimer


class HuggingFaceEmbeddingService(EmbeddingService):
    """Embedding service using Hugging Face transformers library."""
    
    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        device: Optional[str] = None,
        cache_dir: Optional[str] = None,
        trust_remote_code: bool = False,
        batch_size: int = 16,
        max_length: int = 512,
        normalize_embeddings: bool = True,
        **kwargs: Any
    ) -> None:
        """
        Initialize Hugging Face embedding service.
        
        Args:
            model_name: Name of the Hugging Face model
            device: Device to run the model on ('cpu', 'cuda', etc.)
            cache_dir: Directory to cache downloaded models
            trust_remote_code: Whether to trust remote code in models
            batch_size: Batch size for processing
            max_length: Maximum sequence length
            normalize_embeddings: Whether to normalize embeddings
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
        self.device = device
        self.cache_dir = cache_dir
        self.trust_remote_code = trust_remote_code
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        self._model: Optional[Any] = None
        self._tokenizer: Optional[Any] = None
        self._device_name: Optional[str] = None
        
        # Suppress some warnings from transformers
        warnings.filterwarnings("ignore", category=FutureWarning, module="transformers")
        
        # Initialize retry handler
        retry_config = ProviderRetryConfigs.get_local_model_config()
        self.retry_handler = RetryHandler(retry_config, f"HuggingFaceService-{model_name}")
        
        self.logger.info(f"Configured HuggingFaceEmbeddingService with model: {model_name}")
    
    @property
    def model(self) -> Any:
        """Get or load the Hugging Face model."""
        if self._model is None:
            self._load_model()
        return self._model
    
    @property
    def tokenizer(self) -> Any:
        """Get or load the tokenizer."""
        if self._tokenizer is None:
            self._load_model()
        return self._tokenizer
    
    def _load_model(self) -> None:
        """Load the Hugging Face model and tokenizer."""
        try:
            from transformers import AutoModel, AutoTokenizer
            import torch
            
            with PerformanceTimer(f"Loading HuggingFace model {self.model_name}"):
                # Determine device
                if self.device is None:
                    if torch.cuda.is_available():
                        self._device_name = "cuda"
                    else:
                        self._device_name = "cpu"
                else:
                    self._device_name = self.device
                
                self.logger.info(f"Loading model on device: {self._device_name}")
                
                # Load tokenizer
                self._tokenizer = AutoTokenizer.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=self.trust_remote_code
                )
                
                # Load model
                self._model = AutoModel.from_pretrained(
                    self.model_name,
                    cache_dir=self.cache_dir,
                    trust_remote_code=self.trust_remote_code
                )
                
                # Move to device
                assert self._device_name is not None  # Should be set above
                self._model = self._model.to(self._device_name)
                self._model.eval()  # Set to evaluation mode
                
                self.logger.info(
                    f"Loaded HuggingFace model {self.model_name} on device: {self._device_name}"
                )
                
        except ImportError as e:
            error_msg = (
                "Hugging Face transformers and torch are required. "
                "Install with: pip install transformers torch"
            )
            self.logger.error(error_msg)
            raise ModelNotFoundError(error_msg) from e
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
        all_embeddings = []
        
        try:
            # Monitor memory usage
            initial_memory = psutil.virtual_memory().percent
            
            # Process texts in batches
            with PerformanceTimer(f"Generating HuggingFace embeddings for {len(texts)} texts", log_result=False):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i:i + self.batch_size]
                    
                    self.logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_texts)} texts")
                    
                    # Generate embeddings for batch with retry logic
                    batch_embeddings = self.retry_handler.execute(
                        self._generate_batch_embeddings, batch_texts
                    )
                    all_embeddings.extend(batch_embeddings)
                    
                    # Memory management
                    current_memory = psutil.virtual_memory().percent
                    if current_memory - initial_memory > 20:  # If memory usage increased by 20%
                        self.logger.debug("High memory usage detected, running garbage collection")
                        gc.collect()
                        if self._device_name == "cuda":
                            import torch
                            torch.cuda.empty_cache()
            
            processing_time = time.time() - start_time
            
            # Get embedding dimension
            embedding_dim = len(all_embeddings[0]) if all_embeddings else 0
            
            result = EmbeddingResult(
                embeddings=all_embeddings,
                model_name=self.model_name,
                embedding_dimension=embedding_dim,
                processing_time=processing_time,
                metadata={
                    "device": self._device_name,
                    "model_type": "huggingface",
                    "normalized": self.normalize_embeddings,
                    "batch_size": self.batch_size,
                    "max_length": self.max_length,
                    "batch_count": (len(texts) + self.batch_size - 1) // self.batch_size
                }
            )
            
            self.logger.debug(
                f"Generated {len(all_embeddings)} HuggingFace embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate HuggingFace embeddings: {str(e)}"
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
            embeddings = self.retry_handler.execute(
                self._generate_batch_embeddings, [text.strip()]
            )
            return embeddings[0]
            
        except Exception as e:
            error_msg = f"Failed to generate HuggingFace embedding for single text: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def get_embedding_dimension(self) -> int:
        """
        Get the dimension of embeddings produced by this service.
        
        Returns:
            Embedding dimension
        """
        try:
            # Get dimension from model configuration
            config = self.model.config
            if hasattr(config, 'hidden_size'):
                return int(config.hidden_size)
            elif hasattr(config, 'd_model'):
                return int(config.d_model)
            else:
                # Fallback: generate a test embedding
                test_embedding = self.embed_single("test")
                return len(test_embedding)
                
        except Exception as e:
            error_msg = f"Failed to determine embedding dimension: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def _generate_batch_embeddings(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts.
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            import torch
            
            # Tokenize texts
            inputs = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt"
            )
            
            # Move inputs to device
            inputs = {k: v.to(self._device_name) for k, v in inputs.items()}
            
            # Generate embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                
                # Use mean pooling of last hidden states
                embeddings = self._mean_pooling(outputs, inputs['attention_mask'])
                
                # Normalize if requested
                if self.normalize_embeddings:
                    embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
                
                # Convert to list
                embeddings_list: List[List[float]] = embeddings.cpu().numpy().tolist()

            return embeddings_list
            
        except Exception as e:
            # Classify error for retry logic
            error_str = str(e).lower()
            
            if "out of memory" in error_str or "cuda out of memory" in error_str:
                raise RetryableError(str(e), FailureType.UNKNOWN) from e
            elif "timeout" in error_str:
                raise RetryableError(str(e), FailureType.TIMEOUT) from e
            else:
                raise RetryableError(str(e), FailureType.UNKNOWN) from e
    
    def _mean_pooling(self, model_output: Any, attention_mask: Any) -> Any:
        """Apply mean pooling to get sentence embeddings."""
        import torch
        
        token_embeddings = model_output[0]  # First element contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)
    
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
                "device": self._device_name,
                "model_type": "huggingface",
                "batch_size": self.batch_size,
                "max_length": self.max_length,
                "normalize_embeddings": self.normalize_embeddings,
                "cache_dir": self.cache_dir,
                "trust_remote_code": self.trust_remote_code,
                "model_loaded": self._model is not None,
            }
            
            # Add model-specific info if loaded
            if self._model is not None:
                config = self.model.config
                model_info.update({
                    "model_config": {
                        "hidden_size": getattr(config, 'hidden_size', None),
                        "num_attention_heads": getattr(config, 'num_attention_heads', None),
                        "num_hidden_layers": getattr(config, 'num_hidden_layers', None),
                        "vocab_size": getattr(config, 'vocab_size', None),
                    }
                })
            
            return model_info
            
        except Exception as e:
            self.logger.warning(f"Error getting extended model info: {str(e)}")
            return base_info
    
    def unload_model(self) -> None:
        """Unload the model to free memory."""
        if self._model is not None:
            del self._model
            self._model = None
        
        if self._tokenizer is not None:
            del self._tokenizer
            self._tokenizer = None
        
        # Clear GPU cache if using CUDA
        if self._device_name == "cuda":
            try:
                import torch
                torch.cuda.empty_cache()
            except ImportError:
                pass
        
        # Run garbage collection
        gc.collect()
        
        self.logger.info(f"Unloaded HuggingFace model {self.model_name}")
    
    def __del__(self) -> None:
        """Cleanup when object is destroyed."""
        try:
            self.unload_model()
        except Exception:
            pass  # Ignore errors during cleanup
