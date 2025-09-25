"""Cohere embedding service implementation."""

import os
import time
from typing import List, Dict, Any, Optional

from loguru import logger

from .base import EmbeddingService, EmbeddingResult, EmbeddingServiceError, ModelNotFoundError
from .retry import RetryHandler, RetryableError, FailureType
from .provider_configs import ProviderRetryConfigs
from ..utils.progress import PerformanceTimer


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
        model_name: str = "embed-english-v3.0",
        api_key: Optional[str] = None,
        base_url: Optional[str] = None,
        max_retries: int = 3,
        timeout: float = 60.0,
        batch_size: int = 96,
        input_type: str = "search_document",
        **kwargs: Any
    ) -> None:
        """
        Initialize Cohere embedding service.
        
        Args:
            model_name: Name of the Cohere embedding model
            api_key: Cohere API key (if None, will try to get from environment)
            base_url: Custom base URL for Cohere API
            max_retries: Maximum number of retries for failed requests
            timeout: Request timeout in seconds
            batch_size: Maximum batch size for API requests
            input_type: Type of input for embeddings (search_document, search_query, etc.)
            **kwargs: Additional configuration parameters
        """
        super().__init__(model_name, **kwargs)
        
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
            import cohere
            self.client = cohere.Client(
                api_key=self.api_key,
                base_url=base_url,
                timeout=timeout
            )
            
            self.logger.info(f"Initialized Cohere client with model: {model_name}")
            
        except ImportError as e:
            error_msg = "Cohere package is required. Install with: pip install cohere"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
        except Exception as e:
            error_msg = f"Failed to initialize Cohere client: {str(e)}"
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
            # Process texts in batches
            with PerformanceTimer(f"Generating Cohere embeddings for {len(texts)} texts", log_result=False):
                for i in range(0, len(validated_texts), self.batch_size):
                    batch_texts = validated_texts[i:i + self.batch_size]
                    
                    self.logger.debug(f"Processing batch {i//self.batch_size + 1} with {len(batch_texts)} texts")
                    
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
                    "batch_count": (len(texts) + self.batch_size - 1) // self.batch_size,
                    "batch_size": self.batch_size,
                    "input_type": self.input_type
                }
            )
            
            self.logger.debug(
                f"Generated {len(all_embeddings)} Cohere embeddings "
                f"(dim: {embedding_dim}) in {processing_time:.2f}s"
            )
            
            return result
            
        except Exception as e:
            error_msg = f"Failed to generate Cohere embeddings: {str(e)}"
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
                self._make_embedding_request, [text.strip()]
            )
            return embeddings[0]
            
        except Exception as e:
            error_msg = f"Failed to generate Cohere embedding for single text: {str(e)}"
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
            error_msg = f"Failed to determine embedding dimension: {str(e)}"
            self.logger.error(error_msg)
            raise EmbeddingServiceError(error_msg) from e
    
    def _make_embedding_request(self, texts: List[str]) -> List[List[float]]:
        """
        Make an embedding request to Cohere API.

        Args:
            texts: List of texts to embed

        Returns:
            List of embedding vectors
        """
        try:
            response = self.client.embed(
                texts=texts,
                model=self.model_name,
                input_type=self.input_type
            )

            # Extract embeddings from response
            # The response.embeddings should be a List[List[float]] according to Cohere SDK
            embeddings: List[List[float]] = response.embeddings

            if not embeddings:
                raise RetryableError("Empty embeddings returned", FailureType.UNKNOWN)

            return embeddings
            
        except Exception as e:
            # Classify the error for retry logic
            error_str = str(e).lower()
            
            if "rate limit" in error_str or "429" in error_str:
                raise RetryableError(str(e), FailureType.RATE_LIMIT) from e
            elif "quota" in error_str or "limit" in error_str:
                raise RetryableError(str(e), FailureType.QUOTA_EXCEEDED) from e
            elif "timeout" in error_str:
                raise RetryableError(str(e), FailureType.TIMEOUT) from e
            elif "401" in error_str or "403" in error_str:
                raise RetryableError(str(e), FailureType.AUTHENTICATION) from e
            elif any(code in error_str for code in ["500", "502", "503", "504"]):
                raise RetryableError(str(e), FailureType.SERVER_ERROR) from e
            elif "network" in error_str or "connection" in error_str:
                raise RetryableError(str(e), FailureType.NETWORK_ERROR) from e
            else:
                raise RetryableError(str(e), FailureType.UNKNOWN) from e
    
    def get_model_info(self) -> Dict[str, Any]:
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
            if not all(isinstance(x, (int, float)) for x in embedding):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"Cohere health check failed: {str(e)}")
            return False
