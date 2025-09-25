"""Mock implementations for embedding services."""

from typing import List, Dict, Any, Optional
from unittest.mock import Mock, MagicMock
import time
import numpy as np

from pdf_vector_system.embeddings.base import EmbeddingResult, EmbeddingBatch


class MockEmbeddingService:
    """Base mock embedding service for testing."""
    
    def __init__(self, model_name: str = "mock-model", embedding_dim: int = 5):
        self.model_name = model_name
        self.embedding_dimension = embedding_dim
        self._embedding_dimension = embedding_dim
        self.config = {}
        self.call_count = 0
        self.last_texts = []
    
    def embed_texts(self, texts: List[str]) -> EmbeddingResult:
        """Generate mock embeddings for a list of texts."""
        self.call_count += 1
        self.last_texts = texts.copy()
        
        # Generate deterministic mock embeddings based on text content
        embeddings = []
        for i, text in enumerate(texts):
            # Create a simple hash-based embedding
            text_hash = hash(text) % 1000
            embedding = [(text_hash + j) / 1000.0 for j in range(self.embedding_dimension)]
            embeddings.append(embedding)
        
        return EmbeddingResult(
            embeddings=embeddings,
            model_name=self.model_name,
            embedding_dimension=self.embedding_dimension,
            processing_time=0.1,
            token_count=sum(len(text.split()) for text in texts),
            metadata={
                "mock_service": True,
                "call_count": self.call_count,
                "text_count": len(texts)
            }
        )
    
    def embed_single(self, text: str) -> List[float]:
        """Generate mock embedding for a single text."""
        result = self.embed_texts([text])
        return result.embeddings[0]
    
    def get_embedding_dimension(self) -> int:
        """Get the embedding dimension."""
        return self.embedding_dimension
    
    def health_check(self) -> bool:
        """Mock health check - always returns True."""
        return True
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        return {
            "model_name": self.model_name,
            "embedding_dimension": self.embedding_dimension,
            "model_type": "mock",
            "is_mock": True,
            "call_count": self.call_count
        }


class MockSentenceTransformersService(MockEmbeddingService):
    """Mock SentenceTransformers service for testing."""
    
    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
        device: Optional[str] = None,
        embedding_dim: int = 384
    ):
        super().__init__(model_name, embedding_dim)
        self.device = device or "cpu"
        self._model = Mock()
        self._model.device = self.device
        self._model.max_seq_length = 512
        
        # Mock the model's encode method
        def mock_encode(texts, **kwargs):
            embeddings = []
            for text in texts:
                text_hash = hash(text) % 1000
                embedding = np.array([(text_hash + j) / 1000.0 for j in range(self.embedding_dimension)])
                embeddings.append(embedding)
            return np.array(embeddings)
        
        self._model.encode = mock_encode
    
    @property
    def model(self):
        """Get the mock model."""
        return self._model
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        base_info = super().get_model_info()
        return {
            **base_info,
            "device": self.device,
            "max_seq_length": 512,
            "model_loaded": True,
            "model_type": "sentence-transformers"
        }


class MockOpenAIService(MockEmbeddingService):
    """Mock OpenAI embedding service for testing."""
    
    def __init__(
        self,
        model_name: str = "text-embedding-3-small",
        api_key: Optional[str] = None,
        embedding_dim: int = 1536
    ):
        super().__init__(model_name, embedding_dim)
        self.api_key = api_key or "mock-api-key"
        self.batch_size = 100
        self.max_retries = 3
        self.timeout = 60.0
        self.request_count = 0
        
        # Mock OpenAI client
        self._client = Mock()
        
        def mock_create_embedding(input, model, **kwargs):
            self.request_count += 1
            
            # Mock response structure
            mock_response = Mock()
            mock_response.data = []
            
            for i, text in enumerate(input):
                mock_embedding = Mock()
                text_hash = hash(text) % 1000
                mock_embedding.embedding = [(text_hash + j) / 1000.0 for j in range(self.embedding_dimension)]
                mock_response.data.append(mock_embedding)
            
            mock_response.usage = Mock()
            mock_response.usage.total_tokens = sum(len(text.split()) for text in input)
            
            return mock_response
        
        self._client.embeddings.create = mock_create_embedding
    
    @property
    def client(self):
        """Get the mock OpenAI client."""
        return self._client
    
    def get_model_info(self) -> Dict[str, Any]:
        """Get mock model information."""
        base_info = super().get_model_info()
        return {
            **base_info,
            "api_provider": "openai",
            "batch_size": self.batch_size,
            "max_retries": self.max_retries,
            "timeout": self.timeout,
            "has_api_key": bool(self.api_key),
            "request_count": self.request_count
        }


class MockBatchEmbeddingProcessor:
    """Mock batch embedding processor for testing."""
    
    def __init__(self, embedding_service: MockEmbeddingService, max_workers: int = 4):
        self.embedding_service = embedding_service
        self.max_workers = max_workers
        self.batch_size = getattr(embedding_service, 'batch_size', 32)
        self.processed_batches = []
    
    def process_texts(
        self,
        texts: List[str],
        show_progress: bool = True,
        metadata: Optional[Dict[str, Any]] = None
    ) -> EmbeddingResult:
        """Process texts in batches."""
        start_time = time.time()
        
        # Split into batches
        batches = [texts[i:i + self.batch_size] for i in range(0, len(texts), self.batch_size)]
        self.processed_batches = batches.copy()
        
        # Process all batches (mock parallel processing)
        all_embeddings = []
        total_tokens = 0
        
        for batch in batches:
            result = self.embedding_service.embed_texts(batch)
            all_embeddings.extend(result.embeddings)
            if result.token_count:
                total_tokens += result.token_count
        
        processing_time = time.time() - start_time
        
        return EmbeddingResult(
            embeddings=all_embeddings,
            model_name=self.embedding_service.model_name,
            embedding_dimension=self.embedding_service.embedding_dimension,
            processing_time=processing_time,
            token_count=total_tokens if total_tokens > 0 else None,
            metadata={
                "batch_processor": True,
                "total_batches": len(batches),
                "batch_size": self.batch_size,
                "max_workers": self.max_workers,
                "texts_per_second": len(texts) / processing_time if processing_time > 0 else 0
            }
        )


def create_mock_embedding_service(
    service_type: str = "sentence_transformers",
    model_name: Optional[str] = None,
    **kwargs
) -> MockEmbeddingService:
    """Factory function to create mock embedding services."""
    if service_type == "sentence_transformers":
        return MockSentenceTransformersService(
            model_name=model_name or "all-MiniLM-L6-v2",
            **kwargs
        )
    elif service_type == "openai":
        return MockOpenAIService(
            model_name=model_name or "text-embedding-3-small",
            **kwargs
        )
    else:
        return MockEmbeddingService(
            model_name=model_name or "mock-model",
            **kwargs
        )


def create_mock_batch_processor(
    service_type: str = "sentence_transformers",
    max_workers: int = 4,
    **kwargs
) -> MockBatchEmbeddingProcessor:
    """Factory function to create mock batch embedding processor."""
    service = create_mock_embedding_service(service_type, **kwargs)
    return MockBatchEmbeddingProcessor(service, max_workers)
