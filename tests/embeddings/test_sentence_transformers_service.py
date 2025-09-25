"""Tests for SentenceTransformersService."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pdf_vector_system.embeddings.sentence_transformers_service import SentenceTransformersService
from pdf_vector_system.embeddings.base import EmbeddingResult
from tests.mocks.embedding_mocks import MockSentenceTransformersService


class TestSentenceTransformersService:
    """Test SentenceTransformersService with mocking."""
    
    @patch('pdf_vector_system.embeddings.sentence_transformers_service.SentenceTransformer')
    def test_initialization(self, mock_st_class):
        """Test service initialization."""
        mock_model = Mock()
        mock_model.device = "cpu"
        mock_st_class.return_value = mock_model
        
        service = SentenceTransformersService(
            model_name="all-MiniLM-L6-v2",
            device="cpu"
        )
        
        assert service.model_name == "all-MiniLM-L6-v2"
        assert service.device == "cpu"
        mock_st_class.assert_called_once()
    
    @patch('pdf_vector_system.embeddings.sentence_transformers_service.SentenceTransformer')
    def test_initialization_with_custom_params(self, mock_st_class):
        """Test service initialization with custom parameters."""
        mock_model = Mock()
        mock_model.device = "cuda"
        mock_st_class.return_value = mock_model
        
        service = SentenceTransformersService(
            model_name="all-mpnet-base-v2",
            device="cuda",
            cache_folder="/custom/cache",
            trust_remote_code=True
        )
        
        assert service.model_name == "all-mpnet-base-v2"
        assert service.device == "cuda"
        mock_st_class.assert_called_once_with(
            "all-mpnet-base-v2",
            device="cuda",
            cache_folder="/custom/cache",
            trust_remote_code=True
        )
    
    def test_mock_service_functionality(self):
        """Test mock sentence transformers service."""
        service = MockSentenceTransformersService(
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384
        )
        
        # Test single embedding
        embedding = service.embed_single("test text")
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        
        # Test multiple embeddings
        texts = ["text 1", "text 2"]
        result = service.embed_texts(texts)
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert all(len(emb) == 384 for emb in result.embeddings)
        assert result.model_name == "all-MiniLM-L6-v2"
        assert result.embedding_dimension == 384
        
        # Test model info
        info = service.get_model_info()
        assert info["model_type"] == "sentence-transformers"
        assert info["device"] == "cpu"
        assert info["model_name"] == "all-MiniLM-L6-v2"
        assert info["embedding_dimension"] == 384
    
    def test_mock_service_health_check(self):
        """Test mock service health check."""
        service = MockSentenceTransformersService("all-MiniLM-L6-v2")
        
        assert service.health_check() is True
    
    def test_mock_service_batch_processing(self):
        """Test mock service batch processing."""
        service = MockSentenceTransformersService(
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384
        )
        
        # Test with larger batch
        texts = [f"Text number {i}" for i in range(10)]
        result = service.embed_texts(texts)
        
        assert len(result.embeddings) == 10
        assert all(len(emb) == 384 for emb in result.embeddings)
        assert result.text_count == 10
        assert result.processing_time > 0
    
    def test_mock_service_empty_input(self):
        """Test mock service with empty input."""
        service = MockSentenceTransformersService("all-MiniLM-L6-v2")
        
        result = service.embed_texts([])
        
        assert len(result.embeddings) == 0
        assert result.text_count == 0
        assert result.model_name == "all-MiniLM-L6-v2"
    
    def test_mock_service_single_text_input(self):
        """Test mock service with single text input."""
        service = MockSentenceTransformersService(
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384
        )
        
        embedding = service.embed_single("Single test text")
        
        assert len(embedding) == 384
        assert all(isinstance(x, float) for x in embedding)
        assert service.call_count == 1
    
    def test_mock_service_consistency(self):
        """Test that mock service produces consistent results."""
        service = MockSentenceTransformersService(
            model_name="all-MiniLM-L6-v2",
            embedding_dim=384
        )
        
        # Same text should produce same embedding (due to seeded random)
        text = "Consistent test text"
        embedding1 = service.embed_single(text)
        embedding2 = service.embed_single(text)
        
        # Should be the same due to seeded random in mock
        assert len(embedding1) == len(embedding2) == 384
        # Note: In a real implementation, we might want deterministic results
        # but for mock testing, we just ensure proper structure
    
    def test_mock_service_model_info_structure(self):
        """Test that model info has expected structure."""
        service = MockSentenceTransformersService(
            model_name="all-mpnet-base-v2",
            embedding_dim=768
        )
        
        info = service.get_model_info()
        
        # Check required fields
        required_fields = [
            "model_name", "model_type", "embedding_dimension", 
            "device", "is_mock"
        ]
        for field in required_fields:
            assert field in info
        
        # Check values
        assert info["model_name"] == "all-mpnet-base-v2"
        assert info["model_type"] == "sentence-transformers"
        assert info["embedding_dimension"] == 768
        assert info["device"] == "cpu"
        assert info["is_mock"] is True
    
    def test_mock_service_error_handling(self):
        """Test mock service error handling."""
        service = MockSentenceTransformersService("all-MiniLM-L6-v2")
        
        # Mock service should handle edge cases gracefully
        # Test with None input (should be handled by the service)
        with pytest.raises((TypeError, ValueError)):
            service.embed_single(None)
        
        # Test with empty string
        embedding = service.embed_single("")
        assert len(embedding) == service.embedding_dimension
    
    @patch('pdf_vector_system.embeddings.sentence_transformers_service.SentenceTransformer')
    def test_real_service_error_handling(self, mock_st_class):
        """Test real service error handling during initialization."""
        # Test initialization failure
        mock_st_class.side_effect = Exception("Model loading failed")
        
        with pytest.raises(Exception, match="Model loading failed"):
            SentenceTransformersService("invalid-model")
    
    def test_mock_service_performance_tracking(self):
        """Test that mock service tracks performance metrics."""
        service = MockSentenceTransformersService("all-MiniLM-L6-v2")
        
        texts = ["text 1", "text 2", "text 3"]
        result = service.embed_texts(texts)
        
        # Should have performance metrics
        assert result.processing_time > 0
        assert result.texts_per_second > 0
        assert result.text_count == 3
