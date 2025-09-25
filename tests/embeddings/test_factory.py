"""Tests for EmbeddingServiceFactory and BatchEmbeddingProcessor."""

import pytest
from unittest.mock import Mock, patch, MagicMock

from pdf_vector_system.embeddings.factory import EmbeddingServiceFactory, BatchEmbeddingProcessor
from pdf_vector_system.embeddings.base import EmbeddingResult
from pdf_vector_system.config.settings import EmbeddingConfig, EmbeddingModelType
from tests.mocks.embedding_mocks import MockEmbeddingService


class TestEmbeddingServiceFactory:
    """Test EmbeddingServiceFactory."""
    
    def test_create_sentence_transformers_service(self):
        """Test creating sentence transformers service."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-MiniLM-L6-v2",
            batch_size=16
        )
        
        with patch('pdf_vector_system.embeddings.factory.SentenceTransformersService') as mock_class:
            mock_service = Mock()
            mock_class.return_value = mock_service
            
            service = EmbeddingServiceFactory.create_service(config)
            
            assert service == mock_service
            mock_class.assert_called_once_with(
                model_name="all-MiniLM-L6-v2",
                device=None,
                cache_folder=None,
                trust_remote_code=False
            )
    
    def test_create_openai_service(self):
        """Test creating OpenAI service."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-small",
            batch_size=100,
            openai_api_key="test-key"
        )
        
        with patch('pdf_vector_system.embeddings.factory.OpenAIEmbeddingService') as mock_class:
            mock_service = Mock()
            mock_class.return_value = mock_service
            
            service = EmbeddingServiceFactory.create_service(config)
            
            assert service == mock_service
            mock_class.assert_called_once_with(
                model_name="text-embedding-3-small",
                api_key="test-key",
                base_url=None,
                max_retries=3,
                timeout=60.0,
                batch_size=100
            )
    
    def test_unsupported_model_type(self):
        """Test creating service with unsupported model type."""
        config = EmbeddingConfig()
        config.model_type = "unsupported_type"  # Invalid type
        
        with pytest.raises(ValueError, match="Unsupported embedding model type"):
            EmbeddingServiceFactory.create_service(config)
    
    def test_get_supported_models(self):
        """Test getting supported models."""
        models = EmbeddingServiceFactory.get_supported_models()
        
        assert isinstance(models, dict)
        assert EmbeddingModelType.SENTENCE_TRANSFORMERS in models
        assert EmbeddingModelType.OPENAI in models
        
        # Check that each type has a list of models
        for model_type, model_list in models.items():
            assert isinstance(model_list, list)
            assert len(model_list) > 0
    
    def test_create_service_with_custom_parameters(self):
        """Test creating service with custom parameters."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.SENTENCE_TRANSFORMERS,
            model_name="all-mpnet-base-v2",
            batch_size=32,
            device="cuda",
            cache_folder="/custom/cache",
            trust_remote_code=True
        )
        
        with patch('pdf_vector_system.embeddings.factory.SentenceTransformersService') as mock_class:
            mock_service = Mock()
            mock_class.return_value = mock_service
            
            service = EmbeddingServiceFactory.create_service(config)
            
            mock_class.assert_called_once_with(
                model_name="all-mpnet-base-v2",
                device="cuda",
                cache_folder="/custom/cache",
                trust_remote_code=True
            )
    
    def test_create_openai_service_with_custom_url(self):
        """Test creating OpenAI service with custom base URL."""
        config = EmbeddingConfig(
            model_type=EmbeddingModelType.OPENAI,
            model_name="text-embedding-3-large",
            openai_api_key="test-key",
            openai_base_url="https://custom.openai.com/v1",
            max_retries=5,
            timeout_seconds=120
        )
        
        with patch('pdf_vector_system.embeddings.factory.OpenAIEmbeddingService') as mock_class:
            mock_service = Mock()
            mock_class.return_value = mock_service
            
            service = EmbeddingServiceFactory.create_service(config)
            
            mock_class.assert_called_once_with(
                model_name="text-embedding-3-large",
                api_key="test-key",
                base_url="https://custom.openai.com/v1",
                max_retries=5,
                timeout=120.0,
                batch_size=32  # default
            )


class TestBatchEmbeddingProcessor:
    """Test BatchEmbeddingProcessor."""
    
    def test_initialization(self):
        """Test batch processor initialization."""
        mock_service = MockEmbeddingService("test-model")
        processor = BatchEmbeddingProcessor(mock_service, max_workers=2)
        
        assert processor.embedding_service == mock_service
        assert processor.max_workers == 2
        assert processor.batch_size == 32  # Default from mock service
    
    def test_process_texts(self):
        """Test processing texts in batches."""
        mock_service = MockEmbeddingService("test-model", embedding_dim=3)
        processor = BatchEmbeddingProcessor(mock_service, max_workers=1)
        processor.batch_size = 2  # Small batch size for testing
        
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]
        result = processor.process_texts(texts, show_progress=False)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 5
        assert all(len(emb) == 3 for emb in result.embeddings)
        assert result.model_name == "test-model"
        assert result.embedding_dimension == 3
        
        # Should have processed in batches
        assert len(processor.processed_batches) == 3  # [2, 2, 1] texts per batch
    
    def test_process_texts_empty(self):
        """Test processing empty text list."""
        mock_service = MockEmbeddingService("test-model")
        processor = BatchEmbeddingProcessor(mock_service)
        
        result = processor.process_texts([], show_progress=False)
        
        assert len(result.embeddings) == 0
        assert result.model_name == "test-model"
    
    def test_process_texts_with_metadata(self):
        """Test processing texts with metadata."""
        mock_service = MockEmbeddingService("test-model")
        processor = BatchEmbeddingProcessor(mock_service)
        
        texts = ["text 1", "text 2"]
        metadata = {"source": "test", "batch_id": "test_batch"}
        
        result = processor.process_texts(texts, metadata=metadata, show_progress=False)
        
        assert result.metadata["batch_processor"] is True
        assert "texts_per_second" in result.metadata
    
    def test_create_batches(self):
        """Test batch creation."""
        mock_service = MockEmbeddingService("test-model")
        processor = BatchEmbeddingProcessor(mock_service)
        processor.batch_size = 3
        
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5"]
        batches = processor._create_batches(texts)
        
        assert len(batches) == 2
        assert len(batches[0]) == 3
        assert len(batches[1]) == 2
        assert batches[0] == ["text 1", "text 2", "text 3"]
        assert batches[1] == ["text 4", "text 5"]
    
    def test_create_batches_empty(self):
        """Test batch creation with empty texts."""
        mock_service = MockEmbeddingService("test-model")
        processor = BatchEmbeddingProcessor(mock_service)
        
        batches = processor._create_batches([])
        
        assert len(batches) == 0
    
    def test_process_single_batch(self):
        """Test processing a single batch."""
        mock_service = MockEmbeddingService("test-model", embedding_dim=5)
        processor = BatchEmbeddingProcessor(mock_service)
        
        texts = ["text 1", "text 2"]
        result = processor.process_texts(texts, show_progress=False)
        
        assert len(result.embeddings) == 2
        assert all(len(emb) == 5 for emb in result.embeddings)
        assert len(processor.processed_batches) == 1
    
    def test_batch_size_configuration(self):
        """Test batch size configuration."""
        mock_service = MockEmbeddingService("test-model")
        mock_service.batch_size = 50  # Custom batch size
        
        processor = BatchEmbeddingProcessor(mock_service)
        
        assert processor.batch_size == 50
    
    def test_parallel_processing(self):
        """Test parallel processing with multiple workers."""
        mock_service = MockEmbeddingService("test-model", embedding_dim=3)
        processor = BatchEmbeddingProcessor(mock_service, max_workers=2)
        processor.batch_size = 2
        
        texts = ["text 1", "text 2", "text 3", "text 4", "text 5", "text 6"]
        result = processor.process_texts(texts, show_progress=False)
        
        assert len(result.embeddings) == 6
        assert all(len(emb) == 3 for emb in result.embeddings)
        assert len(processor.processed_batches) == 3  # [2, 2, 2] texts per batch
    
    def test_performance_metrics(self):
        """Test that processor tracks performance metrics."""
        mock_service = MockEmbeddingService("test-model")
        processor = BatchEmbeddingProcessor(mock_service)
        
        texts = ["text 1", "text 2", "text 3"]
        result = processor.process_texts(texts, show_progress=False)
        
        # Should have performance metrics
        assert result.processing_time > 0
        assert result.texts_per_second > 0
        assert "batch_processor" in result.metadata
        assert result.metadata["batch_processor"] is True
