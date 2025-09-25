"""Tests for optional embedding providers."""

import pytest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

from pdf_vector_system.embeddings.base import EmbeddingResult, EmbeddingServiceError
from pdf_vector_system.config.settings import EmbeddingConfig, EmbeddingModelType


class TestCohereService:
    """Test Cohere embedding service."""
    
    @pytest.fixture
    def mock_cohere_client(self):
        """Mock Cohere client."""
        mock_client = Mock()
        mock_response = Mock()
        mock_response.embeddings = [np.random.rand(1024).tolist()]
        mock_client.embed.return_value = mock_response
        return mock_client
    
    @pytest.fixture
    def cohere_config(self):
        """Cohere embedding configuration."""
        return EmbeddingConfig(
            model_type=EmbeddingModelType.COHERE,
            model_name="embed-english-v3.0",
            api_key="test-cohere-key"
        )
    
    def test_cohere_service_import(self):
        """Test importing Cohere service."""
        try:
            from pdf_vector_system.embeddings.cohere_service import CohereEmbeddingService
            assert CohereEmbeddingService is not None
        except ImportError:
            pytest.skip("Cohere not available")
    
    @patch('pdf_vector_system.embeddings.cohere_service.cohere')
    def test_cohere_service_initialization(self, mock_cohere, cohere_config):
        """Test Cohere service initialization."""
        try:
            from pdf_vector_system.embeddings.cohere_service import CohereEmbeddingService
        except ImportError:
            pytest.skip("Cohere not available")
        
        mock_cohere.Client.return_value = Mock()
        
        service = CohereEmbeddingService(cohere_config)
        
        assert service.config == cohere_config
        assert service.model_name == "embed-english-v3.0"
        mock_cohere.Client.assert_called_once_with(api_key="test-cohere-key")
    
    @patch('pdf_vector_system.embeddings.cohere_service.cohere')
    def test_cohere_embed_single(self, mock_cohere, cohere_config, mock_cohere_client):
        """Test Cohere single embedding."""
        try:
            from pdf_vector_system.embeddings.cohere_service import CohereEmbeddingService
        except ImportError:
            pytest.skip("Cohere not available")
        
        mock_cohere.Client.return_value = mock_cohere_client
        
        service = CohereEmbeddingService(cohere_config)
        result = service.embed_single("test text")
        
        assert isinstance(result, list)
        assert len(result) == 1024
        mock_cohere_client.embed.assert_called_once()
    
    @patch('pdf_vector_system.embeddings.cohere_service.cohere')
    def test_cohere_embed_batch(self, mock_cohere, cohere_config, mock_cohere_client):
        """Test Cohere batch embedding."""
        try:
            from pdf_vector_system.embeddings.cohere_service import CohereEmbeddingService
        except ImportError:
            pytest.skip("Cohere not available")
        
        # Mock multiple embeddings
        mock_response = Mock()
        mock_response.embeddings = [
            np.random.rand(1024).tolist(),
            np.random.rand(1024).tolist()
        ]
        mock_cohere_client.embed.return_value = mock_response
        mock_cohere.Client.return_value = mock_cohere_client
        
        service = CohereEmbeddingService(cohere_config)
        texts = ["text 1", "text 2"]
        result = service.embed_batch(texts)
        
        assert isinstance(result, EmbeddingResult)
        assert len(result.embeddings) == 2
        assert all(len(emb) == 1024 for emb in result.embeddings)


class TestHuggingFaceService:
    """Test HuggingFace embedding service."""
    
    @pytest.fixture
    def huggingface_config(self):
        """HuggingFace embedding configuration."""
        return EmbeddingConfig(
            model_type=EmbeddingModelType.HUGGINGFACE,
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
    
    def test_huggingface_service_import(self):
        """Test importing HuggingFace service."""
        try:
            from pdf_vector_system.embeddings.huggingface_service import HuggingFaceEmbeddingService
            assert HuggingFaceEmbeddingService is not None
        except ImportError:
            pytest.skip("HuggingFace transformers not available")
    
    @patch('pdf_vector_system.embeddings.huggingface_service.AutoTokenizer')
    @patch('pdf_vector_system.embeddings.huggingface_service.AutoModel')
    def test_huggingface_service_initialization(self, mock_model, mock_tokenizer, huggingface_config):
        """Test HuggingFace service initialization."""
        try:
            from pdf_vector_system.embeddings.huggingface_service import HuggingFaceEmbeddingService
        except ImportError:
            pytest.skip("HuggingFace transformers not available")
        
        mock_tokenizer.from_pretrained.return_value = Mock()
        mock_model.from_pretrained.return_value = Mock()
        
        service = HuggingFaceEmbeddingService(huggingface_config)
        
        assert service.config == huggingface_config
        assert service.model_name == "sentence-transformers/all-MiniLM-L6-v2"
    
    @patch('pdf_vector_system.embeddings.huggingface_service.AutoTokenizer')
    @patch('pdf_vector_system.embeddings.huggingface_service.AutoModel')
    @patch('pdf_vector_system.embeddings.huggingface_service.torch')
    def test_huggingface_embed_single(self, mock_torch, mock_model_class, mock_tokenizer_class, huggingface_config):
        """Test HuggingFace single embedding."""
        try:
            from pdf_vector_system.embeddings.huggingface_service import HuggingFaceEmbeddingService
        except ImportError:
            pytest.skip("HuggingFace transformers not available")
        
        # Mock tokenizer
        mock_tokenizer = Mock()
        mock_tokenizer.return_value = {
            'input_ids': Mock(),
            'attention_mask': Mock()
        }
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer
        
        # Mock model
        mock_model = Mock()
        mock_outputs = Mock()
        mock_outputs.last_hidden_state = Mock()
        mock_outputs.last_hidden_state.mean.return_value = Mock()
        mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value = Mock()
        mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.detach.return_value = Mock()
        mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.detach.return_value.cpu.return_value = Mock()
        mock_outputs.last_hidden_state.mean.return_value.squeeze.return_value.detach.return_value.cpu.return_value.numpy.return_value = np.random.rand(384)
        mock_model.return_value = mock_outputs
        mock_model_class.from_pretrained.return_value = mock_model
        
        # Mock torch
        mock_torch.no_grad.return_value.__enter__ = Mock()
        mock_torch.no_grad.return_value.__exit__ = Mock()
        
        service = HuggingFaceEmbeddingService(huggingface_config)
        result = service.embed_single("test text")
        
        assert isinstance(result, list)
        assert len(result) == 384


class TestGoogleUSEService:
    """Test Google Universal Sentence Encoder service."""
    
    @pytest.fixture
    def google_use_config(self):
        """Google USE embedding configuration."""
        return EmbeddingConfig(
            model_type=EmbeddingModelType.GOOGLE_USE,
            model_name="universal-sentence-encoder"
        )
    
    def test_google_use_service_import(self):
        """Test importing Google USE service."""
        try:
            from pdf_vector_system.embeddings.google_use_service import GoogleUSEService
            assert GoogleUSEService is not None
        except ImportError:
            pytest.skip("TensorFlow/TensorFlow Hub not available")
    
    @patch('pdf_vector_system.embeddings.google_use_service.hub')
    def test_google_use_service_initialization(self, mock_hub, google_use_config):
        """Test Google USE service initialization."""
        try:
            from pdf_vector_system.embeddings.google_use_service import GoogleUSEService
        except ImportError:
            pytest.skip("TensorFlow/TensorFlow Hub not available")
        
        mock_model = Mock()
        mock_hub.load.return_value = mock_model
        
        service = GoogleUSEService(google_use_config)
        
        assert service.config == google_use_config
        mock_hub.load.assert_called_once()
    
    @patch('pdf_vector_system.embeddings.google_use_service.hub')
    def test_google_use_embed_single(self, mock_hub, google_use_config):
        """Test Google USE single embedding."""
        try:
            from pdf_vector_system.embeddings.google_use_service import GoogleUSEService
        except ImportError:
            pytest.skip("TensorFlow/TensorFlow Hub not available")
        
        mock_model = Mock()
        mock_embedding = Mock()
        mock_embedding.numpy.return_value = np.array([np.random.rand(512)])
        mock_model.return_value = mock_embedding
        mock_hub.load.return_value = mock_model
        
        service = GoogleUSEService(google_use_config)
        result = service.embed_single("test text")
        
        assert isinstance(result, list)
        assert len(result) == 512


class TestAzureOpenAIService:
    """Test Azure OpenAI embedding service."""
    
    @pytest.fixture
    def azure_openai_config(self):
        """Azure OpenAI embedding configuration."""
        return EmbeddingConfig(
            model_type=EmbeddingModelType.AZURE_OPENAI,
            model_name="text-embedding-ada-002",
            api_key="test-azure-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2023-05-15"
        )
    
    def test_azure_openai_service_import(self):
        """Test importing Azure OpenAI service."""
        try:
            from pdf_vector_system.embeddings.azure_openai_service import AzureOpenAIEmbeddingService
            assert AzureOpenAIEmbeddingService is not None
        except ImportError:
            pytest.skip("Azure OpenAI not available")
    
    @patch('pdf_vector_system.embeddings.azure_openai_service.AzureOpenAI')
    def test_azure_openai_service_initialization(self, mock_azure_openai, azure_openai_config):
        """Test Azure OpenAI service initialization."""
        try:
            from pdf_vector_system.embeddings.azure_openai_service import AzureOpenAIEmbeddingService
        except ImportError:
            pytest.skip("Azure OpenAI not available")
        
        mock_client = Mock()
        mock_azure_openai.return_value = mock_client
        
        service = AzureOpenAIEmbeddingService(azure_openai_config)
        
        assert service.config == azure_openai_config
        mock_azure_openai.assert_called_once_with(
            api_key="test-azure-key",
            azure_endpoint="https://test.openai.azure.com/",
            api_version="2023-05-15"
        )
    
    @patch('pdf_vector_system.embeddings.azure_openai_service.AzureOpenAI')
    def test_azure_openai_embed_single(self, mock_azure_openai, azure_openai_config):
        """Test Azure OpenAI single embedding."""
        try:
            from pdf_vector_system.embeddings.azure_openai_service import AzureOpenAIEmbeddingService
        except ImportError:
            pytest.skip("Azure OpenAI not available")
        
        # Mock client and response
        mock_client = Mock()
        mock_response = Mock()
        mock_response.data = [Mock()]
        mock_response.data[0].embedding = np.random.rand(1536).tolist()
        mock_client.embeddings.create.return_value = mock_response
        mock_azure_openai.return_value = mock_client
        
        service = AzureOpenAIEmbeddingService(azure_openai_config)
        result = service.embed_single("test text")
        
        assert isinstance(result, list)
        assert len(result) == 1536
        mock_client.embeddings.create.assert_called_once()


class TestProviderAvailability:
    """Test provider availability functions."""
    
    def test_get_available_providers(self):
        """Test getting available providers."""
        from pdf_vector_system.embeddings import get_available_providers
        
        providers = get_available_providers()
        
        # Core providers should always be available
        assert providers["sentence_transformers"] is True
        assert providers["openai"] is True
        
        # Optional providers depend on installed packages
        assert "cohere" in providers
        assert "huggingface" in providers
        assert "google_use" in providers
        assert "azure_openai" in providers
        assert "google_gemini" in providers
        
        # All values should be boolean
        for provider, available in providers.items():
            assert isinstance(available, bool)
    
    def test_check_provider_dependencies(self):
        """Test checking provider dependencies."""
        from pdf_vector_system.embeddings import check_provider_dependencies
        
        missing = check_provider_dependencies()
        
        # Should return installation commands for missing dependencies
        assert isinstance(missing, dict)
        
        # If a provider is missing, should have installation command
        for provider, command in missing.items():
            assert isinstance(command, str)
            assert "pip install" in command
    
    def test_provider_availability_consistency(self):
        """Test consistency between availability and dependencies."""
        from pdf_vector_system.embeddings import get_available_providers, check_provider_dependencies
        
        available = get_available_providers()
        missing = check_provider_dependencies()
        
        # If a provider is not available, it should have missing dependencies
        # (except for core providers which should always be available)
        core_providers = {"sentence_transformers", "openai"}
        
        for provider, is_available in available.items():
            if provider not in core_providers and not is_available:
                assert provider in missing, f"Provider {provider} not available but no missing dependencies listed"
